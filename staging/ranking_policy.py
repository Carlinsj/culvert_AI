"""Fixed-policy staging adapter; no fitting, candidate generation or app hooks."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from culvert_ai.flow_c_bundle import sha256_file
from culvert_ai.inference import (
    RouteZConfig, dynamic_route_z, rank_flow_c_candidates, score_frozen_flow_c,
)

WARNING = "Recommendations require field verification."
SCORE_INTERPRETATION = "Relative ranking score; not a calibrated probability."
FROZEN_MODEL_SHA256 = "bdefae5bc12723e17487001826eea706fb48065c33ec1b13b15e892ad7347486"


class RankingPolicy(str, Enum):
    CURRENT = "dynamic_route_z"
    CHALLENGER = "raw_rf"


def load_policy(path: str | Path) -> RankingPolicy:
    config = json.loads(Path(path).read_text())
    if set(config) != {"environment", "ranking_policy"} or config["environment"] != "staging":
        raise ValueError("Only an explicit staging ranking-policy configuration is accepted.")
    return RankingPolicy(config["ranking_policy"])


def rank_candidates(
    scored: gpd.GeoDataFrame,
    *,
    policy: RankingPolicy | str = RankingPolicy.CURRENT,
    route_z_config: RouteZConfig | None = None,
) -> gpd.GeoDataFrame:
    """Use the identical candidate population and existing score-ordered 20 m NMS.

    Raw ordering uses an identity-score adapter to the *unchanged* production
    consolidator: no route can reach either normalization count threshold, and
    the fallback (score - 0) / 1 is exactly the raw score. Actual ROUTE-Z
    diagnostics are then restored; they do not affect raw representative choice.
    This avoids a second implementation of the frozen physical-site rule.
    """
    policy = RankingPolicy(policy)  # Invalid policies fail closed.
    if scored.empty:
        raise ValueError("At least one scored candidate is required.")
    required = {"candidate_id", "route_id", "flow_c_score"}
    if not required.issubset(scored.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(scored.columns))}")
    if scored.crs is None or scored.crs.to_epsg() != 32618:
        raise ValueError("Staging candidates must use native EPSG:32618 geometry.")
    if scored.candidate_id.isna().any() or not scored.candidate_id.map(lambda x: isinstance(x, str)).all():
        raise ValueError("Candidate IDs must be non-null strings.")
    if scored.candidate_id.duplicated().any():
        raise ValueError("Candidate IDs must be unique.")
    if not np.isfinite(pd.to_numeric(scored.flow_c_score, errors="coerce")).all():
        raise ValueError("RF scores must all be finite.")
    if (scored.geometry.isna().any() or scored.geometry.is_empty.any()
            or not scored.geometry.geom_type.eq("Point").all()):
        raise ValueError("Candidate geometry must contain nonempty points.")
    xy = np.c_[scored.geometry.x, scored.geometry.y]
    if not np.isfinite(xy).all():
        raise ValueError("Candidate point coordinates must be finite.")
    # Unique index prevents duplicate caller indices from corrupting group statistics.
    frame = scored.copy().reset_index(drop=True)
    frame["flow_c_score"] = pd.to_numeric(frame.flow_c_score).astype(float)
    if policy is RankingPolicy.CURRENT:
        sites = rank_flow_c_candidates(frame, route_z_config=route_z_config)
    else:
        identity = RouteZConfig(
            minimum_standard_candidates=len(frame) + 1,
            minimum_robust_candidates=len(frame) + 1,
            global_center=0.0, global_scale=1.0,
        )
        sites = rank_flow_c_candidates(frame, route_z_config=identity)
        diagnostics = dynamic_route_z(frame, config=route_z_config).set_index("candidate_id")
        for column in ("route_z_score", "route_z_method", "route_z_candidate_count",
                       "route_z_center", "route_z_scale"):
            sites[column] = sites.candidate_id.map(diagnostics[column])
        sites["final_priority"] = sites.flow_c_score
    sites["ranking_policy"] = policy.value
    sites["environment"] = "staging"
    sites["score_interpretation"] = SCORE_INTERPRETATION
    sites["field_verification_notice"] = WARNING
    # Production physical_site_id is rank-based and can refer to another location
    # after switching policies. Do not reuse it as an inspection-status join key.
    sites["staging_site_key"] = [
        "staging:" + policy.value + ":" + hashlib.sha256(cid.encode()).hexdigest()
        for cid in sites.candidate_id
    ]
    return sites


def predict_from_features(features, model_bundle, *, policy=RankingPolicy.CURRENT):
    """Use the existing frozen 48-field preprocessing/scorer without any fitting."""
    if sha256_file(Path(model_bundle)) != FROZEN_MODEL_SHA256:
        raise ValueError("Staging requires the exact frozen RF artifact, not a replacement model.")
    scored, bundle = score_frozen_flow_c(features, model_bundle)
    reference = bundle["route_z_global"]
    return rank_candidates(scored, policy=policy, route_z_config=RouteZConfig(
        global_center=reference["center"], global_scale=reference["scale"],
    ))


EXPORT_COLUMNS = (
    "global_rank", "route_id", "latitude", "longitude", "candidate_id",
    "physical_site_id", "staging_site_key", "flow_c_score", "route_z_score",
    "final_priority", "priority_percentile", "route_local_rank", "ranking_policy",
    "environment", "score_interpretation", "field_verification_notice",
)


def export_sites(sites, directory: str | Path):
    """Create a new staging-only CSV/GeoJSON directory; never overwrite outputs.

    Allowlisting prevents training labels and authoritative truth from leaking
    into field exports. GeoJSON geometry uses WGS84, CSV includes WGS84 columns.
    """
    directory = Path(directory)
    if not sites.environment.eq("staging").all():
        raise ValueError("Only staging results may be exported here.")
    clean = sites.loc[:, [*EXPORT_COLUMNS, "geometry"]].copy()
    directory.mkdir(parents=True, exist_ok=False)
    clean.drop(columns="geometry").to_csv(directory / "sites.csv", index=False)
    (directory / "sites.geojson").write_text(clean.to_crs(4326).to_json(drop_id=True) + "\n")
