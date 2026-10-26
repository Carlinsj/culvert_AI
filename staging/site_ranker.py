"""Separate fixed experimental physical-site ranker, never a frozen-model update."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "distance_to_nearest_stream_m",
    "upstream_area_m2",
    "flow_accumulation_cells",
    "slope_degrees",
    "negative_tpi_9x9_m",
    "valley_depth_9x9_m",
    "negative_tpi_31x31_m",
    "valley_depth_31x31_m",
    "elevation_relief_9x9_m",
    "terrain_roughness_9x9_m",
    "route_elevation_dip_60m",
    "route_low_point_score",
)


def training_exclusion(sites, siblings, heldout_roads, purge_m=1000):
    """Exclude complete corridors and ALL sibling labels near held-out roads."""
    if purge_m < 1000 or heldout_roads.empty:
        raise ValueError("A nonempty whole-road holdout and >=1000 m purge are required")
    if any(
        frame.crs is None or frame.crs.to_epsg() != 32618
        for frame in (sites, siblings, heldout_roads)
    ):
        raise ValueError("Native EPSG:32618 metre geometry required for exclusions")

    def distances(frame):
        (left, _), distance = heldout_roads.sindex.nearest(
            frame.geometry, return_all=False, return_distance=True
        )
        values = np.full(len(frame), np.inf)
        values[left] = distance
        return values

    distance = distances(sites)
    original_distance = distances(siblings)
    corridors = set(heldout_roads.heldout_corridor.dropna().astype(str)) - {""}
    purged = set(siblings.loc[original_distance <= purge_m, "site_key"])
    purged.update(siblings.loc[siblings.route.astype(str).isin(corridors), "site_key"])
    keep = (distance > purge_m) & ~sites.site_key.isin(purged).to_numpy()
    return keep, distance, purged


def features(frame):
    """No geography, source/label metadata, inventory proximity or exposed score."""
    values = frame.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    return np.log1p(values.clip(lower=0)).replace([np.inf, -np.inf], np.nan)


def train(sites):
    labels = sites.target.astype(int)
    if set(labels) != {0, 1}:
        raise ValueError("Both confirmed positive and verified negative sites required")
    valid = sites.label_source.eq("field_verified_negative") & labels.eq(0)
    valid |= sites.label_source.eq("authoritative_inventory_positive_20m") & labels.eq(1)
    if not valid.all():
        raise ValueError("Unvisited and unmatched candidates are not negative labels")
    if not sites.site_key.is_unique:
        raise ValueError("One training example per documented physical source site required")
    matrix = features(sites)
    missing_by_class = matrix.isna().groupby(labels).mean()
    columns = [
        c
        for c in FEATURES
        if matrix[c].notna().sum() >= 10
        and abs(missing_by_class.loc[0, c] - missing_by_class.loc[1, c]) <= 0.2
    ]
    if len(columns) < 3 or labels.value_counts().min() < 10:
        raise ValueError("Insufficient compatible features or verified sites")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=20261016
        ),
    )
    model.fit(matrix[columns], labels)
    return dict(
        model=model,
        features=columns,
        version="physical-site-logistic-v1",
        experimental=True,
        calibrated=False,
    )


def score(bundle, frame):
    if bundle.get("version") != "physical-site-logistic-v1" or not bundle.get("experimental"):
        raise ValueError("Separate experimental site bundle required")
    # Logistic output is used for ordering only; sampling is not population prevalence.
    return bundle["model"].predict_proba(features(frame)[bundle["features"]])[:, 1]
