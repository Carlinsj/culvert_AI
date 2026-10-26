import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from staging.model_sprint import (
    PHYSICAL_SIGNALS,
    build_candidate_sites,
    group_extrema,
    multiscale_extrema,
)


def raw_probes(values, road="a"):
    rows = []
    for i, value in enumerate(values):
        row = dict(
            candidate_id=f"c{i:03}", road_id=road, road_part_index=0,
            road_chainage_m=i * 10.0, proposal_kind="internal_road_probe",
            negative_tpi_9x9_m=max(float(value), 0.0), geometry=Point(i * 10.0, 0),
        )
        row.update({name: float(value) for name in PHYSICAL_SIGNALS})
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=32618)


def test_multiscale_extrema_keeps_missing_as_missing_and_stable_plateau():
    values = np.zeros(30)
    values[12:14] = 2
    values[20] = np.nan
    result = multiscale_extrema(raw_probes(values))
    assert "c012" in set(result.candidate_id)
    assert "c013" not in set(result.candidate_id)
    assert "c020" not in set(result.candidate_id)


def test_grouping_is_road_aware_and_reversible():
    left = multiscale_extrema(raw_probes([0] * 12 + [2] + [0] * 17, "a"))
    right = left.copy()
    right["road_id"] = "b"
    right["candidate_id"] = "b_" + right.candidate_id
    grouped = group_extrema(gpd.GeoDataFrame(
        __import__("pandas").concat([left, right], ignore_index=True), geometry="geometry", crs=32618
    ))
    assert len(grouped) == 2
    assert grouped.member_count.eq(1).all()


def test_candidate_union_never_discards_a_baseline_anchor():
    raw = raw_probes([0] * 30)
    hybrid = raw.iloc[[0]].copy()
    hybrid["candidate_id"] = "hybrid"
    baseline = raw.iloc[[1]].copy()
    baseline["candidate_id"] = "baseline"
    sites, _ = build_candidate_sites(hybrid, raw, baseline)
    assert set(sites.candidate_id) == {"baseline", "hybrid"}
    assert sites.candidate_id.is_unique
