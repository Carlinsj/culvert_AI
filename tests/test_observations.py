import geopandas as gpd
from shapely.geometry import Point

from culvert_ai.io import read_vector, write_vector
from culvert_ai.observations import merge_confirmed_observations


def test_merge_confirmed_observations_adds_confirmed_points(tmp_path):
    observations = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-confirmed",
                "observed_at": "2026-06-14T12:00:00Z",
                "status": "confirmed_culvert",
                "candidate_id": "cand-1",
                "field_culvert_id": "FC-20260617-ABCD",
                "layout_source": "nearest_map_candidate",
                "road_name": "State Rte 28",
                "notes": "pipe visible",
                "geometry": Point(-74.1, 42.0),
            },
            {
                "observation_id": "obs-denied",
                "observed_at": "2026-06-14T12:05:00Z",
                "status": "no_culvert",
                "candidate_id": "cand-2",
                "road_name": "State Rte 28",
                "notes": "no crossing structure",
                "geometry": Point(-74.2, 42.1),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    base_known = gpd.GeoDataFrame(
        [
            {
                "culvert_id": "existing",
                "route": "State Rte 32",
                "report_date": "2026-06-01",
                "geometry": Point(-74.3, 42.2),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    observations_path = tmp_path / "field_observations.geojson"
    base_path = tmp_path / "base_known.gpkg"
    output_path = tmp_path / "combined_known.gpkg"
    confirmed_path = tmp_path / "confirmed.gpkg"
    denied_path = tmp_path / "denied.gpkg"
    write_vector(observations, observations_path)
    write_vector(base_known, base_path)

    result = merge_confirmed_observations(
        observations_path=observations_path,
        base_known_path=base_path,
        output_path=output_path,
        confirmed_output_path=confirmed_path,
        denied_output_path=denied_path,
    )
    combined = read_vector(output_path)
    confirmed = read_vector(confirmed_path)
    denied = read_vector(denied_path)

    assert result["confirmed_added"] == 1
    assert result["denied_saved_for_review"] == 1
    assert len(combined) == 2
    assert len(confirmed) == 1
    assert len(denied) == 1
    assert "FC-20260617-ABCD" in set(combined["culvert_id"])
    assert "nearest_map_candidate" in set(combined["layout_source"])
    assert "cand-2" not in set(combined["culvert_id"])
    assert denied.iloc[0]["label"] == "no_culvert"
    assert denied.iloc[0]["candidate_id"] == "cand-2"


def test_merge_observations_can_exclude_confirmed_user_points(tmp_path):
    observations = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-confirmed",
                "observed_at": "2026-06-14T12:00:00Z",
                "status": "confirmed_culvert",
                "candidate_id": "user-added",
                "road_name": "State Rte 28",
                "geometry": Point(-74.1, 42.0),
            },
            {
                "observation_id": "obs-denied",
                "observed_at": "2026-06-14T12:05:00Z",
                "status": "no_culvert",
                "candidate_id": "bad-prediction",
                "road_name": "State Rte 28",
                "geometry": Point(-74.2, 42.1),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    base_known = gpd.GeoDataFrame(
        [
            {
                "culvert_id": "report-point",
                "source_file": "team_report.pdf",
                "geometry": Point(-74.3, 42.2),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    observations_path = tmp_path / "field_observations.geojson"
    base_path = tmp_path / "base_known.gpkg"
    output_path = tmp_path / "combined_known.gpkg"
    denied_path = tmp_path / "denied.gpkg"
    write_vector(observations, observations_path)
    write_vector(base_known, base_path)

    result = merge_confirmed_observations(
        observations_path=observations_path,
        base_known_path=base_path,
        output_path=output_path,
        denied_output_path=denied_path,
        include_confirmed=False,
    )
    combined = read_vector(output_path)
    denied = read_vector(denied_path)

    assert result["confirmed_added"] == 0
    assert result["denied_saved_for_review"] == 1
    assert combined["culvert_id"].tolist() == ["report-point"]
    assert denied["candidate_id"].tolist() == ["bad-prediction"]


def test_merge_observations_turns_far_confirmed_points_into_missed_prediction_labels(tmp_path):
    observations = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-miss",
                "observed_at": "2026-06-14T12:00:00Z",
                "status": "confirmed_culvert",
                "candidate_id": "field-added",
                "field_culvert_id": "ABU-1",
                "missed_candidate_id": "cand_000001",
                "missed_candidate_distance_m": 100.0,
                "nearest_candidate_id": "cand_000001",
                "nearest_candidate_distance_m": 100.0,
                "road_name": "State Rte 28",
                "geometry": Point(-74.1, 42.0),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    observations_path = tmp_path / "field_observations.geojson"
    output_path = tmp_path / "combined_known.gpkg"
    denied_path = tmp_path / "denied.gpkg"
    write_vector(observations, observations_path)

    result = merge_confirmed_observations(
        observations_path=observations_path,
        output_path=output_path,
        denied_output_path=denied_path,
        include_confirmed=False,
        miss_threshold_m=10,
    )
    denied = read_vector(denied_path)

    assert result["confirmed_added"] == 0
    assert result["denied_saved_for_review"] == 1
    assert denied.iloc[0]["label"] == "missed_prediction"
    assert denied.iloc[0]["candidate_id"] == "cand_000001"
    assert denied.iloc[0]["miss_distance_m"] == 100.0


def test_merge_confirmed_observations_deduplicates_repeated_field_ids(tmp_path):
    observations = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-first",
                "observed_at": "2026-06-24T12:38:33Z",
                "status": "confirmed_culvert",
                "field_culvert_id": "FC-20260624-0FTX",
                "missed_candidate_id": "cand_000001",
                "missed_candidate_distance_m": 100.0,
                "geometry": Point(-73.94409298896791, 42.08147275026664),
            },
            {
                "observation_id": "obs-second",
                "observed_at": "2026-06-24T12:38:17Z",
                "status": "confirmed_culvert",
                "field_culvert_id": "FC-20260624-0FTX",
                "missed_candidate_id": "cand_000001",
                "missed_candidate_distance_m": 100.0,
                "geometry": Point(-73.94409298896791, 42.08147275026664),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    observations_path = tmp_path / "field_observations.geojson"
    output_path = tmp_path / "combined_known.gpkg"
    confirmed_path = tmp_path / "confirmed.gpkg"
    denied_path = tmp_path / "denied.gpkg"
    write_vector(observations, observations_path)

    result = merge_confirmed_observations(
        observations_path=observations_path,
        output_path=output_path,
        confirmed_output_path=confirmed_path,
        denied_output_path=denied_path,
        include_confirmed=True,
    )
    confirmed = read_vector(confirmed_path)
    denied = read_vector(denied_path)

    assert result["confirmed_added"] == 1
    assert result["denied_saved_for_review"] == 1
    assert confirmed["field_culvert_id"].tolist() == ["FC-20260624-0FTX"]
    assert denied["field_culvert_id"].tolist() == ["FC-20260624-0FTX"]


def test_merge_observations_ignores_missed_field_added_candidate_ids(tmp_path):
    observations = gpd.GeoDataFrame(
        [
            {
                "observation_id": "obs-miss",
                "observed_at": "2026-06-24T12:00:00Z",
                "status": "confirmed_culvert",
                "field_culvert_id": "FC-20260624-NEW1",
                "missed_candidate_id": "FC-20260624-OLD1",
                "missed_candidate_distance_m": 100.0,
                "geometry": Point(-73.94, 42.08),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    observations_path = tmp_path / "field_observations.geojson"
    output_path = tmp_path / "combined_known.gpkg"
    denied_path = tmp_path / "denied.gpkg"
    write_vector(observations, observations_path)

    result = merge_confirmed_observations(
        observations_path=observations_path,
        output_path=output_path,
        denied_output_path=denied_path,
        include_confirmed=True,
    )

    assert result["confirmed_added"] == 1
    assert result["denied_saved_for_review"] == 0
