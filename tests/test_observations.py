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
    write_vector(observations, observations_path)
    write_vector(base_known, base_path)

    result = merge_confirmed_observations(
        observations_path=observations_path,
        base_known_path=base_path,
        output_path=output_path,
    )
    combined = read_vector(output_path)

    assert result["confirmed_added"] == 1
    assert result["denied_saved_for_review"] == 1
    assert len(combined) == 2
    assert "FC-20260617-ABCD" in set(combined["culvert_id"])
    assert "nearest_map_candidate" in set(combined["layout_source"])
    assert "cand-2" not in set(combined["culvert_id"])
