import geopandas as gpd
from shapely.geometry import LineString, Point

from culvert_ai.io import write_vector
from culvert_ai.point_analysis import analyze_extracted_points


def test_analyze_extracted_points_matches_nearby_candidate(tmp_path):
    points = gpd.GeoDataFrame(
        [{"latitude": 42.0, "longitude": -74.0, "geometry": Point(-74.0, 42.0)}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    roads = gpd.GeoDataFrame(
        [{"FULLNAME": "Test Rd", "geometry": LineString([(-74.001, 42.0), (-73.999, 42.0)])}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    streams = gpd.GeoDataFrame(
        [{"FULLNAME": "Test Stream", "geometry": LineString([(-74.0, 41.999), (-74.0, 42.001)])}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    candidates = gpd.GeoDataFrame(
        [
            {
                "candidate_id": "cand-1",
                "discovery_score": 91.0,
                "discovery_status": "known_field_match",
                "geometry": Point(-74.0001, 42.0001),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    points_path = tmp_path / "points.gpkg"
    roads_path = tmp_path / "roads.gpkg"
    streams_path = tmp_path / "streams.gpkg"
    candidates_path = tmp_path / "candidates.gpkg"
    write_vector(points, points_path)
    write_vector(roads, roads_path)
    write_vector(streams, streams_path)
    write_vector(candidates, candidates_path)

    result = analyze_extracted_points(
        points_path=points_path,
        roads_path=roads_path,
        streams_path=streams_path,
        candidates_path=candidates_path,
        output_geojson=tmp_path / "analysis.geojson",
        output_csv=tmp_path / "analysis.csv",
        output_json=tmp_path / "analysis.json",
        output_markdown=tmp_path / "analysis.md",
    )

    assert result["rows"] == 1
    assert result["matched_existing_candidates"] == 1
    assert result["outside_analysis_extent"] == 0
    assert (tmp_path / "analysis.md").exists()
