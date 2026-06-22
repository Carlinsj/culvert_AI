import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from culvert_ai.io import write_vector
from culvert_ai.point_analysis import analyze_extracted_points, write_high_confidence_training_points


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


def test_write_high_confidence_training_points_filters_ambiguous_points(tmp_path):
    analysis = gpd.GeoDataFrame(
        [
            {
                "point_id": "pt_0001",
                "latitude": 42.0,
                "longitude": -74.0,
                "inside_analysis_extent": True,
                "analysis_flag": "matched_existing_candidate",
                "nearest_candidate_distance_m": 12.0,
                "geometry": Point(-74.0, 42.0),
            },
            {
                "point_id": "pt_0002",
                "latitude": 42.1,
                "longitude": -74.1,
                "inside_analysis_extent": False,
                "analysis_flag": "matched_existing_candidate",
                "nearest_candidate_distance_m": 10.0,
                "geometry": Point(-74.1, 42.1),
            },
            {
                "point_id": "pt_0003",
                "latitude": 42.2,
                "longitude": -74.2,
                "inside_analysis_extent": True,
                "analysis_flag": "needs_manual_review",
                "nearest_candidate_distance_m": 250.0,
                "geometry": Point(-74.2, 42.2),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    analysis_path = tmp_path / "analysis.geojson"
    output_path = tmp_path / "training.gpkg"
    csv_path = tmp_path / "training.csv"
    write_vector(analysis, analysis_path)

    result = write_high_confidence_training_points(
        analysis_path=analysis_path,
        output_path=output_path,
        csv_output=csv_path,
    )

    training = gpd.read_file(output_path)
    assert result["rows"] == 1
    assert result["rejected_rows"] == 2
    assert training["point_id"].tolist() == ["pt_0001"]
    assert training["label_source"].tolist() == ["field_report_coordinate_geospatial_qc"]
    assert csv_path.exists()


def test_analyze_extracted_points_uses_boundary_for_extent(tmp_path):
    points = gpd.GeoDataFrame(
        [
            {"latitude": 42.0, "longitude": -74.0, "geometry": Point(-74.0, 42.0)},
            {"latitude": 42.2, "longitude": -74.2, "geometry": Point(-74.2, 42.2)},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundary = gpd.GeoDataFrame(
        [{"geometry": Polygon([(-74.05, 41.95), (-73.95, 41.95), (-73.95, 42.05), (-74.05, 42.05)])}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    points_path = tmp_path / "points.gpkg"
    boundary_path = tmp_path / "boundary.gpkg"
    analysis_path = tmp_path / "analysis.geojson"
    write_vector(points, points_path)
    write_vector(boundary, boundary_path)

    result = analyze_extracted_points(
        points_path=points_path,
        boundary_path=boundary_path,
        output_geojson=analysis_path,
        output_csv=tmp_path / "analysis.csv",
        output_json=tmp_path / "analysis.json",
        output_markdown=tmp_path / "analysis.md",
    )

    analysis = gpd.read_file(analysis_path).sort_values("latitude").reset_index(drop=True)
    assert result["outside_analysis_extent"] == 1
    assert analysis["inside_analysis_extent"].tolist() == [True, False]
