from __future__ import annotations

import argparse
import json
from pathlib import Path

from culvert_ai.candidates import (
    CandidateSettings,
    generate_candidates,
    generate_road_route_candidates,
    merge_candidate_layers,
)
from culvert_ai.census import (
    DEFAULT_COUNTY_BOUNDARY_URL,
    DEFAULT_TIGER_YEAR,
    download_ulster_census_inputs,
)
from culvert_ai.demo import create_demo_dataset, run_demo_pipeline
from culvert_ai.evaluate import evaluate_predictions
from culvert_ai.features import build_feature_table
from culvert_ai.field_reports import append_field_report_candidates, import_field_reports
from culvert_ai.io import read_vector, write_vector
from culvert_ai.llm_review import import_llm_reviewed_labels, write_llm_label_review_queue
from culvert_ai.model import predict_culvert_probability, train_model
from culvert_ai.observations import merge_confirmed_observations
from culvert_ai.osm import (
    DEFAULT_CENSUS_COUNTY_URL,
    DEFAULT_OVERPASS_URL,
    download_ulster_osm_inputs,
)
from culvert_ai.point_analysis import (
    analyze_extracted_points,
    write_high_confidence_training_points,
    write_point_only_layer,
)
from culvert_ai.region import filter_to_region, get_region, write_region_boundary
from culvert_ai.scoring import (
    build_discovery_ranking,
    score_unlabeled_candidates,
    write_google_earth_kml,
)
from culvert_ai.web_export import export_web_data


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=_json_default))
    return 0


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="culvert-ai",
        description="Predict likely culvert locations from roads, streams, topography, and known culverts.",
    )
    subparsers = parser.add_subparsers(required=True)

    demo_data = subparsers.add_parser("make-demo-data", help="Create synthetic demo GIS files.")
    demo_data.add_argument("--output-dir", default="data/ulster_demo", help="Directory for demo files.")
    demo_data.set_defaults(func=_make_demo_data)

    run_demo = subparsers.add_parser("run-demo", help="Run the full synthetic demo pipeline.")
    run_demo.add_argument("--output-dir", default="data/ulster_demo", help="Directory for demo outputs.")
    run_demo.set_defaults(func=_run_demo)

    download_osm = subparsers.add_parser(
        "download-osm",
        help="Download actual OpenStreetMap roads and waterways for Ulster County.",
    )
    download_osm.add_argument("--output-dir", default="data/raw", help="Output directory.")
    download_osm.add_argument("--county", default="Ulster County")
    download_osm.add_argument("--state", default="New York")
    download_osm.add_argument("--statefp", default="36")
    download_osm.add_argument("--countyfp", default="111")
    download_osm.add_argument("--overpass-url", default=DEFAULT_OVERPASS_URL)
    download_osm.add_argument("--census-county-url", default=DEFAULT_CENSUS_COUNTY_URL)
    download_osm.add_argument("--timeout-seconds", type=int, default=240)
    download_osm.add_argument("--tile-size-degrees", type=float, default=0.18)
    download_osm.set_defaults(func=_download_osm)

    download_census = subparsers.add_parser(
        "download-census",
        help="Download actual Census TIGER/Line roads and linear water for Ulster County.",
    )
    download_census.add_argument("--output-dir", default="data/raw", help="Output directory.")
    download_census.add_argument("--tiger-year", default=DEFAULT_TIGER_YEAR)
    download_census.add_argument("--statefp", default="36")
    download_census.add_argument("--countyfp", default="111")
    download_census.add_argument("--county-boundary-url", default=DEFAULT_COUNTY_BOUNDARY_URL)
    download_census.set_defaults(func=_download_census)

    import_reports = subparsers.add_parser(
        "import-field-reports",
        help="Extract field-observed culvert coordinates from DOCX/PDF daily reports.",
    )
    import_reports.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more field report ZIPs, folders, PDFs, or DOCX files.",
    )
    import_reports.add_argument(
        "--output",
        default="data/processed/field_report_culverts.gpkg",
        help="Output point layer.",
    )
    import_reports.add_argument(
        "--csv-output",
        default="data/processed/field_report_culverts.csv",
        help="Optional CSV output.",
    )
    import_reports.add_argument("--dedupe-precision", type=int, default=6)
    import_reports.set_defaults(func=_import_field_reports)

    llm_review_queue = subparsers.add_parser(
        "prepare-llm-label-review",
        help="Write a JSONL queue for LLM-assisted validation of extracted field-report labels.",
    )
    llm_review_queue.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more field report ZIPs, folders, PDFs, or DOCX files.",
    )
    llm_review_queue.add_argument(
        "--output",
        default="data/processed/field_report_llm_review_queue.jsonl",
        help="Output JSONL review queue.",
    )
    llm_review_queue.add_argument("--dedupe-precision", type=int, default=6)
    llm_review_queue.set_defaults(func=_prepare_llm_label_review)

    import_llm_review = subparsers.add_parser(
        "import-llm-reviewed-labels",
        help="Convert accepted LLM-reviewed JSONL labels into a known culvert point layer.",
    )
    import_llm_review.add_argument("--input", required=True, help="Reviewed JSONL file.")
    import_llm_review.add_argument(
        "--output",
        default="data/processed/field_report_llm_reviewed_culverts.gpkg",
        help="Output point layer.",
    )
    import_llm_review.add_argument(
        "--csv-output",
        default="data/processed/field_report_llm_reviewed_culverts.csv",
        help="Optional CSV output.",
    )
    import_llm_review.set_defaults(func=_import_llm_reviewed_labels)

    point_only = subparsers.add_parser(
        "extract-points-only",
        help="Write a point-only layer from extracted coordinate records.",
    )
    point_only.add_argument(
        "--points",
        default="data/processed/field_report_culverts.gpkg",
        help="Input extracted coordinate point layer.",
    )
    point_only.add_argument(
        "--output",
        default="data/processed/extracted_points_only.geojson",
        help="Output point-only GeoJSON or GPKG.",
    )
    point_only.add_argument(
        "--csv-output",
        default="data/processed/extracted_points_only.csv",
        help="Optional point-only CSV output.",
    )
    point_only.set_defaults(func=_extract_points_only)

    analyze_points = subparsers.add_parser(
        "analyze-extracted-points",
        help="Analyze extracted coordinates against roads, streams, and model candidates.",
    )
    analyze_points.add_argument(
        "--points",
        default="data/processed/field_report_culverts.gpkg",
        help="Input extracted coordinate point layer.",
    )
    analyze_points.add_argument("--roads", default="data/raw/roads.gpkg")
    analyze_points.add_argument("--streams", default="data/raw/streams.gpkg")
    analyze_points.add_argument(
        "--candidates",
        default="data/processed/actual_ulster_discovery_predictions.gpkg",
    )
    analyze_points.add_argument("--boundary", help="Optional boundary used to accept/reject points.")
    analyze_points.add_argument(
        "--output-geojson",
        default="data/processed/extracted_points_analysis.geojson",
    )
    analyze_points.add_argument(
        "--output-csv",
        default="data/processed/extracted_points_analysis.csv",
    )
    analyze_points.add_argument(
        "--output-json",
        default="reports/extracted_points_analysis.json",
    )
    analyze_points.add_argument(
        "--output-markdown",
        default="/private/tmp/culvert_extracted_points_analysis.md",
    )
    analyze_points.add_argument("--match-radius-m", type=float, default=75.0)
    analyze_points.add_argument("--cluster-radius-m", type=float, default=750.0)
    analyze_points.set_defaults(func=_analyze_extracted_points)

    training_points = subparsers.add_parser(
        "build-high-confidence-training-points",
        help="Filter extracted point analysis into a high-confidence training point layer.",
    )
    training_points.add_argument(
        "--analysis",
        default="data/processed/extracted_points_analysis.geojson",
        help="Input extracted point analysis layer.",
    )
    training_points.add_argument(
        "--output",
        default="data/processed/high_confidence_training_points.gpkg",
        help="Output point layer for supervised training.",
    )
    training_points.add_argument(
        "--csv-output",
        default="data/processed/high_confidence_training_points.csv",
        help="Optional CSV output.",
    )
    training_points.add_argument(
        "--accepted-flag",
        action="append",
        dest="accepted_flags",
        help="Analysis flag to accept for training. Repeat to accept multiple flags.",
    )
    training_points.set_defaults(func=_build_high_confidence_training_points)

    merge_observations = subparsers.add_parser(
        "merge-field-observations",
        help="Merge confirmed dashboard observations into the known culvert training layer.",
    )
    merge_observations.add_argument(
        "--observations",
        default="data/processed/field_observations.geojson",
        help="Dashboard field observations GeoJSON.",
    )
    merge_observations.add_argument("--base-known", help="Existing known culvert point layer.")
    merge_observations.add_argument(
        "--output",
        default="data/processed/training_known_culverts.gpkg",
        help="Combined known culvert layer for training.",
    )
    merge_observations.add_argument(
        "--csv-output",
        default="data/processed/training_known_culverts.csv",
        help="Optional CSV export.",
    )
    merge_observations.add_argument(
        "--confirmed-output",
        help="Optional point layer containing only confirmed field observations.",
    )
    merge_observations.add_argument(
        "--denied-output",
        help="Optional point layer containing no-culvert field observations.",
    )
    merge_observations.add_argument(
        "--denied-csv-output",
        help="Optional CSV export for no-culvert field observations.",
    )
    merge_observations.add_argument(
        "--exclude-confirmed",
        action="store_true",
        help="Do not merge confirmed dashboard observations as positive training labels.",
    )
    merge_observations.set_defaults(func=_merge_field_observations)

    add_report_candidates = subparsers.add_parser(
        "add-field-report-candidates",
        help="Append imported field-report culvert points to a candidate layer.",
    )
    add_report_candidates.add_argument("--candidates", required=True, help="Base candidate file.")
    add_report_candidates.add_argument(
        "--field-reports",
        required=True,
        help="Imported field report point layer.",
    )
    add_report_candidates.add_argument("--output", required=True, help="Output candidate file.")
    add_report_candidates.add_argument("--boundary", help="Optional boundary to clip field points.")
    add_report_candidates.set_defaults(func=_add_field_report_candidates)

    region_boundary = subparsers.add_parser(
        "make-region-boundary",
        help="Write a pilot region boundary file, currently focused on Ulster County.",
    )
    region_boundary.add_argument("--region", default="ulster_poughkeepsie")
    region_boundary.add_argument(
        "--output",
        default="configs/regions/ulster_poughkeepsie_pilot.geojson",
        help="Output boundary path.",
    )
    region_boundary.set_defaults(func=_make_region_boundary)

    filter_region = subparsers.add_parser(
        "filter-region",
        help="Filter a vector layer to the Ulster/Poughkeepsie pilot region or a supplied boundary.",
    )
    filter_region.add_argument("--input", required=True, help="Input vector file.")
    filter_region.add_argument("--output", required=True, help="Output vector file.")
    filter_region.add_argument("--region", default="ulster_poughkeepsie")
    filter_region.add_argument(
        "--boundary",
        help="Optional official county/project boundary file. If supplied, it overrides --region.",
    )
    filter_region.add_argument("--no-clip", action="store_true", help="Filter only; do not clip geometry.")
    filter_region.set_defaults(func=_filter_region)

    candidates = subparsers.add_parser(
        "build-candidates", help="Generate likely culvert candidates from road-stream crossings."
    )
    candidates.add_argument("--roads", required=True, help="Road centerline vector file.")
    candidates.add_argument("--streams", required=True, help="Stream/drainage vector file.")
    candidates.add_argument("--output", required=True, help="Output GPKG/GeoJSON/CSV path.")
    candidates.add_argument("--snap-tolerance-m", type=float, default=20.0)
    candidates.add_argument("--min-spacing-m", type=float, default=25.0)
    candidates.add_argument("--road-id-column")
    candidates.add_argument("--stream-id-column")
    candidates.set_defaults(func=_build_candidates)

    road_candidates = subparsers.add_parser(
        "build-road-candidates",
        help="Generate candidate points sampled along selected road routes.",
    )
    road_candidates.add_argument("--roads", required=True, help="Road centerline vector file.")
    road_candidates.add_argument("--output", required=True, help="Output candidate file.")
    road_candidates.add_argument("--routes", nargs="*", help="Routes such as NY28, NY32, I87.")
    road_candidates.add_argument(
        "--routes-from",
        help="Optional field report point layer containing a route column.",
    )
    road_candidates.add_argument("--interval-m", type=float, default=20.0)
    road_candidates.set_defaults(func=_build_road_candidates)

    merge_candidates = subparsers.add_parser(
        "merge-candidates",
        help="Merge multiple candidate layers into one candidate file.",
    )
    merge_candidates.add_argument("--inputs", nargs="+", required=True)
    merge_candidates.add_argument("--output", required=True)
    merge_candidates.set_defaults(func=_merge_candidates)

    features = subparsers.add_parser(
        "build-features", help="Build model features for candidate culvert points."
    )
    features.add_argument("--candidates", required=True, help="Candidate point vector file.")
    features.add_argument("--output", required=True, help="Output feature vector file.")
    features.add_argument("--known-culverts", help="Known culvert point inventory.")
    features.add_argument("--negative-culverts", help="Field-confirmed no-culvert point layer.")
    features.add_argument("--roads", help="Road centerline vector file.")
    features.add_argument("--streams", help="Stream/drainage vector file.")
    features.add_argument("--dem", help="DEM raster path.")
    features.add_argument("--flow-accumulation", help="Optional flow accumulation raster path.")
    features.add_argument("--drainage-area", help="Optional drainage area raster path.")
    features.add_argument("--landcover", help="Land cover raster path.")
    features.add_argument("--positive-radius-m", type=float, default=20.0)
    features.add_argument("--negative-radius-m", type=float, default=20.0)
    features.add_argument("--density-radius-m", type=float, default=75.0)
    features.add_argument(
        "--density-radii-m",
        type=float,
        nargs="*",
        help="Optional extra radii for road/stream density features.",
    )
    features.set_defaults(func=_build_features)

    train = subparsers.add_parser("train", help="Train a culvert probability model.")
    train.add_argument("--features", required=True, help="Training feature vector file.")
    train.add_argument("--model-output", default="models/culvert_model.joblib")
    train.add_argument("--metrics-output", default="reports/metrics.json")
    train.add_argument("--importance-output", default="reports/feature_importance.csv")
    train.add_argument("--target-column", default="is_culvert")
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument(
        "--model-family",
        default="auto",
        choices=[
            "auto",
            "regularized_logistic",
            "random_forest",
            "extra_trees",
            "spatial_regularized_extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "balanced_hist_gradient_boosting",
        ],
        help="Use auto to compare models, or force a specific model family.",
    )
    train.add_argument(
        "--no-spatial-cv",
        action="store_true",
        help="Disable spatial holdout validation.",
    )
    train.add_argument("--spatial-block-size-m", type=float, default=2_500.0)
    train.set_defaults(func=_train)

    score_unlabeled = subparsers.add_parser(
        "score-unlabeled",
        help="Rank likely culvert locations without known local culvert labels.",
    )
    score_unlabeled.add_argument("--features", required=True, help="Feature vector file to score.")
    score_unlabeled.add_argument("--output", required=True, help="Output ranked vector file.")
    score_unlabeled.add_argument("--csv-output", help="Optional CSV export for field review.")
    score_unlabeled.add_argument(
        "--kml-output",
        help="Optional KML export for Google Earth review.",
    )
    score_unlabeled.add_argument(
        "--kml-max-points",
        type=int,
        default=250,
        help="Maximum number of ranked points to include in the KML.",
    )
    score_unlabeled.set_defaults(func=_score_unlabeled)

    discovery = subparsers.add_parser(
        "build-discovery-ranking",
        help="Blend GIS evidence and model probability, ranking undiscovered candidates first.",
    )
    discovery.add_argument(
        "--evidence-predictions",
        required=True,
        help="Prediction file from score-unlabeled with interpretable GIS evidence.",
    )
    discovery.add_argument(
        "--supervised-predictions",
        help="Optional prediction file from predict with culvert_probability.",
    )
    discovery.add_argument("--output", required=True, help="Output discovery-ranked vector file.")
    discovery.add_argument("--csv-output", help="Optional CSV export for field review.")
    discovery.add_argument("--kml-output", help="Optional KML export for Google Earth review.")
    discovery.add_argument("--kml-max-points", type=int, default=500)
    discovery.add_argument("--evidence-weight", type=float, default=0.40)
    discovery.add_argument("--model-weight", type=float, default=0.60)
    discovery.add_argument("--known-radius-m", type=float, default=20.0)
    discovery.set_defaults(func=_build_discovery_ranking)

    export_web = subparsers.add_parser(
        "export-web",
        help="Export ranked findings to web/data for the interactive dashboard.",
    )
    export_web.add_argument("--predictions", required=True, help="Ranked prediction vector file.")
    export_web.add_argument("--output-dir", default="web/data", help="Web data output directory.")
    export_web.add_argument("--limit", type=int, help="Optional max number of findings to export.")
    export_web.set_defaults(func=_export_web)

    predict = subparsers.add_parser("predict", help="Rank likely culvert locations.")
    predict.add_argument("--features", required=True, help="Feature vector file to score.")
    predict.add_argument("--model", required=True, help="Trained model joblib file.")
    predict.add_argument("--output", required=True, help="Output ranked vector file.")
    predict.add_argument("--csv-output", help="Optional CSV export for field review.")
    predict.set_defaults(func=_predict)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against known culverts.")
    evaluate.add_argument("--predictions", required=True, help="Prediction vector file.")
    evaluate.add_argument("--known-culverts", required=True, help="Known culvert point inventory.")
    evaluate.add_argument("--output", default="reports/prediction_eval.json")
    evaluate.add_argument("--probability-threshold", type=float, default=0.7)
    evaluate.add_argument("--match-radius-m", type=float, default=30.0)
    evaluate.set_defaults(func=_evaluate)

    return parser


def _make_demo_data(args) -> dict:
    return create_demo_dataset(args.output_dir)


def _run_demo(args) -> dict:
    return run_demo_pipeline(args.output_dir)


def _download_osm(args) -> dict:
    return download_ulster_osm_inputs(
        output_dir=args.output_dir,
        overpass_url=args.overpass_url,
        census_county_url=args.census_county_url,
        county_name=args.county,
        state_name=args.state,
        statefp=args.statefp,
        countyfp=args.countyfp,
        timeout_seconds=args.timeout_seconds,
        tile_size_degrees=args.tile_size_degrees,
    )


def _download_census(args) -> dict:
    return download_ulster_census_inputs(
        output_dir=args.output_dir,
        tiger_year=args.tiger_year,
        statefp=args.statefp,
        countyfp=args.countyfp,
        county_boundary_url=args.county_boundary_url,
    )


def _import_field_reports(args) -> dict:
    return import_field_reports(
        input_path=args.input,
        output_path=args.output,
        csv_output=args.csv_output,
        dedupe_precision=args.dedupe_precision,
    )


def _prepare_llm_label_review(args) -> dict:
    return write_llm_label_review_queue(
        input_path=args.input,
        output_path=args.output,
        dedupe_precision=args.dedupe_precision,
    )


def _import_llm_reviewed_labels(args) -> dict:
    return import_llm_reviewed_labels(
        review_path=args.input,
        output_path=args.output,
        csv_output=args.csv_output,
    )


def _extract_points_only(args) -> dict:
    return write_point_only_layer(
        points_path=args.points,
        output_path=args.output,
        csv_output=args.csv_output,
    )


def _analyze_extracted_points(args) -> dict:
    return analyze_extracted_points(
        points_path=args.points,
        output_geojson=args.output_geojson,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        roads_path=args.roads,
        streams_path=args.streams,
        candidates_path=args.candidates,
        boundary_path=args.boundary,
        match_radius_m=args.match_radius_m,
        cluster_radius_m=args.cluster_radius_m,
    )


def _build_high_confidence_training_points(args) -> dict:
    return write_high_confidence_training_points(
        analysis_path=args.analysis,
        output_path=args.output,
        csv_output=args.csv_output,
        accepted_flags=tuple(args.accepted_flags or ["matched_existing_candidate"]),
    )


def _merge_field_observations(args) -> dict:
    return merge_confirmed_observations(
        observations_path=args.observations,
        base_known_path=args.base_known,
        output_path=args.output,
        csv_output=args.csv_output,
        confirmed_output_path=args.confirmed_output,
        denied_output_path=args.denied_output,
        denied_csv_output=args.denied_csv_output,
        include_confirmed=not args.exclude_confirmed,
    )


def _add_field_report_candidates(args) -> dict:
    return append_field_report_candidates(
        candidates_path=args.candidates,
        field_reports_path=args.field_reports,
        output_path=args.output,
        boundary_path=args.boundary,
    )


def _make_region_boundary(args) -> dict:
    region = get_region(args.region)
    write_region_boundary(args.output, args.region)
    return {
        "region": region.name,
        "output": Path(args.output),
        "focus_places": ", ".join(region.focus_places),
    }


def _filter_region(args) -> dict:
    data = read_vector(args.input)
    filtered = filter_to_region(
        data,
        region_key=args.region,
        boundary_path=args.boundary,
        clip=not args.no_clip,
    )
    write_vector(filtered, args.output)
    return {"input": Path(args.input), "output": Path(args.output), "rows": len(filtered)}


def _build_candidates(args) -> dict:
    roads = read_vector(args.roads)
    streams = read_vector(args.streams)
    settings = CandidateSettings(
        snap_tolerance_m=args.snap_tolerance_m,
        min_spacing_m=args.min_spacing_m,
        road_id_column=args.road_id_column,
        stream_id_column=args.stream_id_column,
    )
    output = generate_candidates(roads, streams, settings)
    write_vector(output, args.output)
    return {"candidates": Path(args.output), "rows": len(output)}


def _build_road_candidates(args) -> dict:
    roads = read_vector(args.roads)
    routes = list(args.routes or [])
    if args.routes_from:
        route_points = read_vector(args.routes_from)
        if "route" in route_points.columns:
            routes.extend(str(route) for route in route_points["route"].dropna().unique())
    output = generate_road_route_candidates(roads, routes=routes, interval_m=args.interval_m)
    write_vector(output, args.output)
    return {
        "candidates": Path(args.output),
        "rows": len(output),
        "routes": sorted(set(routes)),
        "interval_m": args.interval_m,
    }


def _merge_candidates(args) -> dict:
    layers = [read_vector(path) for path in args.inputs]
    output = merge_candidate_layers(layers)
    write_vector(output, args.output)
    return {"candidates": Path(args.output), "rows": len(output), "inputs": args.inputs}


def _build_features(args) -> dict:
    candidates = read_vector(args.candidates)
    known = read_vector(args.known_culverts) if args.known_culverts else None
    negative = read_vector(args.negative_culverts) if args.negative_culverts else None
    roads = read_vector(args.roads) if args.roads else None
    streams = read_vector(args.streams) if args.streams else None
    output = build_feature_table(
        candidates,
        known_culverts=known,
        negative_culverts=negative,
        roads=roads,
        streams=streams,
        dem_path=args.dem,
        flow_accumulation_path=args.flow_accumulation,
        drainage_area_path=args.drainage_area,
        landcover_path=args.landcover,
        positive_radius_m=args.positive_radius_m,
        negative_radius_m=args.negative_radius_m,
        density_radius_m=args.density_radius_m,
        density_radii_m=tuple(args.density_radii_m) if args.density_radii_m else None,
    )
    write_vector(output, args.output)
    return {"features": Path(args.output), "rows": len(output)}


def _train(args) -> dict:
    features = read_vector(args.features)
    metrics = train_model(
        features,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        importance_output=args.importance_output,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        model_family=args.model_family,
        spatial_cv=not args.no_spatial_cv,
        spatial_block_size_m=args.spatial_block_size_m,
    )
    metrics["model"] = str(args.model_output)
    metrics["metrics"] = str(args.metrics_output)
    metrics["feature_importance_csv"] = str(args.importance_output)
    return metrics


def _predict(args) -> dict:
    features = read_vector(args.features)
    predictions = predict_culvert_probability(features, args.model)
    write_vector(predictions, args.output)
    result = {"predictions": Path(args.output), "rows": len(predictions)}
    if args.csv_output:
        write_vector(predictions, args.csv_output)
        result["predictions_csv"] = Path(args.csv_output)
    return result


def _score_unlabeled(args) -> dict:
    features = read_vector(args.features)
    scored = score_unlabeled_candidates(features)
    write_vector(scored, args.output)
    result = {"predictions": Path(args.output), "rows": len(scored)}
    if args.csv_output:
        write_vector(scored, args.csv_output)
        result["predictions_csv"] = Path(args.csv_output)
    if args.kml_output:
        write_google_earth_kml(scored, args.kml_output, max_points=args.kml_max_points)
        result["google_earth_kml"] = Path(args.kml_output)
    return result


def _build_discovery_ranking(args) -> dict:
    evidence = read_vector(args.evidence_predictions)
    supervised = read_vector(args.supervised_predictions) if args.supervised_predictions else None
    ranked = build_discovery_ranking(
        evidence,
        supervised_predictions=supervised,
        evidence_weight=args.evidence_weight,
        model_weight=args.model_weight,
        known_radius_m=args.known_radius_m,
    )
    write_vector(ranked, args.output)
    result = {"predictions": Path(args.output), "rows": len(ranked)}
    if args.csv_output:
        write_vector(ranked, args.csv_output)
        result["predictions_csv"] = Path(args.csv_output)
    if args.kml_output:
        write_google_earth_kml(ranked, args.kml_output, max_points=args.kml_max_points)
        result["google_earth_kml"] = Path(args.kml_output)
    return result


def _export_web(args) -> dict:
    return export_web_data(args.predictions, args.output_dir, limit=args.limit)


def _evaluate(args) -> dict:
    predictions = read_vector(args.predictions)
    known = read_vector(args.known_culverts)
    return evaluate_predictions(
        predictions,
        known,
        output_path=args.output,
        probability_threshold=args.probability_threshold,
        match_radius_m=args.match_radius_m,
    )


if __name__ == "__main__":
    raise SystemExit(main())
