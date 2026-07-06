# Graph Report - .  (2026-07-06)

## Corpus Check
- Corpus is ~41,361 words - fits in a single context window. You may not need a graph.

## Summary
- 832 nodes · 2089 edges · 50 communities (36 shown, 14 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 315 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Retraining API|Retraining API]]
- [[_COMMUNITY_Feature Engineering|Feature Engineering]]
- [[_COMMUNITY_Candidate Generation|Candidate Generation]]
- [[_COMMUNITY_Model Training|Model Training]]
- [[_COMMUNITY_Discovery Scoring|Discovery Scoring]]
- [[_COMMUNITY_Map App Core|Map App Core]]
- [[_COMMUNITY_Field Report Parsing|Field Report Parsing]]
- [[_COMMUNITY_Dev Server API|Dev Server API]]
- [[_COMMUNITY_CLI Pipeline|CLI Pipeline]]
- [[_COMMUNITY_Node Project Config|Node Project Config]]
- [[_COMMUNITY_Visible Marker List|Visible Marker List]]
- [[_COMMUNITY_Detail Feedback UI|Detail Feedback UI]]
- [[_COMMUNITY_Observation Fetching|Observation Fetching]]
- [[_COMMUNITY_Point Analysis|Point Analysis]]
- [[_COMMUNITY_Web Export|Web Export]]
- [[_COMMUNITY_Census Inputs|Census Inputs]]
- [[_COMMUNITY_LLM Label Import|LLM Label Import]]
- [[_COMMUNITY_Draft Point UI|Draft Point UI]]
- [[_COMMUNITY_DEM Acquisition|DEM Acquisition]]
- [[_COMMUNITY_Observation Labels|Observation Labels]]
- [[_COMMUNITY_Model Summary|Model Summary]]
- [[_COMMUNITY_Config Model Docs|Config Model Docs]]
- [[_COMMUNITY_Prediction Evaluation|Prediction Evaluation]]
- [[_COMMUNITY_Observation Tests|Observation Tests]]
- [[_COMMUNITY_Vercel Config|Vercel Config]]
- [[_COMMUNITY_Draft Point Save|Draft Point Save]]
- [[_COMMUNITY_Location Tracking|Location Tracking]]
- [[_COMMUNITY_Modeling Dependencies|Modeling Dependencies]]
- [[_COMMUNITY_Region Boundary|Region Boundary]]
- [[_COMMUNITY_Field Feedback Loop|Field Feedback Loop]]
- [[_COMMUNITY_Field Data State|Field Data State]]
- [[_COMMUNITY_Research Notes|Research Notes]]
- [[_COMMUNITY_Prediction Workflow|Prediction Workflow]]
- [[_COMMUNITY_HTML Review Shell|HTML Review Shell]]
- [[_COMMUNITY_Vercel Observation Pull|Vercel Observation Pull]]
- [[_COMMUNITY_Web Build Verify|Web Build Verify]]
- [[_COMMUNITY_KML Export|KML Export]]
- [[_COMMUNITY_Python Wrapper|Python Wrapper]]
- [[_COMMUNITY_Python Bootstrap|Python Bootstrap]]
- [[_COMMUNITY_Actual Predictions Script|Actual Predictions Script]]
- [[_COMMUNITY_Census Pipeline Script|Census Pipeline Script]]
- [[_COMMUNITY_OSM Pipeline Script|OSM Pipeline Script]]
- [[_COMMUNITY_Demo Script|Demo Script]]
- [[_COMMUNITY_Real Pipeline Script|Real Pipeline Script]]
- [[_COMMUNITY_Transfer Script|Transfer Script]]
- [[_COMMUNITY_Ulster Pipeline Script|Ulster Pipeline Script]]
- [[_COMMUNITY_Unlabeled Pipeline Script|Unlabeled Pipeline Script]]
- [[_COMMUNITY_Web Serve Script|Web Serve Script]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Project Root|Project Root]]

## God Nodes (most connected - your core abstractions)
1. `write_vector()` - 33 edges
2. `build_parser()` - 30 edges
3. `build_feature_table()` - 30 edges
4. `read_vector()` - 30 edges
5. `score_unlabeled_candidates()` - 22 edges
6. `train_model()` - 20 edges
7. `scripts` - 19 edges
8. `generate_road_route_candidates()` - 19 edges
9. `renderList()` - 18 edges
10. `generate_candidates()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Research Plan` --semantically_similar_to--> `Geospatial ML Workflow`  [INFERRED] [semantically similar]
  docs/research_notes.md → README.md
- `Leaflet Field Review UI` --semantically_similar_to--> `Culvert AI Field Review HTML`  [INFERRED] [semantically similar]
  README.md → web/index.html
- `Continuous Learning Behavior` --semantically_similar_to--> `Continuous Retraining Trigger`  [INFERRED] [semantically similar]
  model.md → README.md
- `Discovery Score` --semantically_similar_to--> `Field Review Ranking Logic`  [INFERRED] [semantically similar]
  model.md → README.md
- `Feature Table` --semantically_similar_to--> `Example Feature Parameters`  [INFERRED] [semantically similar]
  model.md → configs/example.yml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Field Feedback Retraining Loop** — readme_vercel_blob_persistence, readme_continuous_retraining_trigger, model_continuous_learning_behavior, track_automatic_retraining_state, web_index_observation_controls [INFERRED 0.85]
- **Geospatial Modeling Stack** — configs_example_model_parameters, configs_ulster_poughkeepsie_model_parameters, model_supervised_model, requirements_python_geospatial_ml_dependencies, readme_model_summary_hist_gradient_boosting [INFERRED 0.85]
- **Field Report Training Data Flow** — configs_field_report_inputs_field_report_manifest, readme_field_report_coordinate_extraction, model_candidate_generation, model_supervised_model, track_current_data_state, docs_research_notes_research_plan [INFERRED 0.85]

## Communities (50 total, 14 thin omitted)

### Community 0 - "Retraining API"
Cohesion: 0.08
Nodes (63): handler(), handler(), appendEvidenceSummary(), applyFeedbackToFindings(), applyMissedPredictionToFeature(), applyObservationToFeature(), baseFingerprint(), blobConfigured() (+55 more)

### Community 1 - "Feature Engineering"
Cohesion: 0.09
Nodes (58): Point, add_approved_known_dem_similarity_features(), add_candidate_derived_features(), add_dem_culvert_terrain_features(), add_dem_hydrology_proxies(), add_hydrology_raster_features(), add_known_culvert_labels(), add_known_culvert_pattern_features() (+50 more)

### Community 2 - "Candidate Generation"
Cohesion: 0.09
Nodes (50): LineString, _auto_sample_numbered_road(), CandidateSettings, _crossing_angle_degrees(), _deduplicate(), _first_numeric(), _first_value(), generate_candidates() (+42 more)

### Community 3 - "Model Training"
Cohesion: 0.12
Nodes (41): ExtraTreesClassifier, HistGradientBoostingClassifier, ndarray, _balanced_hist_gradient_boosting(), _candidate_models(), _classification_metrics(), _compare_models(), _cross_validate_model() (+33 more)

### Community 4 - "Discovery Scoring"
Cohesion: 0.16
Nodes (40): _attach_supervised_probability(), _boolean_score(), build_discovery_ranking(), _crossing_geometry_score(), _dem_route_drainage_score(), _discovery_evidence_summary(), _drainage_strength_score(), _evidence_summary() (+32 more)

### Community 5 - "Map App Core"
Cohesion: 0.09
Nodes (36): candidateCanvasColor(), candidateCanvasLabel(), compareCanvasCandidatePriority(), DATA_URLS, declutterCanvasItems(), definitionItem(), discoveryStatusLabel(), drainageLabel() (+28 more)

### Community 6 - "Field Report Parsing"
Cohesion: 0.12
Nodes (34): Match, _clean_text_line(), CoordinateRecord, _culvert_ids(), _deduplicate_records(), _docx_text(), extract_field_report_records(), _extract_from_files() (+26 more)

### Community 7 - "Dev Server API"
Cohesion: 0.12
Nodes (29): appendObservation(), canRead(), DATA_DIR, deleteObservation(), __dirname, emptyFeatureCollection(), fileInfo(), handleRequest() (+21 more)

### Community 8 - "CLI Pipeline"
Cohesion: 0.15
Nodes (29): ArgumentParser, _add_field_report_candidates(), _analyze_extracted_points(), _build_discovery_ranking(), _build_features(), _build_high_confidence_training_points(), build_parser(), _build_road_candidates() (+21 more)

### Community 9 - "Node Project Config"
Cohesion: 0.07
Nodes (28): dependencies, @vercel/blob, description, engines, node, name, private, scripts (+20 more)

### Community 10 - "Visible Marker List"
Cohesion: 0.12
Nodes (29): bindControls(), buildSearchText(), compareFeaturesForList(), distanceMeters(), featureLatLng(), fitVisibleMarkers(), idOf(), isMobileDrawerOpen() (+21 more)

### Community 11 - "Detail Feedback UI"
Cohesion: 0.15
Nodes (28): abuObservations(), bindDetailCloseAction(), bindFeedbackActions(), compactEvidenceSummary(), detailCell(), escapeAttr(), escapeHtml(), fieldCulvertContextHtml() (+20 more)

### Community 12 - "Observation Fetching"
Cohesion: 0.14
Nodes (26): addObservation(), applyDashboardData(), clearLocalObservations(), createCandidateCanvasLayer(), deleteObservationById(), fetchFirst(), fetchFirstOptional(), fetchJson() (+18 more)

### Community 13 - "Point Analysis"
Cohesion: 0.18
Nodes (23): _analysis_flag(), analyze_extracted_points(), _attach_nearest_candidate(), _attach_nearest_line(), _candidate_score(), _cluster_ids(), _distance_stats(), _feature_name() (+15 more)

### Community 14 - "Web Export"
Cohesion: 0.23
Nodes (17): _decluster_for_web(), _export_key_value(), export_web_data(), _known_match_count(), _limit_for_web(), _prediction_export_pool(), GeoDataFrame, Path (+9 more)

### Community 15 - "Census Inputs"
Cohesion: 0.24
Nodes (14): _county_boundary(), _download_if_missing(), download_ulster_census_inputs(), _normalize_linear_water(), _normalize_roads(), GeoDataFrame, Path, Download actual county-level TIGER/Line roads and linear-water data for Ulster C (+6 more)

### Community 16 - "LLM Label Import"
Cohesion: 0.28
Nodes (14): _clean_float(), _clean_string(), import_llm_reviewed_labels(), Any, Path, _queue_row(), _read_jsonl(), _review_id() (+6 more)

### Community 17 - "Draft Point UI"
Cohesion: 0.20
Nodes (16): bindDraftPointActions(), cancelPlacePointMode(), clearDetail(), hideDetailPanel(), observationIcon(), observationIdOf(), observationLatLng(), openSelectedFeaturePopup() (+8 more)

### Community 18 - "DEM Acquisition"
Cohesion: 0.23
Nodes (13): dem_tiles_for_bounds(), _download_if_missing(), download_usgs_3dep_dem(), _expanded_bounds(), Path, Return USGS 3DEP tile IDs intersecting WGS84 bounds.      USGS 3DEP current elev, Download and mosaic USGS 3DEP DEM tiles covering a boundary layer., _tile_id() (+5 more)

### Community 19 - "Observation Labels"
Cohesion: 0.41
Nodes (14): _confirmed_observations_as_known(), _date_part(), _dedupe_observation_rows(), _denied_observations_as_negative(), _empty_observation_labels(), _field_negative_observations(), _first_non_empty_series(), _is_prediction_candidate_id() (+6 more)

### Community 20 - "Model Summary"
Cohesion: 0.42
Nodes (12): build_summary(), _csv_row_count(), _float_or_none(), _int_or_none(), main(), _nested_number(), _point_qc(), Any (+4 more)

### Community 21 - "Config Model Docs"
Cohesion: 0.21
Nodes (12): Example Candidate Generation Parameters, Example Feature Parameters, Example Model Parameters, Example Project Configuration, Ulster Candidate Generation Parameters, Ulster Model Parameters, Ulster Poughkeepsie Project Configuration, Candidate Generation (+4 more)

### Community 22 - "Prediction Evaluation"
Cohesion: 0.29
Nodes (10): _actual_id(), evaluate_predictions(), evaluate_success_rate_at_actuals(), _optional_number(), GeoDataFrame, Path, Measure field success as actual culverts with a prediction within max_distance_m, _unknown_prediction_pool() (+2 more)

### Community 23 - "Observation Tests"
Cohesion: 0.33
Nodes (11): append_field_report_candidates(), write_vector(), merge_confirmed_observations(), Path, _read_optional_base(), test_append_field_report_candidates_continues_existing_field_ids(), test_merge_confirmed_observations_adds_confirmed_points(), test_merge_confirmed_observations_deduplicates_repeated_field_ids() (+3 more)

### Community 24 - "Vercel Config"
Cohesion: 0.17
Nodes (11): includeFiles, maxDuration, buildCommand, crons, framework, functions, api/**/*.js, headers (+3 more)

### Community 25 - "Draft Point Save"
Cohesion: 0.21
Nodes (12): centerMapOnPoint(), draftPointIcon(), draftPointSaveHtml(), draftPointStartLatLng(), handleMapClick(), makeFieldCulvertId(), renderDraftPointDetail(), renderDraftPointMarker() (+4 more)

### Community 26 - "Location Tracking"
Cohesion: 0.33
Nodes (12): focusUserLocation(), handleLocationError(), handleLocationSuccess(), isMobileViewport(), recenterOnUserLocation(), renderLocationMarker(), setLocationStatus(), startLocationTracking() (+4 more)

### Community 27 - "Modeling Dependencies"
Cohesion: 0.25
Nodes (11): Spatial Holdout Model Selection, Supervised Model, Geospatial ML Workflow, Spatial Holdout Validation, USGS 3DEP DEM Features, GeoPandas, Joblib, Python Geospatial ML Dependencies (+3 more)

### Community 28 - "Region Boundary"
Cohesion: 0.35
Nodes (9): _make_region_boundary(), filter_to_region(), get_region(), GeoDataFrame, Path, Region, region_boundary(), write_region_boundary() (+1 more)

### Community 29 - "Field Feedback Loop"
Cohesion: 0.22
Nodes (10): ABU User Observations, Continuous Retraining Trigger, Culvert AI Ulster County Pilot, Leaflet Field Review UI, Field-Useful Prediction, Strict 10 m Match Rule, Vercel Blob Persistence, Automatic Retraining State (+2 more)

### Community 30 - "Field Data State"
Cohesion: 0.22
Nodes (9): Field Report Input Manifest, Team 2 Selected Report Folder, Team 4 Daily Report PDFs, Field Report Coordinate Extraction, Current Selected Model Hist Gradient Boosting, Culvert AI Track, Current Data State, Current Goal (+1 more)

### Community 31 - "Research Notes"
Cohesion: 0.22
Nodes (9): Data Needed, Field Search Time Bottleneck, LLM Role Boundary, Research Notes, Research Plan, Accuracy Bottlenecks, LLM Not Primary Predictor, Bottlenecks (+1 more)

### Community 32 - "Prediction Workflow"
Cohesion: 0.36
Nodes (8): Continuous Learning Behavior, Culvert Prediction Model, Culvert Probability, Current Run Soft Voting Ensemble, Discovery Score, Interpretable GIS Evidence Score, Web Export, Field Review Ranking Logic

### Community 33 - "HTML Review Shell"
Cohesion: 0.32
Nodes (8): App JS Asset, Candidate Template, Culvert AI Field Review HTML, Detail Modal, Leaflet 1.9.4 Assets, Map Area, Sidebar Candidate Controls, Styles CSS Asset

### Community 35 - "Web Build Verify"
Cohesion: 0.33
Nodes (5): findings, modelSummary, requiredFiles, root, summary

### Community 36 - "KML Export"
Cohesion: 0.50
Nodes (4): ensure_parent_dir(), Path, Path, write_google_earth_kml()

## Knowledge Gaps
- **87 isolated node(s):** `ROOT`, `STATIC_FINDINGS_PATH`, `STATIC_SUMMARY_PATH`, `OBSERVATION_STATUSES`, `FEEDBACK_MATCH_RADIUS_M` (+82 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `write_vector()` connect `Observation Tests` to `Candidate Generation`, `KML Export`, `Field Report Parsing`, `CLI Pipeline`, `Point Analysis`, `Census Inputs`, `LLM Label Import`, `Region Boundary`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Why does `build_feature_table()` connect `Feature Engineering` to `CLI Pipeline`, `Candidate Generation`, `Census Inputs`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `read_vector()` connect `CLI Pipeline` to `Candidate Generation`, `KML Export`, `Point Analysis`, `Web Export`, `Census Inputs`, `LLM Label Import`, `DEM Acquisition`, `Observation Tests`, `Region Boundary`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Are the 35 inferred relationships involving `Point` (e.g. with `_demo_known_culverts()` and `import_field_reports()`) actually correct?**
  _`Point` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `write_vector()` (e.g. with `download_ulster_census_inputs()` and `_build_candidates()`) actually correct?**
  _`write_vector()` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `build_parser()` (e.g. with `_add_field_report_candidates()` and `_analyze_extracted_points()`) actually correct?**
  _`build_parser()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `build_feature_table()` (e.g. with `_build_features()` and `run_demo_pipeline()`) actually correct?**
  _`build_feature_table()` has 12 INFERRED edges - model-reasoned connections that need verification._