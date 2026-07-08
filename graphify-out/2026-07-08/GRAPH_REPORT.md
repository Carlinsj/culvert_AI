# Graph Report - /Users/Carli/culvert_AI  (2026-07-08)

## Corpus Check
- 14 files · ~46,071 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 863 nodes · 2076 edges · 48 communities (33 shown, 15 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 252 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b1713b0f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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
- [[_COMMUNITY_Field Reviewer System|Field Reviewer System]]
- [[_COMMUNITY_Deeper Validation Roadmap|Deeper Validation Roadmap]]
- [[_COMMUNITY_Vercel Config|Vercel Config]]
- [[_COMMUNITY_Draft Point Save|Draft Point Save]]
- [[_COMMUNITY_Location Tracking|Location Tracking]]
- [[_COMMUNITY_Modeling Dependencies|Modeling Dependencies]]
- [[_COMMUNITY_Continuous Learning Behavior|Continuous Learning Behavior]]
- [[_COMMUNITY_Field Feedback Loop|Field Feedback Loop]]
- [[_COMMUNITY_Field Data State|Field Data State]]
- [[_COMMUNITY_Research Notes|Research Notes]]
- [[_COMMUNITY_Prediction Workflow|Prediction Workflow]]
- [[_COMMUNITY_Vercel Observation Pull|Vercel Observation Pull]]
- [[_COMMUNITY_Web Build Verify|Web Build Verify]]
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
1. `build_feature_table()` - 31 edges
2. `build_parser()` - 29 edges
3. `write_vector()` - 25 edges
4. `score_unlabeled_candidates()` - 23 edges
5. `train_model()` - 21 edges
6. `scripts` - 19 edges
7. `read_vector()` - 18 edges
8. `renderList()` - 18 edges
9. `build_discovery_ranking()` - 18 edges
10. `generate_candidates()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Discovery Score` --semantically_similar_to--> `Discovery Score`  [INFERRED] [semantically similar]
  outputs/culvert-ai-research-portfolio-contact-sheet.jpg → model.md
- `Spatial Candidate Workflow` --semantically_similar_to--> `Candidate Generation`  [INFERRED] [semantically similar]
  outputs/culvert-ai-research-portfolio.pdf → model.md
- `Supervised Classification Task` --semantically_similar_to--> `Supervised Model`  [INFERRED] [semantically similar]
  outputs/culvert-ai-research-portfolio.pdf → model.md
- `Discovery Score Blend` --semantically_similar_to--> `Discovery Score`  [INFERRED] [semantically similar]
  outputs/culvert-ai-research-portfolio.pdf → model.md
- `Soft Voting Ensemble` --semantically_similar_to--> `Current Run Soft Voting Ensemble`  [INFERRED] [semantically similar]
  outputs/culvert-ai-research-portfolio-contact-sheet.jpg → model.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Culvert Prediction Review Pipeline** — model_candidate_generation, model_feature_table, model_supervised_model, model_discovery_score, outputs_culvert_ai_research_portfolio_field_review_interface [INFERRED 0.95]
- **Model Improvement Levers** — graphify_out_memory_query_20260707_173049_now_how_to_improve_model_train_this_model_or_switc_data_quality_improvement_levers, model_data_and_validation_limitations, outputs_culvert_ai_research_portfolio_data_and_protocol_bottlenecks, outputs_culvert_ai_research_portfolio_deeper_validation_roadmap [INFERRED 0.85]
- **Contact Sheet Validation Roadmap** — outputs_culvert_ai_research_portfolio_contact_sheet_soft_voting_ensemble, outputs_culvert_ai_research_portfolio_contact_sheet_discovery_score, outputs_culvert_ai_research_portfolio_contact_sheet_validation_roadmap [INFERRED 0.85]
- **Field Feedback Retraining Loop** — readme_vercel_blob_persistence, readme_continuous_retraining_trigger, model_continuous_learning_behavior, track_automatic_retraining_state, web_index_observation_controls [INFERRED 0.85]
- **Field Report Training Data Flow** — configs_field_report_inputs_field_report_manifest, readme_field_report_coordinate_extraction, model_candidate_generation, model_supervised_model, track_current_data_state, docs_research_notes_research_plan [INFERRED 0.85]

## Communities (48 total, 15 thin omitted)

### Community 0 - "Retraining API"
Cohesion: 0.08
Nodes (63): handler(), handler(), appendEvidenceSummary(), applyFeedbackToFindings(), applyMissedPredictionToFeature(), applyObservationToFeature(), baseFingerprint(), blobConfigured() (+55 more)

### Community 1 - "Feature Engineering"
Cohesion: 0.09
Nodes (56): add_approved_known_dem_similarity_features(), add_candidate_derived_features(), add_dem_culvert_terrain_features(), add_dem_hydrology_proxies(), add_hydrology_raster_features(), add_known_culvert_labels(), add_known_culvert_pattern_features(), add_known_culvert_pattern_score() (+48 more)

### Community 2 - "Candidate Generation"
Cohesion: 0.21
Nodes (23): _auto_sample_numbered_road(), CandidateSettings, _crossing_angle_degrees(), _deduplicate(), _first_numeric(), _first_value(), generate_candidates(), generate_road_route_candidates() (+15 more)

### Community 3 - "Model Training"
Cohesion: 0.12
Nodes (42): ExtraTreesClassifier, Auto Model Family Comparison, HistGradientBoostingClassifier, ndarray, _balanced_hist_gradient_boosting(), _candidate_models(), _classification_metrics(), _compare_models() (+34 more)

### Community 4 - "Discovery Scoring"
Cohesion: 0.14
Nodes (44): Data Quality Improvement Levers, Model Improvement Query, Data and Validation Limitations, _attach_supervised_probability(), _boolean_score(), build_discovery_ranking(), _crossing_geometry_score(), _dem_route_drainage_score() (+36 more)

### Community 5 - "Map App Core"
Cohesion: 0.09
Nodes (36): candidateCanvasColor(), candidateCanvasLabel(), compareCanvasCandidatePriority(), DATA_URLS, declutterCanvasItems(), definitionItem(), discoveryStatusLabel(), drainageLabel() (+28 more)

### Community 6 - "Field Report Parsing"
Cohesion: 0.09
Nodes (45): Match, _clean_text_line(), CoordinateRecord, _culvert_ids(), _deduplicate_records(), _docx_text(), extract_field_report_records(), _extract_from_files() (+37 more)

### Community 7 - "Dev Server API"
Cohesion: 0.12
Nodes (29): appendObservation(), canRead(), DATA_DIR, deleteObservation(), __dirname, emptyFeatureCollection(), fileInfo(), handleRequest() (+21 more)

### Community 8 - "CLI Pipeline"
Cohesion: 0.09
Nodes (33): ArgumentParser, _add_field_report_candidates(), _analyze_extracted_points(), _build_candidates(), _build_discovery_ranking(), _build_features(), _build_high_confidence_training_points(), build_parser() (+25 more)

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
Cohesion: 0.06
Nodes (79): Point, _county_boundary(), _download_if_missing(), download_ulster_census_inputs(), _normalize_linear_water(), _normalize_roads(), GeoDataFrame, Path (+71 more)

### Community 14 - "Web Export"
Cohesion: 0.20
Nodes (21): _decluster_for_web(), _drop_exported_candidates(), _export_key_value(), export_web_data(), _field_recall_export_pool(), _known_match_count(), _limit_for_web(), _prediction_export_pool() (+13 more)

### Community 15 - "Census Inputs"
Cohesion: 0.17
Nodes (22): LineString, create_demo_dataset(), _demo_known_culverts(), _demo_roads(), _demo_streams(), _move_to_ulster_pilot(), GeoDataFrame, Path (+14 more)

### Community 16 - "LLM Label Import"
Cohesion: 0.23
Nodes (12): Current Run Soft Voting Ensemble, Spatial Holdout Model Selection, Soft Voting Ensemble, Culvert AI Research Portfolio, Discovery Score Blend, Field Review Interface, LLMs Not Primary Predictor, Soft Voting Model Choice (+4 more)

### Community 17 - "Draft Point UI"
Cohesion: 0.20
Nodes (16): bindDraftPointActions(), cancelPlacePointMode(), clearDetail(), hideDetailPanel(), observationIcon(), observationIdOf(), observationLatLng(), openSelectedFeaturePopup() (+8 more)

### Community 18 - "DEM Acquisition"
Cohesion: 0.23
Nodes (13): dem_tiles_for_bounds(), _download_if_missing(), download_usgs_3dep_dem(), _expanded_bounds(), Path, Return USGS 3DEP tile IDs intersecting WGS84 bounds.      USGS 3DEP current elev, Download and mosaic USGS 3DEP DEM tiles covering a boundary layer., _tile_id() (+5 more)

### Community 19 - "Observation Labels"
Cohesion: 0.38
Nodes (8): filter_to_region(), get_region(), GeoDataFrame, Path, Region, region_boundary(), write_region_boundary(), test_filter_to_ulster_poughkeepsie_region()

### Community 20 - "Model Summary"
Cohesion: 0.42
Nodes (12): build_summary(), _csv_row_count(), _float_or_none(), _int_or_none(), main(), _nested_number(), _point_qc(), Any (+4 more)

### Community 21 - "Config Model Docs"
Cohesion: 0.32
Nodes (8): Example Candidate Generation Parameters, Example Feature Parameters, Example Model Parameters, Example Project Configuration, Ulster Candidate Generation Parameters, Ulster Model Parameters, Ulster Poughkeepsie Project Configuration, PyYAML

### Community 22 - "Field Reviewer System"
Cohesion: 0.33
Nodes (6): Browser Based Field Review Queue, Candidate Prediction Pipeline, Culvert AI Research Portfolio Contact Sheet, Culvert AI, Discovery Score, Field Reviewer System

### Community 23 - "Deeper Validation Roadmap"
Cohesion: 0.67
Nodes (3): Validation Roadmap, Data and Protocol Bottlenecks, Deeper Validation Roadmap

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
Cohesion: 0.28
Nodes (9): Geospatial ML Workflow, Spatial Holdout Validation, USGS 3DEP DEM Features, GeoPandas, Joblib, Python Geospatial ML Dependencies, Rasterio, Scikit Learn (+1 more)

### Community 29 - "Field Feedback Loop"
Cohesion: 0.13
Nodes (19): ABU User Observations, Continuous Retraining Trigger, Culvert AI Ulster County Pilot, Leaflet Field Review UI, Field-Useful Prediction, Field Review Ranking Logic, Strict 10 m Match Rule, Vercel Blob Persistence (+11 more)

### Community 30 - "Field Data State"
Cohesion: 0.22
Nodes (9): Field Report Input Manifest, Team 2 Selected Report Folder, Team 4 Daily Report PDFs, Field Report Coordinate Extraction, Current Selected Model Hist Gradient Boosting, Culvert AI Track, Current Data State, Current Goal (+1 more)

### Community 31 - "Research Notes"
Cohesion: 0.22
Nodes (9): Data Needed, Field Search Time Bottleneck, LLM Role Boundary, Research Notes, Research Plan, Accuracy Bottlenecks, LLM Not Primary Predictor, Bottlenecks (+1 more)

### Community 32 - "Prediction Workflow"
Cohesion: 0.27
Nodes (12): Candidate Generation, Culvert Prediction Model, Culvert Probability, Discovery Score, Feature Table, Hydrology and DEM Features, Interpretable GIS Evidence Score, Missed Road Feedback Rationale (+4 more)

### Community 35 - "Web Build Verify"
Cohesion: 0.33
Nodes (5): findings, modelSummary, requiredFiles, root, summary

## Knowledge Gaps
- **95 isolated node(s):** `ROOT`, `STATIC_FINDINGS_PATH`, `STATIC_SUMMARY_PATH`, `OBSERVATION_STATUSES`, `FEEDBACK_MATCH_RADIUS_M` (+90 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_feature_table()` connect `Feature Engineering` to `Prediction Workflow`, `Discovery Scoring`, `Point Analysis`, `Census Inputs`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `Python Geospatial ML Dependencies` connect `Modeling Dependencies` to `Prediction Workflow`, `Config Model Docs`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `Feature Table` connect `Prediction Workflow` to `Feature Engineering`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `build_feature_table()` (e.g. with `run_demo_pipeline()` and `add_wgs84_coordinates()`) actually correct?**
  _`build_feature_table()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `build_parser()` (e.g. with `_add_field_report_candidates()` and `_analyze_extracted_points()`) actually correct?**
  _`build_parser()` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `Point` (e.g. with `_demo_known_culverts()` and `import_field_reports()`) actually correct?**
  _`Point` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `write_vector()` (e.g. with `download_ulster_census_inputs()` and `create_demo_dataset()`) actually correct?**
  _`write_vector()` has 21 INFERRED edges - model-reasoned connections that need verification._