# Graph Report - culvert_AI  (2026-07-13)

## Corpus Check
- 72 files · ~120,006 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1061 nodes · 2567 edges · 56 communities (38 shown, 18 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 228 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d2b2fb17`
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
- [[_COMMUNITY_handleLocationSuccess|handleLocationSuccess]]
- [[_COMMUNITY_CLI Pipeline|CLI Pipeline]]
- [[_COMMUNITY_Node Project Config|Node Project Config]]
- [[_COMMUNITY_Visible Marker List|Visible Marker List]]
- [[_COMMUNITY_Detail Feedback UI|Detail Feedback UI]]
- [[_COMMUNITY_renderDetail|renderDetail]]
- [[_COMMUNITY_Point Analysis|Point Analysis]]
- [[_COMMUNITY_Web Export|Web Export]]
- [[_COMMUNITY_Census Inputs|Census Inputs]]
- [[_COMMUNITY_LLM Label Import|LLM Label Import]]
- [[_COMMUNITY_Draft Point UI|Draft Point UI]]
- [[_COMMUNITY_DEM Acquisition|DEM Acquisition]]
- [[_COMMUNITY_Observation Labels|Observation Labels]]
- [[_COMMUNITY_Model Summary|Model Summary]]
- [[_COMMUNITY_Config Model Docs|Config Model Docs]]
- [[_COMMUNITY_route-count.js|route-count.js]]
- [[_COMMUNITY_Deeper Validation Roadmap|Deeper Validation Roadmap]]
- [[_COMMUNITY_Vercel Config|Vercel Config]]
- [[_COMMUNITY_verify_security.js|verify_security.js]]
- [[_COMMUNITY_renderMovedOffsetForObservation|renderMovedOffsetForObservation]]
- [[_COMMUNITY_Continuous Learning Behavior|Continuous Learning Behavior]]
- [[_COMMUNITY_GeoDataFrame|GeoDataFrame]]
- [[_COMMUNITY_Field Data State|Field Data State]]
- [[_COMMUNITY_Research Notes|Research Notes]]
- [[_COMMUNITY_Path|Path]]
- [[_COMMUNITY_Series|Series]]
- [[_COMMUNITY_Vercel Observation Pull|Vercel Observation Pull]]
- [[_COMMUNITY_Web Build Verify|Web Build Verify]]
- [[_COMMUNITY_region.py|region.py]]
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
- [[_COMMUNITY_normalizeLongitude|normalizeLongitude]]
- [[_COMMUNITY_Soft Voting Ensemble|Soft Voting Ensemble]]
- [[_COMMUNITY_io.py|io.py]]
- [[_COMMUNITY_evaluate_success_rate_at_actuals|evaluate_success_rate_at_actuals]]
- [[_COMMUNITY_region.py|region.py]]
- [[_COMMUNITY_isMovedObservation|isMovedObservation]]

## God Nodes (most connected - your core abstractions)
1. `build_parser()` - 30 edges
2. `build_feature_table()` - 29 edges
3. `escapeHtml()` - 23 edges
4. `write_vector()` - 22 edges
5. `score_unlabeled_candidates()` - 22 edges
6. `bindControls()` - 21 edges
7. `renderList()` - 21 edges
8. `_records_from_text()` - 21 edges
9. `Culvert AI: Ulster County Pilot` - 20 edges
10. `scripts` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Spatial Candidate Workflow` --semantically_similar_to--> `Candidate Generation`  [INFERRED] [semantically similar]
  outputs/culvert-ai-research-portfolio.pdf → model.md
- `Supervised Classification Task` --semantically_similar_to--> `Supervised Model`  [INFERRED] [semantically similar]
  outputs/culvert-ai-research-portfolio.pdf → model.md
- `Data Needed` --semantically_similar_to--> `Accuracy Bottlenecks`  [INFERRED] [semantically similar]
  docs/research_notes.md → README.md
- `Model Improvement Query` --references--> `build_feature_table()`  [EXTRACTED]
  graphify-out/memory/query_20260707_173049_now_how_to_improve_model_train_this_model_or_switc.md → src/culvert_ai/features.py
- `Model Improvement Query` --references--> `score_unlabeled_candidates()`  [EXTRACTED]
  graphify-out/memory/query_20260707_173049_now_how_to_improve_model_train_this_model_or_switc.md → src/culvert_ai/scoring.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Contact Sheet Validation Roadmap** — outputs_culvert_ai_research_portfolio_contact_sheet_soft_voting_ensemble, outputs_culvert_ai_research_portfolio_contact_sheet_discovery_score, outputs_culvert_ai_research_portfolio_contact_sheet_validation_roadmap [INFERRED 0.85]

## Communities (56 total, 18 thin omitted)

### Community 0 - "Retraining API"
Cohesion: 0.08
Nodes (63): handler(), handler(), appendEvidenceSummary(), applyFeedbackToFindings(), applyMissedPredictionToFeature(), applyObservationToFeature(), baseFingerprint(), blobConfigured() (+55 more)

### Community 1 - "Feature Engineering"
Cohesion: 0.09
Nodes (58): add_approved_known_dem_similarity_features(), add_candidate_derived_features(), add_dem_culvert_terrain_features(), add_dem_hydrology_proxies(), add_hydrology_raster_features(), add_known_culvert_labels(), add_known_culvert_pattern_features(), add_known_culvert_pattern_score() (+50 more)

### Community 2 - "Candidate Generation"
Cohesion: 0.14
Nodes (33): build_route_count_report(), _cluster_predictions(), _cluster_probability_series(), _default_thresholds(), _empty_clusters(), _filter_by_segment(), _line_length_m(), _nearby_cluster_id() (+25 more)

### Community 3 - "Model Training"
Cohesion: 0.11
Nodes (44): ExtraTreesClassifier, Auto Model Family Comparison, Data Quality Improvement Levers, Model Improvement Query, HistGradientBoostingClassifier, ndarray, _balanced_hist_gradient_boosting(), _candidate_models() (+36 more)

### Community 4 - "Discovery Scoring"
Cohesion: 0.15
Nodes (43): _attach_supervised_probability(), _boolean_score(), build_discovery_ranking(), _crossing_geometry_score(), _dem_route_drainage_score(), _discovery_evidence_summary(), _drainage_strength_score(), _evidence_summary() (+35 more)

### Community 5 - "Map App Core"
Cohesion: 0.06
Nodes (48): bboxAroundLatLng(), buildSearchText(), candidateCanvasColor(), candidateCanvasLabel(), clampIntegerValue(), clampLatitude(), clampLongitude(), clampNumberValue() (+40 more)

### Community 6 - "Field Report Parsing"
Cohesion: 0.10
Nodes (45): Match, append_field_report_candidates(), _clean_text_line(), CoordinateRecord, _culvert_ids(), _deduplicate_coordinate_records(), _deduplicate_records(), _docx_text() (+37 more)

### Community 7 - "handleLocationSuccess"
Cohesion: 0.12
Nodes (32): bindControls(), cancelScheduledAutoRouteTargets(), clearRouteTargetMarkers(), currentMapBbox(), fetchRouteCount(), focusRouteCountTarget(), focusUserLocation(), handleLocationError() (+24 more)

### Community 8 - "CLI Pipeline"
Cohesion: 0.07
Nodes (63): ArgumentParser, _auto_sample_numbered_road(), CandidateSettings, _crossing_angle_degrees(), _deduplicate(), _first_numeric(), _first_value(), generate_candidates() (+55 more)

### Community 9 - "Node Project Config"
Cohesion: 0.07
Nodes (29): dependencies, @vercel/blob, description, engines, node, name, private, scripts (+21 more)

### Community 10 - "Visible Marker List"
Cohesion: 0.11
Nodes (33): appendObservation(), canRead(), DATA_DIR, deleteObservation(), __dirname, emptyFeatureCollection(), fileInfo(), handleRequest() (+25 more)

### Community 11 - "Detail Feedback UI"
Cohesion: 0.10
Nodes (46): bindDetailCloseAction(), bindFeedbackActions(), compactEvidenceSummary(), definitionItem(), detailCell(), discoveryStatusLabel(), draftPointSaveHtml(), drainageLabel() (+38 more)

### Community 12 - "renderDetail"
Cohesion: 0.32
Nodes (15): Point, ensure_parent_dir(), Path, read_vector(), write_vector(), merge_confirmed_observations(), write_high_confidence_training_points(), test_merge_confirmed_observations_adds_confirmed_points() (+7 more)

### Community 13 - "Point Analysis"
Cohesion: 0.24
Nodes (19): _analysis_flag(), analyze_extracted_points(), _attach_nearest_candidate(), _attach_nearest_line(), _candidate_score(), _cluster_ids(), _distance_stats(), _feature_name() (+11 more)

### Community 14 - "Web Export"
Cohesion: 0.18
Nodes (24): GeoDataFrame, Path, Series, _decluster_for_web(), _drop_exported_candidates(), _export_key_value(), export_web_data(), _field_recall_export_pool() (+16 more)

### Community 15 - "Census Inputs"
Cohesion: 0.19
Nodes (21): LineString, create_demo_dataset(), _demo_known_culverts(), _demo_roads(), _demo_streams(), _move_to_ulster_pilot(), GeoDataFrame, Path (+13 more)

### Community 16 - "LLM Label Import"
Cohesion: 0.06
Nodes (38): Example Candidate Generation Parameters, Example Feature Parameters, Example Model Parameters, Example Project Configuration, Ulster Candidate Generation Parameters, Ulster Model Parameters, Ulster Poughkeepsie Project Configuration, Candidate Generation (+30 more)

### Community 17 - "Draft Point UI"
Cohesion: 0.34
Nodes (16): _confirmed_observations_as_known(), _date_part(), _dedupe_observation_rows(), _denied_observations_as_negative(), _empty_observation_labels(), _field_negative_observations(), _first_non_empty_series(), _is_prediction_candidate_id() (+8 more)

### Community 18 - "DEM Acquisition"
Cohesion: 0.23
Nodes (13): dem_tiles_for_bounds(), _download_if_missing(), download_usgs_3dep_dem(), _expanded_bounds(), Path, Return USGS 3DEP tile IDs intersecting WGS84 bounds.      USGS 3DEP current elev, Download and mosaic USGS 3DEP DEM tiles covering a boundary layer., _tile_id() (+5 more)

### Community 19 - "Observation Labels"
Cohesion: 0.09
Nodes (49): bindDraftPointActions(), bindMovedRepairAction(), cancelPlacePointMode(), centerMapOnPoint(), clearDetail(), clearMovedOffsetOverlay(), compareFeaturesForList(), draftPointIcon() (+41 more)

### Community 20 - "Model Summary"
Cohesion: 0.42
Nodes (12): build_summary(), _csv_row_count(), _float_or_none(), _int_or_none(), main(), _nested_number(), _point_qc(), Any (+4 more)

### Community 21 - "Config Model Docs"
Cohesion: 0.25
Nodes (7): Changed Files, Current State, Handoff, Notes For Next Operator, Required Production Env Vars, Security Fixes Applied, Verification Completed

### Community 22 - "route-count.js"
Cohesion: 0.09
Nodes (45): buildRouteCountReport(), cellKey(), clamp(), clampInteger(), clampNumber(), clusterPredictions(), clusterProbability(), DEFAULT_PROBABILITY_THRESHOLDS (+37 more)

### Community 23 - "Deeper Validation Roadmap"
Cohesion: 0.67
Nodes (3): Validation Roadmap, Data and Protocol Bottlenecks, Deeper Validation Roadmap

### Community 24 - "Vercel Config"
Cohesion: 0.17
Nodes (11): includeFiles, maxDuration, buildCommand, crons, framework, functions, api/**/*.js, headers (+3 more)

### Community 25 - "verify_security.js"
Cohesion: 0.27
Nodes (7): bearerToken(), constantTimeTokenEquals(), isAuthorizedBearer(), requireFeedbackWriteAuth(), safeString(), handler(), previousEnv

### Community 26 - "renderMovedOffsetForObservation"
Cohesion: 0.20
Nodes (22): addObservation(), appendLocalObservationStatus(), applyDashboardData(), clearFeedbackWriteToken(), clearLocalObservations(), deleteObservationById(), fetchFeedbackWrite(), fetchFirst() (+14 more)

### Community 28 - "Continuous Learning Behavior"
Cohesion: 0.33
Nodes (6): Browser Based Field Review Queue, Candidate Prediction Pipeline, Culvert AI Research Portfolio Contact Sheet, Culvert AI, Discovery Score, Field Reviewer System

### Community 30 - "Field Data State"
Cohesion: 0.22
Nodes (8): Bottlenecks, Culvert AI Track, Current Data State, Current Goal, Current Model, How To Resume, Next Best Work, UI State

### Community 31 - "Research Notes"
Cohesion: 0.07
Nodes (26): Data Needed, Field Search Time Bottleneck, LLM Role Boundary, Research Notes, Research Plan, Accuracy Bottlenecks, Common Commands, Continuous Retraining Trigger (+18 more)

### Community 35 - "Web Build Verify"
Cohesion: 0.29
Nodes (6): findings, modelSummary, requiredFiles, root, routeCountSource, summary

### Community 36 - "region.py"
Cohesion: 0.28
Nodes (14): _clean_float(), _clean_string(), import_llm_reviewed_labels(), Any, Path, _queue_row(), _read_jsonl(), _review_id() (+6 more)

### Community 50 - "normalizeLongitude"
Cohesion: 0.16
Nodes (23): bucketFromScore(), distanceMeters(), featureLatLng(), fitVisibleMarkers(), idOf(), isRouteCandidateId(), knownFeatures(), latLngFromValues() (+15 more)

### Community 52 - "io.py"
Cohesion: 0.24
Nodes (14): _county_boundary(), _download_if_missing(), download_ulster_census_inputs(), _normalize_linear_water(), _normalize_roads(), GeoDataFrame, Path, Download actual county-level TIGER/Line roads and linear-water data for Ulster C (+6 more)

### Community 53 - "evaluate_success_rate_at_actuals"
Cohesion: 0.29
Nodes (10): _actual_id(), evaluate_predictions(), evaluate_success_rate_at_actuals(), _optional_number(), GeoDataFrame, Path, Measure field success as actual culverts with a prediction within max_distance_m, _unknown_prediction_pool() (+2 more)

### Community 54 - "region.py"
Cohesion: 0.38
Nodes (8): filter_to_region(), get_region(), GeoDataFrame, Path, Region, region_boundary(), write_region_boundary(), test_filter_to_ulster_poughkeepsie_region()

### Community 55 - "isMovedObservation"
Cohesion: 0.42
Nodes (9): fieldObservationView(), isConfirmedPredictionObservation(), isMovedObservation(), observationDisplayCode(), observationIcon(), observationStatus(), observationTitle(), saveRouteTargetReview() (+1 more)

## Knowledge Gaps
- **135 isolated node(s):** `DATA_URLS`, `SUMMARY_URLS`, `MODEL_SUMMARY_URLS`, `OBSERVATION_STATUSES`, `FIELD_VIEW_LABELS` (+130 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `write_vector()` connect `renderDetail` to `region.py`, `Point Analysis`, `Census Inputs`, `io.py`, `region.py`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `run_demo_pipeline()` connect `Census Inputs` to `Feature Engineering`, `Model Training`, `Discovery Scoring`, `CLI Pipeline`, `renderDetail`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `build_feature_table()` connect `Feature Engineering` to `CLI Pipeline`, `Model Training`, `Census Inputs`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `build_parser()` (e.g. with `_add_field_report_candidates()` and `_analyze_extracted_points()`) actually correct?**
  _`build_parser()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `build_feature_table()` (e.g. with `_build_features()` and `run_demo_pipeline()`) actually correct?**
  _`build_feature_table()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `write_vector()` (e.g. with `download_ulster_census_inputs()` and `create_demo_dataset()`) actually correct?**
  _`write_vector()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `DATA_URLS`, `SUMMARY_URLS`, `MODEL_SUMMARY_URLS` to the rest of the system?**
  _153 weakly-connected nodes found - possible documentation gaps or missing edges._