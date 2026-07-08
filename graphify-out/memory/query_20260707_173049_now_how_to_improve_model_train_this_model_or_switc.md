---
type: "query"
date: "2026-07-07T17:30:49.035013+00:00"
question: "Now how to improve model train this model or switch models to make this the best prediction possible"
contributor: "graphify"
outcome: "useful"
source_nodes: ["train_model", "_candidate_models", "_spatial_holdout_score", "build_feature_table", "score_unlabeled_candidates", "build_discovery_ranking"]
---

# Q: Now how to improve model train this model or switch models to make this the best prediction possible

## Answer

Expanded from original query via vocab: [model, training, train, feature, features, candidate, candidates, prediction, predict, probability, scoring, score, holdout, ensemble, supervised]. The graph and code point to train_model, _candidate_models, _spatial_holdout_score, build_feature_table, score_unlabeled_candidates, and build_discovery_ranking. Best next action is not blindly switching models: first run CULVERT_MODEL_FAMILY=auto npm run predict:actual to compare all built-in families by spatial holdout average precision, then improve labels, candidate coverage, DEM/hydrology features, official GIS layers, and field-confirmed negatives. Current local summary shows soft_voting_ensemble with spatial AP about 0.837 and P@10 1.0, but the current metrics file only contains the forced soft_voting run.

## Outcome

- Signal: useful

## Source Nodes

- train_model
- _candidate_models
- _spatial_holdout_score
- build_feature_table
- score_unlabeled_candidates
- build_discovery_ranking