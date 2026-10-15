# Experimental staging

Two separate opt-in paths live here. Neither changes the main application or
automatically promotes a model. Recommendations require field verification.

## Hydrology-driven sites v2

```sh
scripts/python.sh scripts/serve_hydrology_staging.py --port 8094
```

Open `http://127.0.0.1:8094/`. `hydrology_sites.py`, `hydrology_service.py` and
`web/` implement actual road/drainage crossing proposals, independent road
profiles, topology-aware physical sites and optional candidate/hydrology layers.
Compatible experimental RF/HGB v2 artifacts and local GIS assets are required;
this server never trains or substitutes the old model.

The default physical-site layer is not a completeness claim. Dense internal
search reaches 700/705 in-footprint inventory records, while strict crossings
reach only 63/705. The retrospective match gains do not establish field accuracy
or justify promotion. See the
[implementation and evaluation report](../reports/experiments/hydrology_driven_sites_20261015/REPORT.md)
and [verification](../reports/experiments/hydrology_driven_sites_20261015/VERIFICATION.md).

## Fixed RF ranking-policy staging (separate offline path)

This is an opt-in, offline staging implementation. Nothing in the production
server, application, inference module, model, candidate generator, feature
extractor, training data, or ground truth was edited. The application continues
to use dynamic ROUTE-Z. No model was trained, tuned, promoted, or deployed.

Recommendations require field verification.

## Policies and exact change

`ranking-policy.json` defaults to `dynamic_route_z` (CURRENT). The explicit
`raw_rf` override selects CHALLENGER. Unknown policies and non-staging
configuration files fail closed. Switch back with `--policy dynamic_route_z`
or omit the override while using the supplied default configuration.

The staging adapter consumes the same complete scored candidate table under
both policies, without route filtering, quotas, thresholds, or candidate changes:

- CURRENT: the existing dynamic ROUTE-Z ordering, RF-score tie-break, then
  candidate-ID ascending tie-break; existing global score-ordered 20 m suppression.
- CHALLENGER: unnormalized RF descending, candidate-ID ascending; the *same*
  production suppression function selects representatives in that order.

The raw adapter forces the existing consolidator's fallback to `(RF - 0) / 1`
by setting both minimum route sample counts above the entire candidate count.
This is an identity mapping, not new normalization or calibration. It avoids
duplicating or changing the frozen consolidation code. Actual ROUTE-Z diagnostic
columns are restored separately after selection; `final_priority` is raw RF for
CHALLENGER and ROUTE-Z for CURRENT. Representatives retain their selected order.
All finite scores participate; the 20 m boundary remains inclusive, including
cross-road suppression. This is greedy suppression, not transitive clustering.

`predict_from_features` calls the original frozen scorer with unchanged feature
ordering, all 48 fields, missing-value treatment and preprocessing. Its model
file must match SHA-256
`bdefae5bc12723e17487001826eea706fb48065c33ec1b13b15e892ad7347486`.
Candidate generation and GIS enrichment remain upstream and unchanged. This
staging entry point starts at the existing complete-feature-table boundary;
it is not a new road-loader, application endpoint, or UI mode.

RF scores and percentiles are relative ranking values, not calibrated
probabilities. No 70% precision claim is supported.

## Run and reverse

From the repository root, use an existing complete feature table produced by
the frozen inference pipeline. It must include unique string `candidate_id`,
`route_id`, all 48 frozen feature names and native EPSG:32618 point geometry.
The original base GPKG alone lacks hydrology/crossing enrichments and is not a
complete inference feature table. Missing features fail rather than silently
substituting a different feature pipeline.

```sh
scripts/python.sh scripts/stage_ranking_policy.py --features /absolute/path/complete_features.gpkg --policy raw_rf --output reports/staging/my_raw_run
scripts/python.sh scripts/stage_ranking_policy.py --features /absolute/path/complete_features.gpkg --policy dynamic_route_z --output reports/staging/my_route_z_run
```

Each output directory must be new; existing outputs are never overwritten.
CSV and WGS84 GeoJSON exports allowlist fields, record policy and staging
environment, and include the field-verification notice. Do not import these
outputs into production without a separate review and explicit promotion.

The existing `physical_site_id` is rank-based and therefore is **not** a stable
inspection-status join key across policies. Use the exported `staging_site_key`
(policy plus representative candidate identity) for staging records. Even this
does not assert that different representatives under different policies are the
same physical site; cross-policy spatial reconciliation is a separate task.

## Reproduce validation

```sh
scripts/python.sh -m pytest tests/test_staging_ranking_policy.py tests/test_inference.py tests/test_candidates.py -q
scripts/python.sh scripts/verify_staging_ranking_policy.py --output reports/staging/new_policy_verification
```

The verifier reads preserved RF OOF scores, identical candidate IDs and native
application geometry. It uses the existing independent-at-each-K, maximum
one-to-one inventory matcher at 20 m. It never calls a training runner or fits a
model. It asserts exact saved representative identity/order under both policies,
15/100 and 33/250 for raw, 0/100 and 5/250 for ROUTE-Z, and 653/714 full-list
coverage under each. It tests shuffled inputs, CSV/GeoJSON round trips, rollback,
the real frozen RF on arbitrary unseen road-ID strings, and protected-file hashes.

The generated `software_check_features.parquet` permits a small CLI smoke check;
it contains a saved development feature matrix on renamed IDs, **not** independent
field observations or a new GIS feature-generation run:

```sh
scripts/python.sh scripts/stage_ranking_policy.py --features reports/staging/new_policy_verification/software_check_features.parquet --policy raw_rf --output reports/staging/new_policy_verification/cli_raw_check
```

See the [validation report](../reports/staging/ranking_policy_correction_20261015/REPORT.md)
for results, concentration, limitations and the deployment recommendation.
