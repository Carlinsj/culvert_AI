"""Reproduce real Beaver Kill trace, fixed geometry sensitivity and report."""

import json
import shutil
import sys
from html import escape
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_hydrology_sites import DEFAULT_OUT as OUT, PRIOR, protected, save_json  # noqa: E402
from scripts.validate_inventory_heldout_roads import coverage, fold_for  # noqa: E402
from staging.hydrology_service import analyze, datasets  # noqa: E402
from staging.hydrology_sites import consolidate_events, detect_events  # noqa: E402
from culvert_ai import product_service  # noqa: E402


def table(frame):
    rows = [list(frame.columns), *frame.astype(str).values.tolist()]
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [rows[0], ["---"] * len(rows[0]), *rows[1:]]
    )


def feature_collection(payload):
    return gpd.GeoDataFrame.from_features(payload["features"], crs=4326).to_crs(32618)


def focal_location():
    roads, streams, _ = datasets()
    road = roads.loc[roads.name.fillna("").str.contains("212")].geometry.union_all()
    water = streams.loc[streams.name.fillna("").str.fullmatch("Beaver Kill")].copy()
    water["parallel_reach_m"] = water.geometry.intersection(road.buffer(100)).length
    chosen = water.sort_values(["parallel_reach_m", "id"], ascending=[False, True]).iloc[0]
    from shapely.ops import nearest_points

    point = nearest_points(road, chosen.geometry.interpolate(0.5, normalized=True))[0]
    wgs = gpd.GeoSeries([point], crs=32618).to_crs(4326).iloc[0]
    return dict(latitude=wgs.y, longitude=wgs.x, radius_m=1500), chosen


def demo():
    path = OUT / "beaver_after.json"
    location, chosen = focal_location()
    save_json(
        OUT / "beaver_location_selection.json",
        dict(
            **location,
            mapped_stream_id=str(chosen.id),
            adjacent_reach_m=chosen.parallel_reach_m,
            selection="Longest mapped Beaver Kill component within100m of roads named212; midpoint projected onto road. Representative example, no screenshot-location identification.",
        ),
    )
    if path.exists() and (OUT / "beaver_marker_trace.json").exists():
        return (
            location,
            json.loads((OUT / "beaver_before.json").read_text()),
            json.loads(path.read_text()),
        )
    captured = []
    original = product_service.predict_culverts

    def with_trace(*args, **kwargs):
        def capture(features):
            captured.append(features.copy())
            return features

        kwargs["feature_enricher"] = capture
        return original(*args, **kwargs)

    product_service.predict_culverts = with_trace
    try:
        before = product_service.analyze_nearby_payload(location)
    finally:
        product_service.predict_culverts = original
    after = analyze(**location)
    save_json(OUT / "beaver_before.json", before)
    save_json(path, after)
    features = captured[0]
    features.to_parquet(OUT / "beaver_original_features.parquet", index=False)
    sites = feature_collection(before["sites"])
    subset = sites.loc[sites.road_name.fillna("").str.contains("212")].sort_values("global_rank")
    selected = subset.iloc[0]
    row = features.set_index("candidate_id").loc[selected.candidate_id]
    trace = dict(
        candidate_id=selected.candidate_id,
        geometry_wgs84=list(
            before["sites"]["features"][int(selected.name)]["geometry"]["coordinates"]
        ),
        source=str(row.source),
        road_id=str(row.road_id),
        road_name=str(row.road_name),
        sample_chainage_m=float(row.route_sample_distance_m),
        model="FLOW-C Frozen v1,800trees,48features",
        flow_c_score=float(selected.flow_c_score),
        route_z_score=float(selected.route_z_score),
        global_rank=int(selected.global_rank),
        profile={
            c: float(row[c])
            for c in [
                "route_elevation_dip_60m",
                "route_elevation_dip_rank_pct",
                "route_low_point_score",
            ]
        },
        geographic_evidence={
            c: float(row[c]) if np.isfinite(row[c]) else None
            for c in [
                "distance_to_nearest_stream_m",
                "local_road_stream_angle_degrees",
                "distance_to_nearest_mapped_crossing_m",
            ]
        },
        visible_because="renderMapData loops all API sites; displayed list limit is not applied to map; global20m NMS does not remove75m samples",
        attribution_limit="Cannot prove a marker ranked highly solely because of river proximity without feature attribution/counterfactual analysis; distances and angle are retained, not causal claims.",
    )
    save_json(OUT / "beaver_marker_trace.json", trace)
    return location, before, after


def visual(location, before, after):
    center = (
        gpd.GeoSeries([Point(location["longitude"], location["latitude"])], crs=4326)
        .to_crs(32618)
        .iloc[0]
    )
    radius = location["radius_m"]
    roads = feature_collection(after["analysis"]["roads"])
    streams = feature_collection(after["hydrology"]["mapped"])
    drainage = feature_collection(after["hydrology"]["derived"])
    panels = []
    for offset, payload, title, subtitle in [
        (
            0,
            before,
            "Before · production75m search representatives",
            "All API sites drawn as markers",
        ),
        (
            700,
            after,
            "After · hydrology-supported physical sites",
            "Actual intersections; no coordinates moved",
        ),
    ]:
        parts = [
            f'<g transform="translate({offset},0)"><rect width="690" height="750" fill="#f2f4ec"/><text x="20" y="30" font-size="19" font-weight="bold">{escape(title)}</text><text x="20" y="54" font-size="14">{subtitle}</text><svg x="15" y="75" width="660" height="660" viewBox="0 0 660 660">'
        ]

        def xy(p):
            return (p[0] - center.x + radius) / (2 * radius) * 660, (center.y + radius - p[1]) / (
                2 * radius
            ) * 660

        for frame, color, width in [
            (roads, "#9ba69b", 2),
            (drainage, "#71bcb2", 1),
            (streams, "#347fae", 3),
        ]:
            for geometry in frame.geometry:
                lines = (
                    [geometry]
                    if geometry.geom_type == "LineString"
                    else list(getattr(geometry, "geoms", []))
                )
                for line in lines:
                    if line.geom_type != "LineString":
                        continue
                    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, line.coords))
                    parts.append(
                        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>'
                    )
        sites = feature_collection(payload["sites"])
        for row in sites.itertuples():
            x, y = xy((row.geometry.x, row.geometry.y))
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3 if offset == 0 else 5}" fill="{("#a36735" if offset == 0 else "#194f3f")}" stroke="white" stroke-width="1"><title>{escape(str(row.candidate_id))}</title></circle>'
            )
        parts.append(
            f'</svg><text x="20" y="735" font-size="13">{len(sites)} markers · identical extent · roads grey / mapped water blue / DEM teal</text></g>'
        )
        panels.append("".join(parts))
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1390" height="800" font-family="Arial" fill="#203b31">'
        + "".join(panels)
        + '<text x="20" y="780" font-size="15">Representative Beaver Kill / NY212 reach · real geographic data · Recommendations require field verification.</text></svg>'
    )
    (OUT / "beaver_before_after.svg").write_text(svg)


def sensitivity():
    output = OUT / "crossing_angle_sensitivity.csv"
    if output.exists():
        return
    roads = gpd.read_parquet(PRIOR / "recovered_road_segments.parquet")
    roads["id"], roads["route_id"] = roads.segment_id, roads.corridor
    _, streams, networks = datasets()
    loose, _ = detect_events(roads, streams, networks, minimum_angle=15)
    inventory = pd.read_csv(PRIOR / "inventory_split_manifest.csv")
    truth = gpd.GeoDataFrame(
        inventory, geometry=gpd.points_from_xy(inventory.source_x, inventory.source_y), crs=26918
    ).to_crs(32618)
    rows = []
    for angle in (15, 30, 45):
        events = loose.loc[loose.crossing_angle_degrees.ge(angle)].copy()
        physical = consolidate_events(events, np.zeros(len(events)))
        for fold in range(3):
            real = truth.loc[truth.fold.eq(fold)]
            selected = events.loc[events.route_id.map(fold_for).eq(fold)]
            sites = physical.loc[physical.route_id.map(fold_for).eq(fold)]
            rows.append(
                dict(
                    minimum_angle=angle,
                    fold=fold,
                    proposals=len(selected),
                    physical_sites=len(sites),
                    covered_inventory=int((coverage(selected, real) <= 20).sum()),
                    representative_covered=int((coverage(sites, real) <= 20).sum()),
                    official_records=len(real),
                )
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def report(location, before, after):
    metrics = pd.read_csv(OUT / "heldout_group_metrics.csv")
    selected = metrics.loc[metrics.k.eq(250)]
    summary = (
        selected.groupby(["cohort", "model", "policy"], sort=False)[
            [
                "matched",
                "actual_denominator",
                "raw_candidate_covered",
                "full_list_matches",
                "candidates",
                "consolidated_sites",
            ]
        ]
        .sum()
        .reset_index()
    )
    summary.to_csv(OUT / "top250_summary.csv", index=False)
    missing = pd.read_csv(OUT / "feature_missingness.csv")
    profiles = (
        missing.loc[
            missing.model.eq("RF")
            & missing.cohort.eq("repaired75_v2")
            & missing.feature.isin(
                ["route_elevation_dip_60m", "route_elevation_dip_rank_pct", "route_low_point_score"]
            )
        ]
        .groupby("feature")[["rows", "missing", "valid_zero"]]
        .sum()
        .reset_index()
    )
    profiles["valid_nonzero"] = profiles.rows - profiles.missing - profiles.valid_zero
    profiles.to_csv(OUT / "profile_repair_summary.csv", index=False)
    inventory = pd.read_csv(PRIOR / "inventory_split_manifest.csv")
    truth = gpd.GeoDataFrame(
        inventory, geometry=gpd.points_from_xy(inventory.source_x, inventory.source_y), crs=26918
    ).to_crs(32618)
    global_rows = []
    for name, path in [
        ("production75", PRIOR / "current75_features.parquet"),
        ("dense10", OUT / "dense_search_probes.parquet"),
        ("crossings", OUT / "crossing_events.parquet"),
    ]:
        frame = gpd.read_parquet(path)
        global_rows.append(
            dict(
                geometry=name,
                candidates=len(frame),
                native_union_covered_714=int((coverage(frame, truth) <= 20).sum()),
            )
        )
    pd.DataFrame(global_rows).to_csv(OUT / "native_union_coverage.csv", index=False)
    groups = selected.loc[
        selected.cohort.isin(["production75_v1", "physical_sites_v2"]),
        ["cohort", "fold", "model", "policy", "matched", "largest_route", "largest_route_share"],
    ]
    road = pd.read_csv(OUT / "heldout_road_metrics.csv")
    fixed = road.loc[
        road.cohort.eq("physical_sites_v2")
        & road.model.eq("RF")
        & road.policy.eq("raw_rf")
        & road.k.eq(25)
    ]
    neglected = (
        ",".join(
            fixed.loc[
                fixed.raw_candidate_covered.eq(0) & fixed.official_culverts.gt(0), "corridor"
            ].astype(str)
        )
        or "none"
    )
    script_files = [
        "staging/hydrology_sites.py",
        "staging/hydrology_service.py",
        "staging/web/index.html",
        "staging/web/app.js",
        "staging/web/core.js",
        "staging/web/styles.css",
        "scripts/evaluate_hydrology_sites.py",
        "scripts/serve_hydrology_staging.py",
        "scripts/report_hydrology_sites.py",
        "scripts/test_hydrology_web.mjs",
        "tests/test_hydrology_sites.py",
    ]
    content = (
        f"""# Hydrology-driven sites v2 — engineering implementation and retrospective validation

Date: 2026-10-15. Experimental staging only. **Do not deploy or promote.** Recommendations require field verification.

## Outcome and deployment decision

Uniform road samples no longer define the experimental visible site layer. New sites are real transverse, bilateral road/drainage intersections, consolidated using road topology and channel identity. Dense internal10m search restores candidate coverage; independent road profiles repair the75m shoulder bug. New v2 feature semantics are used only with versioned experimental RF/HGB copies trained with the same prior IDs, labels and fixed settings. No production model, production frontend, official inventory or protected file was changed.

This is **not** a field-ready replacement: strict crossing proposals cover just63/705 within their held-out groups, versus700/705 for internal dense search. The modest Top250-budget improvement does not compensate for this recall ceiling. Empty maps must not imply absence of culverts. No70% accuracy claim is supported.

## Root cause and complete marker trace

1. `src/culvert_ai/inference.py:231` calls the75m road generator, beginning at half the interval. These are unconditional search positions, not positive classifier decisions.
2. The frozen800-tree RF evaluates every row with its48-feature contract; ROUTE-Z changes relative ordering. There is no hydrological eligibility gate, calibrated decision threshold or equal-road recommendation quota.
3. Global greedy20m score-ordered suppression is much smaller than75m spacing, so most consecutive samples survive. The resulting "physical sites" are often simply isolated road probes.
4. `web/app.js` `renderMapData()` loops **all** returned site features. The list budget does not constrain the marker loop. Thus even low-ranked probes become visible across every road.
5. Existing60m profiles search left/right shoulders20–60m away among proposal rows. At75m spacing shoulders are absent, and all three features become0. This is an actual model-input distribution shift, not only a cosmetic rendering issue.

`beaver_marker_trace.json` traces one real State Rte212 marker through generated chainage, original48 inputs/profile zeros, frozen RF score, ROUTE-Z score/rank, original API and the marker loop. `beaver_original_features.parquet` preserves the full label-free input trace. We cannot infer causal feature attribution from proximity alone: the old model does have mapped crossing distances/angles, but nearby streams do not gate its recommendations.

## Implementation and contracts

- `staging/hydrology_sites.py`: deterministic10m **internal** probes; exact mapped and four DEM-network intersections; local tangent acute angle>=30° and water geometry on both sides within60m with2m side tolerance. Parallel-only proximity and collinear overlaps are insufficient. Fine DEM channels represent concentrated/convergent runoff; no fabricated ditch locations or arbitrary terrain-score boosts. The100,000m² smallest available network misses smaller drainage. No new unsupported independent depression/ditch detector was invented.
- 10m DEM road-profile measurements are taken on complete road parts independently of proposal spacing. Unavailable shoulders/raster data remainNaN; valid flat profiles remain0. Raw upstream area/flow accumulation are sampled and retained as evidence. Actual DEM edge-flow flags replace default0. DEM composites use training-only normalization references, missing components remain unavailable, and imputation uses only training medians. No new input columns are passed to old models.
- Contract version `hydrology-sites-v2` explicitly changes feature semantics despite retaining48 feature names. Only six held-out and two final **experimental** estimators accept it. RF uses an unchanged800-tree estimator specification; HGB uses unchanged HGB_REGULAR settings. Original fitted RF/HGB artifacts are read-only baselines. Exact previous training IDs/targets and HGB weights were reused, no model search or new label selection.
- Density normalization uses label-free dense road probes, not proposed crossings or inventory points. References and preprocessing are shared by offline and live code. Per-route density context depends on analyzed extent, which remains a validation limitation.
- Physical grouping: same road ID, road part and finest nearby DEM-channel identity; both Euclidean and road-chainage distances<=20m from an anchor; bounded clustering (no transitive chains). Different mapped-water IDs stay separate. Representative is an actual mapped intersection where available, otherwise an actual DEM intersection, deterministically tied by candidate ID. Coordinates are never jittered/snapped to inventory or moved to improve appearance. Preserve representative model score, member IDs, member count, source layers and maximum member score separately; ranking uses the representative score, not a hidden boost.
- Distinct road IDs/parts prevent merging parallel roads and hairpin recrossings. This is conservative: concurrent/fragmented road IDs or channel fragments can still produce duplicates. Water geometries can have map/topology errors.
- `staging/hydrology_service.py`: local arbitrary-road inference without any inventory reads, explicit v2 model guard, RF/HGB choice, measured evidence and null missing values. `scripts/serve_hydrology_staging.py` is a separate read-only loopback server, not a production route replacement.
- `staging/web/`: dependency-free geographic SVG map, ranked physical markers and matching list/export budget, hidden internal-candidate diagnostic layer rendered as one path (not thousands of interactive nodes), optional mapped/DEM/intersection layers, pan/zoom and traceable site details. Scores are relative, never probabilities. This staging diagnostic map has no street-name raster basemap; real road/water geometry is shown. Existing production frontend functionality is unchanged by isolation. Browser geolocation and supported-area jump are available, GeoJSON exports retain model/warning/provenance. No writes to original field-observation data.

## Retrospective protocol and results

All714 inventory records previously influenced development;701 directly influenced the original RF training. **Zero untouched independent test roads.** Reused fixed3 geometric corridor folds from the prior validation: whole held-out roads excluded from each experimental fit with the prior100m road buffer, same training manifests, no tuned hyperparameters, thresholds or model selection. Models/held-out geometries are evaluated separately; scores from different fold models are not pooled into a purported globalTop250. Each reported aggregate below is3 independent within-group budgets of250, total750 slots. Native20m matching is one-to-one for site-level results; inventory matches are not verified accuracy and unmatched sites are not confirmed false positives. Historical geometry/exposure limitations and inventory-versus-field negative conflicts remain.

Six fixed ablations distinguish75m feature repair, dense probes, hybrid candidates, geometric eligibility and consolidation. Candidate populations necessarily change for candidate-generation ablations; raw/RZ within each cohort use identical populations.

{table(summary)}

Columns: `raw_candidate_covered` and `full_list_matches` sum over705 in-footprint records in disjoint held-out groups; `matched` is Top250 **per group**; candidate/site counts sum the3 disjoint geometric cohorts. Event_queue has duplicate source-network intersections; physical_sites consolidates them. `hybrid_all` includes dense search plus exact intersections, but ungated global ranking still performs poorly; merely adding more candidates does not fix prioritization. No universal RF-vs-HGB winner was selected for production. Staging RF is an explicit research default, not a validated champion; HGB remains selectable.

Native geometry union over **all714** (allows a candidate from another fold, so not a held-out scoring claim):

{table(pd.DataFrame(global_rows))}

The production348/714 union is not contradictory to344/705 within-fold coverage. Nine inventory records lie outside the recovered road footprint and remain in the audit denominator, not silently excluded from claimed statewide coverage.

### Profile repair and missingness

Original11136 current75 rows had0 for each of the3 profile features. Repaired measurements on the same query geometry:

{table(profiles)}

`feature_missingness.csv` reports missing and valid-zero counts for every feature, family, fold and cohort. Missing values are retained until explicitly recorded training-only median imputation. Imputing an entirely missing training column falls back to0 only at estimator preparation, not as a claimed geographic measurement. No old model receives repaired features.

### Per-road concentration, spacing and sensitivity

{table(groups)}

Detailed `heldout_road_metrics.csv` includes every held-out corridor, candidate coverage, match rates, and fixed budgets. No per-road quotas exist. AtTop250 physical RF raw, largest-route shares are26.8%,38.0%,31.6%; rankings can still neglect roads with weak model scores. Roads with official inventory but **zero** crossing coverage here: {neglected}. This is more severe than ranking concentration and must block field promotion.

`recommendation_spacing.csv` reports nearest-neighbor distributions. Physical-site median spacings are approximately213/207/197m across the3 groups, not a forced target. Repaired75m queues still show near75m spacing, denseNMS queues near20–30m. Intersection-only queues contain repeated0m source duplicates; consolidation removes within-channel duplicates while preserving distinct nearby crossings. Spatial irregularity alone is not evidence of accuracy.

`matching_tolerance_sensitivity.csv`: fixed10/20/25m matching atTop250 per group; no radius chosen to maximize a headline. `crossing_angle_sensitivity.csv`: fixed15/30/45° crossing eligibility counts/coverage with no refitting or selection. Primary30° remains preregistered. Channel gate/DEM resolution/centerline offsets all constrain recall.

## Beaver Kill, actual before/after

No Beaver Kill screenshot was attached to this request; the earlier screenshots in this chat show a general coverage UI, not a uniquely identified Beaver Kill road segment. Representative coordinates were selected from actual data, not guessed as the exact user's view: {location["latitude"]:.6f}, {location["longitude"]:.6f}, radius1500m. See `beaver_location_selection.json` for reproducible longest-parallel-reach selection. No individual markers were adjusted.

Original live service: {before["site_count"]} returned/displayed markers. New research service: {after["diagnostics"]["internal_probe_count"]} internal probes, {after["diagnostics"]["crossing_proposals"]} actual crossing proposals, {after["site_count"]} physical sites. `beaver_before.json` and `beaver_after.json` contain actual API payloads. The new default display shows up to25 sites (in this area all{after["site_count"]}); internal probes are hidden. Mapped Beaver Kill adjacency does not generate a line of proposals; tributary/DEM lines must actually intersect transversely and extend to both road sides. Intersections may be bridges, map errors, or DEM artifacts, not necessarily culverts. Bridge context and edge-risk warnings are retained, not overruled.

![Actual same-extent before/after](beaver_before_after.svg)

## Remaining limitations and readiness

- Current crossing-only recall63/705 is unacceptable for a general culvert-finding product. Small unmapped/intermittent catchments, roadside ditches, embankment/bridge/culvert DEM artifacts,10m raster resolution, centerline registration and fragmented channel topology remain unresolved. Dense search keeps potential coverage internally but is deliberately not mislabeled as detected physical sites.
- Flow direction is used indirectly through existing precomputed DEM drainage/accumulation/edge networks; no unverified Whitebox D8 pointer decoding was invented. Two-sided mapped geometry and local angle are geometric connectivity evidence, not hydraulic proof. We do not have reliable surveyed upstream/downstream heads or a validated embankment-opening detector. No unsupported one-sided water-impoundment value is invented.
- Smallest supplied drainage network threshold0.1km² omits many small culvert catchments. Generating and independently validating finer catchments/topology is necessary before broad deployment; no post-hoc angle/model tuning on these exposed labels is claimed.
- Model input repair alone did not improve ranking consistently; dense coverage alone is insufficient. Stratified prospective field checks on newly collected, isolated roads and surveyed structure type are needed. None of the results establishes calibrated probabilities,70% accuracy, completeness, or safe autonomous field decisions.
- New models preserve fixed specifications but are different fitted experimental copies by design. Old models are not destroyed. Switching back means continuing the unchanged original app on8093; nothing was promoted.

## Changed files, integrity and reproduction

Only these **new** files were added for this task (pre-existing working-tree edits were preserved):

"""
        + "\n".join("- `" + name + "`" for name in script_files)
        + """

Versioned outputs/models live only in this new experiment directory. `input_integrity.json` and `protected_before.json` record all291 protected file SHA256s; unchanged on completion. The official workbook is read-only. `code_snapshot/` and `code_manifest.json` capture exact new sources used at handoff. Existing reports are not overwritten.

Reproduce from repository root (Python dependencies already installed):

```sh
scripts/python.sh scripts/evaluate_hydrology_sites.py --output /private/tmp/culvert_hydrology_reproduction
# --resume only for an interrupted, incomplete output; completed experiments reject overwrites.
# Report/staging default uses the fixed dated directory; edit constants only in a separate reproduction checkout if relocating outputs.
scripts/python.sh scripts/report_hydrology_sites.py
scripts/python.sh -m pytest -q
node --test scripts/test_hydrology_web.mjs
scripts/python.sh -m ruff check staging/hydrology_sites.py staging/hydrology_service.py scripts/evaluate_hydrology_sites.py scripts/serve_hydrology_staging.py scripts/report_hydrology_sites.py tests/test_hydrology_sites.py
scripts/python.sh scripts/serve_hydrology_staging.py --port 8094
```

POST `/api/analysis/nearby` with `{"location":{"latitude":42.07647,"longitude":-74.207172},"radius_m":1500,"model":"RF"}`. RF orHGB must load the matching v2 contract; no fallback to old model. The first request computes real GIS features; subsequent identical requests are cached. Server binds loopback only. No `npm run dev`, training hook, remote deployment or production mutation is required.

Regression and actual browser verification evidence is recorded in `VERIFICATION.md`. Recommendations require field verification.
"""
    )
    (OUT / "REPORT.md").write_text(content)
    snapshots = {}
    for name in script_files:
        legacy = OUT / "code_snapshot" / name
        destination = legacy.with_name(legacy.name + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Snapshots are evidence, not executable modules/test-discovery inputs.
        if legacy.exists():
            legacy.rename(destination)
        shutil.copyfile(ROOT / name, destination)
        import hashlib

        snapshots[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    save_json(OUT / "code_manifest.json", snapshots)


def main():
    protected()
    location, before, after = demo()
    visual(location, before, after)
    sensitivity()
    report(location, before, after)
    protected()
    print(f"Saved report and actual before/after diagnostics: {OUT}", flush=True)


if __name__ == "__main__":
    main()
