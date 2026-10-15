import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { exportGeoJson, filterSites, priorityLabel, rowsToCsv, safeExportRows } from "../web/product-core.js";

const features = Array.from({ length: 300 }, (_, index) => ({
  type: "Feature",
  geometry: { type: "Point", coordinates: [-74, 42] },
  properties: {
    global_rank: index + 1,
    route_id: index % 2 ? "Unseen Road" : "Main Street",
    latitude: 42,
    longitude: -74,
    flow_c_score: 0.8,
    route_z_score: 1.2,
    candidate_id: `candidate-${index}`,
    physical_site_id: `site-${index}`,
    known_culvert_distance_m: 2,
    ground_truth_label: 1,
  },
}));

for (const limit of [25, 50, 100, 250]) assert.equal(filterSites(features, { limit }).length, limit);
assert.equal(filterSites(features, { limit: "all" }).length, 300);
const roadFiltered = filterSites(features, { limit: "all", road: "unseen" });
assert.deepEqual(roadFiltered.slice(0, 3).map((item) => item.properties.global_rank), [2, 4, 6]);
assert.equal(priorityLabel(25), "High Priority");
assert.equal(priorityLabel(26), "Medium Priority");

const observations = new Map([["site-0", { status: "confirmed_culvert" }]]);
const rows = safeExportRows(features.slice(0, 1), observations);
assert.equal(rows[0].inspection_status, "confirmed_culvert");
assert.equal(rows[0].recommendation_tier, "review");
assert.ok(!("ground_truth_label" in rows[0]));
assert.ok(!("known_culvert_distance_m" in rows[0]));
assert.match(rowsToCsv(rows), /inspection_status/);
const geojson = exportGeoJson(features.slice(0, 1), observations);
assert.equal(geojson.features[0].properties.global_rank, 1);
assert.ok(!JSON.stringify(geojson).includes("ground_truth"));
assert.equal(geojson.features[0].properties.recommendation_tier, "review");
const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
assert.match(html, /Use my location/);
assert.match(html, /Analyze this area/);
assert.ok(!html.includes('id="roadFile"'));
console.log("Verified product filters, preserved ranks, field status, and safe exports.");
