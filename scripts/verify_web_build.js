import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const requiredFiles = [
  "web/index.html",
  "web/styles.css",
  "web/app.js",
  "web/data/findings.geojson",
  "web/data/route_count_source.geojson.gz",
  "web/data/summary.json",
  "web/data/model_summary.json",
  "api/findings.js",
  "api/health.js",
  "api/observations.js",
  "api/route-count.js",
  "api/cron/retrain.js",
  "api/summary.js",
];

for (const relativePath of requiredFiles) {
  const filePath = path.join(root, relativePath);
  if (!existsSync(filePath)) {
    throw new Error(`${relativePath} is missing. Run npm run predict:actual before deploying.`);
  }

  if (statSync(filePath).size === 0) {
    throw new Error(`${relativePath} is empty.`);
  }
}

const findings = JSON.parse(readFileSync(path.join(root, "web/data/findings.geojson"), "utf8"));
if (findings.type !== "FeatureCollection" || !Array.isArray(findings.features)) {
  throw new Error("web/data/findings.geojson must be a GeoJSON FeatureCollection.");
}

const routeCountSource = JSON.parse(
  gunzipSync(readFileSync(path.join(root, "web/data/route_count_source.geojson.gz"))).toString("utf8"),
);
if (routeCountSource.type !== "FeatureCollection" || !Array.isArray(routeCountSource.features)) {
  throw new Error("web/data/route_count_source.geojson.gz must be a gzipped GeoJSON FeatureCollection.");
}

const summary = JSON.parse(readFileSync(path.join(root, "web/data/summary.json"), "utf8"));
if (!Number.isFinite(Number(summary.rows))) {
  throw new Error("web/data/summary.json must include a numeric rows value.");
}
if (!Number.isFinite(Number(summary.route_count_rows))) {
  throw new Error("web/data/summary.json must include route_count_rows from export-web.");
}

const modelSummary = JSON.parse(readFileSync(path.join(root, "web/data/model_summary.json"), "utf8"));
if (typeof modelSummary.available !== "boolean") {
  throw new Error("web/data/model_summary.json must include an available boolean.");
}

console.log(
  `Verified static Vercel build assets: ${findings.features.length} findings, ${routeCountSource.features.length} route-count rows.`,
);

await Promise.all([
  import("../api/findings.js"),
  import("../api/health.js"),
  import("../api/observations.js"),
  import("../api/route-count.js"),
  import("../api/cron/retrain.js"),
  import("../api/summary.js"),
]);

console.log("Verified Vercel API handlers import cleanly.");
