import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { get } from "@vercel/blob";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = path.join(root, "data", "processed", "field_observations.geojson");
const blobPath = process.env.CULVERT_OBSERVATIONS_BLOB_PATH || "culvert-ai/field_observations.geojson";

if (!process.env.BLOB_READ_WRITE_TOKEN && !(process.env.VERCEL_OIDC_TOKEN && process.env.BLOB_STORE_ID)) {
  throw new Error("Set BLOB_READ_WRITE_TOKEN before pulling Vercel field observations.");
}

const result = await get(blobPath, { access: "private", useCache: false });
if (!result?.stream) {
  throw new Error(`No field observations found in Vercel Blob at ${blobPath}.`);
}

const body = await new Response(result.stream).text();
const parsed = JSON.parse(body);
if (parsed?.type !== "FeatureCollection" || !Array.isArray(parsed.features)) {
  throw new Error(`Blob ${blobPath} is not a GeoJSON FeatureCollection.`);
}

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(parsed, null, 2)}\n`, "utf8");

console.log(`Pulled ${parsed.features.length} Vercel observations to ${path.relative(root, outputPath)}.`);
