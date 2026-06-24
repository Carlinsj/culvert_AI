import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { get } from "@vercel/blob";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = path.join(root, "data", "processed", "field_observations.geojson");
const blobPath = process.env.CULVERT_OBSERVATIONS_BLOB_PATH || "culvert-ai/field_observations.geojson";
const observationsUrl =
  process.env.CULVERT_OBSERVATIONS_URL || "https://culvert-ai.vercel.app/api/observations";

const parsed = hasBlobCredentials()
  ? await readObservationsFromBlob()
  : await readObservationsFromApi();

if (parsed?.type !== "FeatureCollection" || !Array.isArray(parsed.features)) {
  throw new Error("Field observations response is not a GeoJSON FeatureCollection.");
}

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(parsed, null, 2)}\n`, "utf8");

console.log(`Pulled ${parsed.features.length} Vercel observations to ${path.relative(root, outputPath)}.`);

function hasBlobCredentials() {
  return Boolean(
    process.env.BLOB_READ_WRITE_TOKEN ||
      (process.env.VERCEL_OIDC_TOKEN && process.env.BLOB_STORE_ID),
  );
}

async function readObservationsFromBlob() {
  const result = await get(blobPath, { access: "private", useCache: false });
  if (!result?.stream) {
    throw new Error(`No field observations found in Vercel Blob at ${blobPath}.`);
  }

  const body = await new Response(result.stream).text();
  return JSON.parse(body);
}

async function readObservationsFromApi() {
  const response = await fetch(observationsUrl, {
    headers: {
      accept: "application/geo+json, application/json",
    },
  });
  if (!response.ok) {
    throw new Error(`Could not pull Vercel observations from ${observationsUrl}: HTTP ${response.status}.`);
  }
  return response.json();
}
