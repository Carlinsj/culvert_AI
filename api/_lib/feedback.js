import { readFile } from "node:fs/promises";
import path from "node:path";

import { get, put } from "@vercel/blob";

const ROOT = process.cwd();
const STATIC_FINDINGS_PATH = path.join(ROOT, "web", "data", "findings.geojson");
const STATIC_SUMMARY_PATH = path.join(ROOT, "web", "data", "summary.json");

const OBSERVATIONS_BLOB_PATH =
  process.env.CULVERT_OBSERVATIONS_BLOB_PATH || "culvert-ai/field_observations.geojson";
const FINDINGS_BLOB_PATH = process.env.CULVERT_FINDINGS_BLOB_PATH || "culvert-ai/findings.geojson";
const SUMMARY_BLOB_PATH = process.env.CULVERT_SUMMARY_BLOB_PATH || "culvert-ai/summary.json";

const OBSERVATION_STATUSES = new Set(["confirmed_culvert", "no_culvert", "uncertain"]);
const FEEDBACK_MATCH_RADIUS_M = Number(process.env.CULVERT_FEEDBACK_MATCH_RADIUS_M || 10);

export function blobConfigured() {
  return Boolean(
    process.env.BLOB_READ_WRITE_TOKEN ||
      (process.env.VERCEL_OIDC_TOKEN && process.env.BLOB_STORE_ID),
  );
}

export async function readObservations() {
  const stored = await readBlobJson(OBSERVATIONS_BLOB_PATH);
  return ensureFeatureCollection(stored);
}

export async function saveObservation(payload) {
  const feature = observationFeature(payload);
  const collection = await readObservations();
  const features = mergeObservations([...collection.features, feature]);
  const observations = { type: "FeatureCollection", features };

  if (!blobConfigured()) {
    return {
      feature,
      observations,
      storage: "memory",
      warning: "Vercel Blob is not configured. Set BLOB_READ_WRITE_TOKEN to persist observations.",
    };
  }

  await writeBlobJson(OBSERVATIONS_BLOB_PATH, observations, "application/geo+json");
  return { feature, observations, storage: "vercel_blob" };
}

export async function deleteObservation(observationId) {
  const id = safeString(observationId, 80);
  if (!id) {
    throw new Error("Observation id is required.");
  }

  const collection = await readObservations();
  const features = collection.features.filter((feature) => feature.properties?.observation_id !== id);
  if (features.length === collection.features.length) {
    throw new Error(`Observation not found: ${id}`);
  }

  const observations = { type: "FeatureCollection", features };
  if (!blobConfigured()) {
    return {
      observation_id: id,
      observations,
      storage: "memory",
      warning: "Vercel Blob is not configured. Set BLOB_READ_WRITE_TOKEN to persist observations.",
    };
  }

  await writeBlobJson(OBSERVATIONS_BLOB_PATH, observations, "application/geo+json");
  return { observation_id: id, observations, storage: "vercel_blob" };
}

export async function loadPublishedData({ refresh = false } = {}) {
  const [baseFindings, baseSummary, observations] = await Promise.all([
    readStaticFindings(),
    readStaticSummary(),
    readObservations(),
  ]);
  const fingerprint = baseFingerprint(baseFindings);

  if (!refresh && blobConfigured()) {
    const [storedSummary, storedFindings] = await Promise.all([
      readBlobJson(SUMMARY_BLOB_PATH),
      readBlobJson(FINDINGS_BLOB_PATH),
    ]);

    if (
      storedFindings?.type === "FeatureCollection" &&
      storedSummary?.feedback?.baseFingerprint === fingerprint
    ) {
      return {
        findings: storedFindings,
        summary: storedSummary,
        observations,
        storage: "vercel_blob",
      };
    }
  }

  const findings = applyFeedbackToFindings(baseFindings, observations);
  const summary = summarizeFindings(findings, observations, baseSummary, fingerprint);

  if (blobConfigured() && observations.features.length > 0) {
    await Promise.all([
      writeBlobJson(FINDINGS_BLOB_PATH, findings, "application/geo+json"),
      writeBlobJson(SUMMARY_BLOB_PATH, summary, "application/json"),
    ]);
    return { findings, summary, observations, storage: "vercel_blob" };
  }

  return { findings, summary, observations, storage: "static" };
}

export async function refreshPublishedData(observations) {
  const [baseFindings, baseSummary] = await Promise.all([readStaticFindings(), readStaticSummary()]);
  const fingerprint = baseFingerprint(baseFindings);
  const findings = applyFeedbackToFindings(baseFindings, observations);
  const summary = summarizeFindings(findings, observations, baseSummary, fingerprint);

  if (blobConfigured()) {
    await Promise.all([
      writeBlobJson(FINDINGS_BLOB_PATH, findings, "application/geo+json"),
      writeBlobJson(SUMMARY_BLOB_PATH, summary, "application/json"),
    ]);
  }

  return { findings, summary };
}

async function readStaticFindings() {
  return JSON.parse(await readFile(STATIC_FINDINGS_PATH, "utf8"));
}

async function readStaticSummary() {
  return JSON.parse(await readFile(STATIC_SUMMARY_PATH, "utf8"));
}

async function readBlobJson(pathname) {
  if (!blobConfigured()) return null;

  try {
    const result = await get(pathname, { access: "private", useCache: false });
    if (!result?.stream) return null;
    return JSON.parse(await new Response(result.stream).text());
  } catch (error) {
    if (error?.name === "BlobNotFoundError" || /not found/i.test(String(error?.message || ""))) {
      return null;
    }
    throw error;
  }
}

async function writeBlobJson(pathname, payload, contentType) {
  const body = `${JSON.stringify(payload, null, 2)}\n`;
  return put(pathname, body, {
    access: "private",
    allowOverwrite: true,
    cacheControlMaxAge: 0,
    contentType,
  });
}

function ensureFeatureCollection(collection) {
  if (collection?.type === "FeatureCollection" && Array.isArray(collection.features)) {
    return {
      type: "FeatureCollection",
      features: mergeObservations(collection.features),
    };
  }

  return emptyFeatureCollection();
}

function emptyFeatureCollection() {
  return { type: "FeatureCollection", features: [] };
}

function observationFeature(payload) {
  const props = payload?.properties || payload || {};
  const coordinates = payload?.geometry?.coordinates || [props.longitude, props.latitude];
  const longitude = Number(props.longitude ?? coordinates[0]);
  const latitude = Number(props.latitude ?? coordinates[1]);

  if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
    throw new Error("Observation needs a valid latitude.");
  }
  if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
    throw new Error("Observation needs a valid longitude.");
  }

  const status = OBSERVATION_STATUSES.has(props.status) ? props.status : "uncertain";
  const observedAt = safeString(props.observed_at, 48) || new Date().toISOString();

  return {
    type: "Feature",
    properties: {
      observation_id: safeString(props.observation_id, 80) || makeObservationId(),
      observed_at: observedAt,
      status,
      candidate_id: safeString(props.candidate_id, 120),
      road_name: safeString(props.road_name, 180),
      stream_name: safeString(props.stream_name, 180),
      notes: safeString(props.notes, 2000),
      source: safeString(props.source, 80) || "field_review",
      prediction_score: numberOrNull(props.prediction_score),
      priority_rank: numberOrNull(props.priority_rank),
      priority_bucket: safeString(props.priority_bucket, 40),
      field_culvert_id: safeString(props.field_culvert_id, 80),
      layout_source: safeString(props.layout_source, 80),
      layout_scan_summary: safeString(props.layout_scan_summary, 600),
      nearest_candidate_id: safeString(props.nearest_candidate_id, 120),
      nearest_candidate_distance_m: numberOrNull(props.nearest_candidate_distance_m),
      missed_candidate_id: safeString(props.missed_candidate_id, 120),
      missed_candidate_distance_m: numberOrNull(props.missed_candidate_distance_m),
      inferred_from_candidate: numberOrNull(props.inferred_from_candidate),
      latitude,
      longitude,
    },
    geometry: {
      type: "Point",
      coordinates: [longitude, latitude],
    },
  };
}

function mergeObservations(features) {
  const byId = new Map();
  features.map(normalizeObservationFeature).filter(Boolean).forEach((feature) => {
    byId.set(feature.properties.observation_id, feature);
  });

  return [...byId.values()].sort((a, b) =>
    String(b.properties.observed_at || "").localeCompare(String(a.properties.observed_at || "")),
  );
}

function normalizeObservationFeature(feature) {
  try {
    return observationFeature(feature);
  } catch {
    return null;
  }
}

function applyFeedbackToFindings(baseFindings, observations) {
  const features = cloneFeatures(baseFindings.features || []);
  const byCandidateId = new Map(
    features
      .map((feature) => [String(feature.properties?.candidate_id || ""), feature])
      .filter(([candidateId]) => candidateId),
  );
  const matchedObservationIds = new Set();

  for (const observation of observations.features || []) {
    const props = observation.properties || {};
    const missedCandidate = missedCandidateForObservation(props, byCandidateId);
    if (missedCandidate) {
      applyMissedPredictionToFeature(missedCandidate, observation);
      matchedObservationIds.add(props.observation_id);
    }

    const direct = props.candidate_id ? byCandidateId.get(String(props.candidate_id)) : null;
    const nearest = direct || nearestFeature(features, observation, FEEDBACK_MATCH_RADIUS_M);

    if (nearest) {
      applyObservationToFeature(nearest, observation);
      matchedObservationIds.add(props.observation_id);
    } else if (props.status === "confirmed_culvert") {
      features.push(featureFromConfirmedObservation(observation));
      matchedObservationIds.add(props.observation_id);
    }
  }

  const ranked = rerankFeatures(features);
  ranked.forEach((feature) => {
    const props = feature.properties || {};
    if (props.field_feedback_observation_id) {
      props.field_feedback_applied = 1;
    }
  });

  return {
    type: "FeatureCollection",
    features: ranked,
    feedback: {
      observations: observations.features.length,
      appliedObservations: matchedObservationIds.size,
      refreshedAt: new Date().toISOString(),
      mode: "deployed_feedback_ranking",
    },
  };
}

function missedCandidateForObservation(props, byCandidateId) {
  if (props.status !== "confirmed_culvert") return null;
  const candidateId = safeString(props.missed_candidate_id || props.nearest_candidate_id, 120);
  const distance = numberOrNull(props.missed_candidate_distance_m ?? props.nearest_candidate_distance_m);
  if (!candidateId || !Number.isFinite(distance) || distance <= FEEDBACK_MATCH_RADIUS_M) {
    return null;
  }
  return byCandidateId.get(candidateId) || null;
}

function cloneFeatures(features) {
  return features.map((feature) => ({
    ...feature,
    properties: { ...(feature.properties || {}) },
    geometry: feature.geometry ? { ...feature.geometry } : null,
  }));
}

function nearestFeature(features, observation, radiusM) {
  const obsLat = Number(observation.properties?.latitude);
  const obsLon = Number(observation.properties?.longitude);
  if (!Number.isFinite(obsLat) || !Number.isFinite(obsLon)) return null;

  let best = null;
  let bestDistance = Infinity;
  for (const feature of features) {
    const props = feature.properties || {};
    const lat = Number(props.latitude ?? feature.geometry?.coordinates?.[1]);
    const lon = Number(props.longitude ?? feature.geometry?.coordinates?.[0]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    const distance = distanceMeters(obsLat, obsLon, lat, lon);
    if (distance < bestDistance) {
      best = feature;
      bestDistance = distance;
    }
  }

  return bestDistance <= radiusM ? best : null;
}

function applyObservationToFeature(feature, observation) {
  const props = feature.properties || {};
  const obs = observation.properties || {};
  const status = obs.status;

  props.field_feedback_status = status;
  props.field_feedback_observation_id = obs.observation_id;
  props.field_feedback_at = obs.observed_at;
  props.field_feedback_notes = obs.notes || "";

  if (status === "confirmed_culvert") {
    props.is_known_field_match = 1;
    props.is_culvert = 1;
    props.discovery_status = "confirmed_field_observation";
    props.discovery_score = Math.max(numberOrZero(props.discovery_score), 98);
    props.culvert_likelihood_score = Math.max(numberOrZero(props.culvert_likelihood_score), 98);
    props.model_probability_score = Math.max(numberOrZero(props.model_probability_score), 98);
    props.field_report_support_score = 1;
    props.evidence_summary = appendEvidenceSummary(
      props.evidence_summary,
      "confirmed field observation; retraining label collected",
    );
    props.nearest_field_report_culvert_id = obs.field_culvert_id || obs.candidate_id || obs.observation_id;
    props.nearest_field_report_source_file = "vercel_field_observations";
    props.nearest_field_report_date = safeString(obs.observed_at, 10);
    props.field_added_culvert_id = obs.field_culvert_id || "";
    props.field_layout_source = obs.layout_source || "";
    return;
  }

  if (status === "no_culvert") {
    props.is_known_field_match = 0;
    props.is_culvert = 0;
    props.discovery_status = "field_denied";
    props.discovery_score = 0;
    props.culvert_likelihood_score = 0;
    props.model_probability_score = 0;
    props.field_report_support_score = 0;
    props.priority_bucket = "low";
    props.evidence_summary = "field observation says no culvert; deprioritized for retraining";
    return;
  }

  props.discovery_status = props.discovery_status || "field_uncertain";
  props.evidence_summary = appendEvidenceSummary(props.evidence_summary, "field observation uncertain");
}

function applyMissedPredictionToFeature(feature, observation) {
  const props = feature.properties || {};
  const obs = observation.properties || {};
  const distance = numberOrNull(obs.missed_candidate_distance_m ?? obs.nearest_candidate_distance_m);

  props.field_feedback_status = "missed_prediction";
  props.field_feedback_observation_id = obs.observation_id;
  props.field_feedback_at = obs.observed_at;
  props.field_feedback_notes = appendEvidenceSummary(
    obs.notes || "",
    Number.isFinite(distance)
      ? `confirmed field culvert was ${distance.toFixed(1)} m away`
      : "confirmed field culvert was outside the hit radius",
  );
  props.discovery_status = "field_missed_prediction";
  props.discovery_score = Math.min(numberOrZero(props.discovery_score), 15);
  props.culvert_likelihood_score = Math.min(numberOrZero(props.culvert_likelihood_score), 15);
  props.model_probability_score = Math.min(numberOrZero(props.model_probability_score), 15);
  props.field_report_support_score = 0;
  props.priority_bucket = "low";
  props.missed_by_observation_id = obs.observation_id;
  props.missed_by_distance_m = distance;
  props.evidence_summary = appendEvidenceSummary(
    props.evidence_summary,
    Number.isFinite(distance)
      ? `field-confirmed culvert was ${distance.toFixed(1)} m away; not counted as a hit`
      : "field-confirmed culvert was outside the hit radius; not counted as a hit",
  );
}

function featureFromConfirmedObservation(observation) {
  const obs = observation.properties || {};
  const candidateId = obs.field_culvert_id || obs.candidate_id || obs.observation_id;
  const latitude = Number(obs.latitude);
  const longitude = Number(obs.longitude);

  return {
    type: "Feature",
    properties: {
      candidate_id: candidateId,
      discovery_rank: 0,
      discovery_score: 100,
      discovery_status: "confirmed_field_observation",
      is_known_field_match: 1,
      evidence_score: 100,
      model_probability_score: 100,
      model_rank_score: 100,
      priority_rank: 0,
      priority_bucket: "very_high",
      culvert_likelihood_score: 100,
      culvert_probability: 1,
      road_name: obs.road_name || "Field observation",
      stream_name: obs.stream_name || "",
      source: "vercel_field_observation",
      evidence_summary: "confirmed field observation; retraining label collected",
      google_earth_url: `https://earth.google.com/web/search/${latitude},${longitude}`,
      latitude,
      longitude,
      field_report_support_score: 1,
      is_culvert: 1,
      nearest_field_report_date: safeString(obs.observed_at, 10),
      nearest_field_report_culvert_id: candidateId,
      nearest_field_report_source_file: "vercel_field_observations",
      field_added_culvert_id: obs.field_culvert_id || "",
      field_layout_source: obs.layout_source || "",
      field_feedback_status: "confirmed_culvert",
      field_feedback_observation_id: obs.observation_id,
      field_feedback_at: obs.observed_at,
      field_feedback_notes: obs.notes || "",
    },
    geometry: {
      type: "Point",
      coordinates: [longitude, latitude],
    },
  };
}

function rerankFeatures(features) {
  const ranked = features
    .slice()
    .sort((a, b) => {
      const aProps = a.properties || {};
      const bProps = b.properties || {};
      const aKnown = truthy(aProps.is_known_field_match) ? 1 : 0;
      const bKnown = truthy(bProps.is_known_field_match) ? 1 : 0;
      const aDenied = aProps.discovery_status === "field_denied" ? 1 : 0;
      const bDenied = bProps.discovery_status === "field_denied" ? 1 : 0;
      if (aDenied !== bDenied) return aDenied - bDenied;
      if (aKnown !== bKnown) return aKnown - bKnown;
      return numberOrZero(bProps.discovery_score) - numberOrZero(aProps.discovery_score);
    });

  ranked.forEach((feature, index) => {
    const props = feature.properties || {};
    props.discovery_rank = index + 1;
    props.priority_rank = props.discovery_rank;
    props.priority_bucket = bucketForScore(numberOrZero(props.discovery_score));
  });

  return ranked;
}

function summarizeFindings(findings, observations, baseSummary, fingerprint) {
  const features = findings.features || [];
  const knownMatches = features.filter((feature) => truthy(feature.properties?.is_known_field_match)).length;
  const buckets = {};
  for (const feature of features) {
    const bucket = String(feature.properties?.priority_bucket || "unknown");
    buckets[bucket] = (buckets[bucket] || 0) + 1;
  }

  const discoveryScores = features
    .filter((feature) => !truthy(feature.properties?.is_known_field_match))
    .map((feature) => Number(feature.properties?.discovery_score))
    .filter(Number.isFinite);
  const feedbackCounts = countObservationStatuses(observations);

  return {
    ...baseSummary,
    rows: features.length,
    discovery_candidates: features.length - knownMatches,
    known_field_matches: knownMatches,
    score_column: "discovery_score",
    max_score: discoveryScores.length ? Math.max(...discoveryScores) : null,
    mean_score: discoveryScores.length
      ? discoveryScores.reduce((sum, score) => sum + score, 0) / discoveryScores.length
      : null,
    priority_buckets: buckets,
    bounds: boundsForFeatures(features) || baseSummary.bounds,
    feedback: {
      mode: "deployed_feedback_ranking",
      baseFingerprint: fingerprint,
      observations: observations.features.length,
      confirmed_culverts: feedbackCounts.confirmed_culvert || 0,
      denied_culverts: feedbackCounts.no_culvert || 0,
      uncertain: feedbackCounts.uncertain || 0,
      refreshedAt: new Date().toISOString(),
      note: "Deployed feedback is applied to the served ranking immediately. Full supervised retraining uses this same observation GeoJSON in the Python pipeline.",
    },
  };
}

function countObservationStatuses(observations) {
  const counts = {};
  for (const feature of observations.features || []) {
    const status = observationStatus(feature.properties?.status);
    counts[status] = (counts[status] || 0) + 1;
  }
  return counts;
}

function boundsForFeatures(features) {
  let minLon = Infinity;
  let minLat = Infinity;
  let maxLon = -Infinity;
  let maxLat = -Infinity;

  for (const feature of features) {
    const lon = Number(feature.properties?.longitude ?? feature.geometry?.coordinates?.[0]);
    const lat = Number(feature.properties?.latitude ?? feature.geometry?.coordinates?.[1]);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
    minLon = Math.min(minLon, lon);
    minLat = Math.min(minLat, lat);
    maxLon = Math.max(maxLon, lon);
    maxLat = Math.max(maxLat, lat);
  }

  return Number.isFinite(minLon) ? [minLon, minLat, maxLon, maxLat] : null;
}

function baseFingerprint(findings) {
  const features = findings.features || [];
  const first = features[0]?.properties?.candidate_id || "";
  const last = features.at(-1)?.properties?.candidate_id || "";
  return `${features.length}:${first}:${last}`;
}

function appendEvidenceSummary(current, addition) {
  const text = safeString(current, 600);
  if (!text) return addition;
  if (text.includes(addition)) return text;
  return `${text}; ${addition}`;
}

function bucketForScore(score) {
  if (score > 75) return "very_high";
  if (score > 55) return "high";
  if (score > 35) return "medium";
  return "low";
}

function observationStatus(status) {
  return OBSERVATION_STATUSES.has(status) ? status : "uncertain";
}

function distanceMeters(latA, lonA, latB, lonB) {
  const radius = 6_371_000;
  const phiA = toRadians(latA);
  const phiB = toRadians(latB);
  const deltaPhi = toRadians(latB - latA);
  const deltaLambda = toRadians(lonB - lonA);
  const a =
    Math.sin(deltaPhi / 2) ** 2 +
    Math.cos(phiA) * Math.cos(phiB) * Math.sin(deltaLambda / 2) ** 2;
  return 2 * radius * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function toRadians(value) {
  return (value * Math.PI) / 180;
}

function truthy(value) {
  return value === true || value === 1 || value === "1" || String(value).toLowerCase() === "true";
}

function safeString(value, maxLength) {
  return String(value || "").trim().slice(0, maxLength);
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function numberOrZero(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function makeObservationId() {
  return `obs_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}
