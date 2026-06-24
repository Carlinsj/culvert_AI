import { createHash } from "node:crypto";

import { get, put } from "@vercel/blob";

import { blobConfigured } from "./feedback.js";

const RETRAIN_STATE_BLOB_PATH =
  process.env.CULVERT_RETRAIN_STATE_BLOB_PATH || "culvert-ai/retrain_state.json";
const DEFAULT_MIN_INTERVAL_SECONDS = 15 * 60;

export function retrainAutomationConfigured() {
  return Boolean(
    process.env.CULVERT_RETRAIN_WEBHOOK_URL ||
      (process.env.GITHUB_RETRAIN_TOKEN && process.env.GITHUB_REPOSITORY),
  );
}

export async function maybeTriggerRetrain({ observations, reason = "field_observation", force = false } = {}) {
  const payload = retrainPayload(observations, reason);
  const minIntervalSeconds = retrainMinIntervalSeconds();

  if (!blobConfigured() && process.env.CULVERT_RETRAIN_ALLOW_VOLATILE_OBSERVATIONS !== "1") {
    return {
      configured: retrainAutomationConfigured(),
      status: "not_persisted",
      ...payloadSummary(payload),
      note:
        "Observation feedback was not queued for retraining because Vercel Blob persistence is not configured.",
    };
  }

  if (!retrainAutomationConfigured()) {
    return {
      configured: false,
      status: "not_configured",
      ...payloadSummary(payload),
      note:
        "Set CULVERT_RETRAIN_WEBHOOK_URL or GITHUB_RETRAIN_TOKEN plus GITHUB_REPOSITORY to queue automatic retraining.",
    };
  }

  let state = null;
  let stateWarning = "";
  try {
    state = await readRetrainState();
  } catch (error) {
    stateWarning = error.message || "Could not read retrain state.";
  }

  const nowMs = Date.now();
  const lastDispatchFailed = state?.last_status === "failed";
  if (
    !force &&
    !lastDispatchFailed &&
    state?.last_observation_fingerprint === payload.observation_fingerprint
  ) {
    return {
      configured: true,
      status: "unchanged",
      ...payloadSummary(payload),
      state_warning: stateWarning || undefined,
      note: "The observation set is unchanged since the last queued retrain.",
    };
  }

  const lastRequestedMs = Date.parse(state?.last_requested_at || "");
  if (
    !force &&
    !lastDispatchFailed &&
    Number.isFinite(lastRequestedMs) &&
    minIntervalSeconds > 0 &&
    nowMs - lastRequestedMs < minIntervalSeconds * 1000
  ) {
    const nextAllowedAt = new Date(lastRequestedMs + minIntervalSeconds * 1000).toISOString();
    return {
      configured: true,
      status: "debounced",
      ...payloadSummary(payload),
      min_interval_seconds: minIntervalSeconds,
      next_allowed_at: nextAllowedAt,
      state_warning: stateWarning || undefined,
      note: "Observation saved; retraining was recently queued and will not be started again yet.",
    };
  }

  try {
    await writeRetrainState({
      last_requested_at: payload.requested_at,
      last_observation_fingerprint: payload.observation_fingerprint,
      last_reason: reason,
      last_status: "requested",
      counts: payload.counts,
    });
  } catch (error) {
    stateWarning = stateWarning || error.message || "Could not write retrain state.";
  }

  const dispatch = await dispatchRetrain(payload);
  const status = dispatch.ok ? "queued" : "failed";

  try {
    await writeRetrainState({
      last_requested_at: payload.requested_at,
      last_observation_fingerprint: payload.observation_fingerprint,
      last_reason: reason,
      last_status: status,
      last_target: dispatch.target,
      last_status_code: dispatch.status_code,
      counts: payload.counts,
    });
  } catch (error) {
    stateWarning = stateWarning || error.message || "Could not write retrain state.";
  }

  return {
    configured: true,
    status,
    ...payloadSummary(payload),
    target: dispatch.target,
    status_code: dispatch.status_code,
    status_text: dispatch.status_text,
    min_interval_seconds: minIntervalSeconds,
    state_warning: stateWarning || undefined,
    note: dispatch.ok
      ? "Retraining has been queued on the configured external worker."
      : "Observation saved, but the retraining worker trigger failed.",
  };
}

export function retrainPayload(observations, reason = "field_observation") {
  const features = Array.isArray(observations?.features) ? observations.features : [];
  const counts = countObservationStatuses(features);
  const deploymentUrl = deploymentOrigin();
  const observationsUrl =
    process.env.CULVERT_OBSERVATIONS_URL ||
    (deploymentUrl ? `${deploymentUrl}/api/observations` : undefined);

  return {
    source: "culvert-ai-vercel",
    reason,
    requested_at: new Date().toISOString(),
    observation_count: features.length,
    counts,
    observation_fingerprint: observationFingerprint(features),
    observations_url: observationsUrl,
    deployment_url: deploymentUrl,
    npm_command: "npm run retrain:from-vercel",
    git_commit: process.env.VERCEL_GIT_COMMIT_SHA || undefined,
  };
}

async function dispatchRetrain(payload) {
  if (process.env.CULVERT_RETRAIN_WEBHOOK_URL) {
    return dispatchWebhook(payload);
  }
  return dispatchGitHubRepositoryEvent(payload);
}

async function dispatchWebhook(payload) {
  const headers = { "content-type": "application/json" };
  if (process.env.CULVERT_RETRAIN_WEBHOOK_SECRET) {
    headers["x-culvert-retrain-secret"] = process.env.CULVERT_RETRAIN_WEBHOOK_SECRET;
  }

  try {
    const response = await fetch(process.env.CULVERT_RETRAIN_WEBHOOK_URL, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    return {
      target: "webhook",
      ok: response.ok,
      status_code: response.status,
      status_text: response.statusText,
    };
  } catch (error) {
    return {
      target: "webhook",
      ok: false,
      status_code: 0,
      status_text: error.message || "Webhook request failed.",
    };
  }
}

async function dispatchGitHubRepositoryEvent(payload) {
  const repository = process.env.GITHUB_REPOSITORY;
  const eventType = process.env.GITHUB_RETRAIN_EVENT_TYPE || "culvert-observations-updated";

  try {
    const response = await fetch(`https://api.github.com/repos/${repository}/dispatches`, {
      method: "POST",
      headers: {
        accept: "application/vnd.github+json",
        authorization: `Bearer ${process.env.GITHUB_RETRAIN_TOKEN}`,
        "content-type": "application/json",
        "user-agent": "culvert-ai-vercel",
        "x-github-api-version": "2022-11-28",
      },
      body: JSON.stringify({
        event_type: eventType,
        client_payload: payload,
      }),
    });
    return {
      target: "github_repository_dispatch",
      ok: response.ok || response.status === 204,
      status_code: response.status,
      status_text: response.statusText,
      repository,
      event_type: eventType,
    };
  } catch (error) {
    return {
      target: "github_repository_dispatch",
      ok: false,
      status_code: 0,
      status_text: error.message || "GitHub dispatch failed.",
      repository,
      event_type: eventType,
    };
  }
}

function payloadSummary(payload) {
  return {
    observation_count: payload.observation_count,
    confirmed_culverts: payload.counts.confirmed_culvert || 0,
    denied_culverts: payload.counts.no_culvert || 0,
    uncertain: payload.counts.uncertain || 0,
    observation_fingerprint: payload.observation_fingerprint,
  };
}

function countObservationStatuses(features) {
  const counts = {};
  for (const feature of features) {
    const status = String(feature?.properties?.status || "uncertain");
    counts[status] = (counts[status] || 0) + 1;
  }
  return counts;
}

function observationFingerprint(features) {
  const rows = features
    .map((feature) => {
      const props = feature?.properties || {};
      const coords = feature?.geometry?.coordinates || [];
      return [
        props.observation_id || "",
        props.status || "",
        props.observed_at || "",
        props.field_culvert_id || "",
        props.candidate_id || "",
        coords[0] ?? props.longitude ?? "",
        coords[1] ?? props.latitude ?? "",
      ].join(":");
    })
    .sort();

  return createHash("sha256").update(rows.join("|")).digest("hex").slice(0, 24);
}

function deploymentOrigin() {
  const explicit = process.env.CULVERT_APP_URL || process.env.NEXT_PUBLIC_APP_URL;
  if (explicit) return explicit.replace(/\/+$/, "");

  const vercelUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL || process.env.VERCEL_URL;
  return vercelUrl ? `https://${vercelUrl.replace(/^https?:\/\//, "").replace(/\/+$/, "")}` : "";
}

function retrainMinIntervalSeconds() {
  const value = Number(process.env.CULVERT_RETRAIN_MIN_INTERVAL_SECONDS);
  if (!Number.isFinite(value)) return DEFAULT_MIN_INTERVAL_SECONDS;
  return Math.max(0, Math.floor(value));
}

async function readRetrainState() {
  if (!blobConfigured()) return null;

  try {
    const result = await get(RETRAIN_STATE_BLOB_PATH, { access: "private", useCache: false });
    if (!result?.stream) return null;
    return JSON.parse(await new Response(result.stream).text());
  } catch (error) {
    if (error?.name === "BlobNotFoundError" || /not found/i.test(String(error?.message || ""))) {
      return null;
    }
    throw error;
  }
}

async function writeRetrainState(state) {
  if (!blobConfigured()) return;

  await put(RETRAIN_STATE_BLOB_PATH, `${JSON.stringify(state, null, 2)}\n`, {
    access: "private",
    allowOverwrite: true,
    cacheControlMaxAge: 0,
    contentType: "application/json",
  });
}
