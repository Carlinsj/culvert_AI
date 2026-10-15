import {
  blobConfigured,
  deleteObservation,
  readObservations,
  refreshPublishedData,
  saveObservation,
  saveObservations,
} from "./_lib/feedback.js";
import { requireFeedbackWriteAuth } from "./_lib/auth.js";
import { readJsonBody, requireMethod, sendError, sendJson } from "./_lib/http.js";

export default async function handler(request, response) {
  if (!requireMethod(request, response, ["GET", "POST", "DELETE"])) return;

  try {
    if (request.method === "GET") {
      const observations = await readObservations();
      sendJson(response, observations, 200, {
        "x-culvert-feedback-storage": blobConfigured() ? "vercel_blob" : "static",
      });
      return;
    }

    if (request.method === "DELETE") {
      if (!requireFeedbackWriteAuth(request, response)) return;

      const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
      const deleted = await deleteObservation(url.searchParams.get("id"));
      const { findings, summary } = await refreshPublishedData(deleted.observations);
      sendJson(response, {
        status: "deleted",
        observation_id: deleted.observation_id,
        observations: deleted.observations,
        findings,
        summary,
        storage: deleted.storage,
        model_update: "Field observations are stored separately and do not alter the frozen model.",
        warning: deleted.warning,
      });
      return;
    }

    if (!requireFeedbackWriteAuth(request, response)) return;

    const payload = await readJsonBody(request);
    const bulkPayload = Array.isArray(payload?.features) ? payload.features : Array.isArray(payload) ? payload : null;
    const saved = bulkPayload ? await saveObservations(bulkPayload) : await saveObservation(payload);
    const { findings, summary } = await refreshPublishedData(saved.observations);
    sendJson(response, {
      status: "saved",
      feature: saved.feature || saved.features?.[saved.features.length - 1] || null,
      features: saved.features,
      saved_count: saved.features?.length || 1,
      observations: saved.observations,
      findings,
      summary,
      storage: saved.storage,
      warning: saved.warning,
      model_update: "Field observations are stored separately and do not alter the frozen model.",
    }, 201);
  } catch (error) {
    const message = error.message || "Invalid observation.";
    const statusCode = /observation|latitude|longitude|json/i.test(message) ? 400 : 500;
    sendError(response, statusCode, message);
  }
}
