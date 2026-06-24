import {
  blobConfigured,
  deleteObservation,
  readObservations,
  refreshPublishedData,
  saveObservation,
} from "./_lib/feedback.js";
import { readJsonBody, requireMethod, sendError, sendJson } from "./_lib/http.js";
import { maybeTriggerRetrain } from "./_lib/retrain.js";

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
      const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
      const deleted = await deleteObservation(url.searchParams.get("id"));
      const { findings, summary } = await refreshPublishedData(deleted.observations);
      const retraining = await maybeTriggerRetrain({
        observations: deleted.observations,
        reason: "field_observation_deleted",
      });

      sendJson(response, {
        status: "deleted",
        observation_id: deleted.observation_id,
        observations: deleted.observations,
        findings,
        summary,
        storage: deleted.storage,
        retraining,
        warning: deleted.warning,
      });
      return;
    }

    const payload = await readJsonBody(request);
    const saved = await saveObservation(payload);
    const { findings, summary } = await refreshPublishedData(saved.observations);
    const retraining = await maybeTriggerRetrain({
      observations: saved.observations,
      reason: "field_observation_saved",
    });

    sendJson(response, {
      status: "saved",
      feature: saved.feature,
      observations: saved.observations,
      findings,
      summary,
      storage: saved.storage,
      warning: saved.warning,
      retraining,
      training: {
        mode: "deployed_feedback_ranking",
        status: "applied",
        fullRetrain:
          retraining.status === "queued"
            ? "Automatic retraining was queued on the configured worker."
            : "Run npm run retrain:from-vercel locally, or configure CULVERT_RETRAIN_WEBHOOK_URL/GITHUB_RETRAIN_TOKEN for automatic retraining.",
      },
    }, 201);
  } catch (error) {
    const message = error.message || "Invalid observation.";
    const statusCode = /observation|latitude|longitude|json/i.test(message) ? 400 : 500;
    sendError(response, statusCode, message);
  }
}
