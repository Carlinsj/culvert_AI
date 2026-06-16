import {
  blobConfigured,
  readObservations,
  refreshPublishedData,
  saveObservation,
} from "./_lib/feedback.js";
import { readJsonBody, requireMethod, sendError, sendJson } from "./_lib/http.js";

export default async function handler(request, response) {
  if (!requireMethod(request, response, ["GET", "POST"])) return;

  try {
    if (request.method === "GET") {
      const observations = await readObservations();
      sendJson(response, observations, 200, {
        "x-culvert-feedback-storage": blobConfigured() ? "vercel_blob" : "static",
      });
      return;
    }

    const payload = await readJsonBody(request);
    const saved = await saveObservation(payload);
    const { findings, summary } = await refreshPublishedData(saved.observations);

    sendJson(response, {
      status: "saved",
      feature: saved.feature,
      observations: saved.observations,
      findings,
      summary,
      storage: saved.storage,
      warning: saved.warning,
      training: {
        mode: "deployed_feedback_ranking",
        status: "applied",
        fullRetrain: "Run npm run retrain:from-vercel locally to fold Blob observations into the Python supervised model.",
      },
    }, 201);
  } catch (error) {
    const message = error.message || "Invalid observation.";
    const statusCode = /observation|latitude|longitude|json/i.test(message) ? 400 : 500;
    sendError(response, statusCode, message);
  }
}
