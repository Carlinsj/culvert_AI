import { readObservations } from "../_lib/feedback.js";
import { requireMethod, sendError, sendJson } from "../_lib/http.js";
import { maybeTriggerRetrain } from "../_lib/retrain.js";

export default async function handler(request, response) {
  if (!requireMethod(request, response, ["GET", "POST"])) return;

  const secret = process.env.CRON_SECRET;
  const production = Boolean(process.env.VERCEL || process.env.VERCEL_ENV);
  if (!secret && production) {
    sendError(response, 500, "CRON_SECRET must be configured for production cron retraining.");
    return;
  }

  if (secret) {
    const authorization = request.headers.authorization || "";
    if (authorization !== `Bearer ${secret}`) {
      sendError(response, 401, "Unauthorized cron request.");
      return;
    }
  }

  try {
    const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
    const observations = await readObservations();
    const retraining = await maybeTriggerRetrain({
      observations,
      reason: "scheduled_retrain_check",
      force: url.searchParams.get("force") === "1",
    });

    sendJson(response, {
      status: "ok",
      retraining,
    });
  } catch (error) {
    sendError(response, 500, error.message || "Could not queue retraining.");
  }
}
