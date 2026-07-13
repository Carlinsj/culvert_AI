import { loadPublishedData } from "./_lib/feedback.js";
import { requireMethod, sendError, sendJson } from "./_lib/http.js";
import { buildRouteCountReport, loadRouteCountCollection } from "./_lib/route-count.js";

export default async function handler(request, response) {
  if (!requireMethod(request, response, ["GET"])) return;

  try {
    const url = new URL(request.url || "/", `https://${request.headers.host || "localhost"}`);
    const { findings, storage } = await loadPublishedData();
    const { collection, source } = await loadRouteCountCollection(findings);
    const report = buildRouteCountReport(collection, url.searchParams);
    sendJson(
      response,
      {
        ...report,
        data_source: source,
      },
      200,
      {
        "x-culvert-feedback-storage": storage,
        "x-culvert-route-count-source": source.name,
      },
    );
  } catch (error) {
    sendError(response, 500, "Could not calculate route count.", error.message);
  }
}
