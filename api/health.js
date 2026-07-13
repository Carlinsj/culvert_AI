import { loadPublishedData } from "./_lib/feedback.js";
import { requireMethod, sendError, sendJson } from "./_lib/http.js";
import { loadRouteCountCollection } from "./_lib/route-count.js";

export default async function handler(request, response) {
  if (!requireMethod(request, response, ["GET"])) return;

  try {
    const { findings, summary, storage } = await loadPublishedData();
    const { source } = await loadRouteCountCollection(findings);
    const findingsCount = Array.isArray(findings?.features) ? findings.features.length : 0;
    const summaryRows = Number(summary?.rows ?? 0);

    sendJson(
      response,
      {
        status: "ok",
        activeTask: null,
        python: {
          bin: "vercel",
          ready: true,
        },
        data: {
          findings: {
            exists: findingsCount > 0,
            rows: findingsCount,
          },
          summary: {
            exists: summaryRows > 0,
            rows: summaryRows,
          },
          routeCount: {
            exists: source.rows > 0,
            rows: source.rows,
            complete: Boolean(source.complete),
          },
          observations: {
            exists: false,
          },
        },
      },
      200,
      {
        "x-culvert-feedback-storage": storage,
        "x-culvert-route-count-source": source.name,
      },
    );
  } catch (error) {
    sendError(response, 500, "Could not load health status.", error.message);
  }
}
