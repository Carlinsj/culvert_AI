import { loadPublishedData } from "./_lib/feedback.js";
import { requireMethod, sendError, sendJson } from "./_lib/http.js";

export default async function handler(request, response) {
  if (!requireMethod(request, response, ["GET"])) return;

  try {
    const { summary, storage } = await loadPublishedData();
    sendJson(response, summary, 200, {
      "x-culvert-feedback-storage": storage,
    });
  } catch (error) {
    sendError(response, 500, "Could not load summary.", error.message);
  }
}
