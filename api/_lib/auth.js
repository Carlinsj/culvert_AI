import { timingSafeEqual } from "node:crypto";

const FEEDBACK_AUTH_REALM = "culvert-feedback";

export function requireFeedbackWriteAuth(request, response) {
  const configuredToken = safeString(process.env.CULVERT_FEEDBACK_WRITE_TOKEN);
  const requireToken =
    Boolean(configuredToken) ||
    process.env.CULVERT_REQUIRE_FEEDBACK_AUTH === "1" ||
    Boolean(process.env.VERCEL || process.env.VERCEL_ENV);

  if (!requireToken) {
    return true;
  }

  if (!configuredToken) {
    response.setHeader("content-type", "application/json; charset=utf-8");
    response.setHeader("cache-control", "no-store");
    response.statusCode = 500;
    response.end(
      JSON.stringify({
        error: "CULVERT_FEEDBACK_WRITE_TOKEN must be configured before deployed feedback writes are enabled.",
      }),
    );
    return false;
  }

  const suppliedToken =
    bearerToken(request.headers.authorization) || safeString(request.headers["x-culvert-feedback-token"]);
  if (constantTimeTokenEquals(suppliedToken, configuredToken)) {
    return true;
  }

  response.setHeader("www-authenticate", `Bearer realm="${FEEDBACK_AUTH_REALM}"`);
  response.setHeader("content-type", "application/json; charset=utf-8");
  response.setHeader("cache-control", "no-store");
  response.statusCode = 401;
  response.end(JSON.stringify({ error: "Unauthorized feedback write." }));
  return false;
}

export function isAuthorizedBearer(request, expectedToken) {
  const configuredToken = safeString(expectedToken);
  if (!configuredToken) return false;
  return constantTimeTokenEquals(bearerToken(request.headers.authorization), configuredToken);
}

function bearerToken(header) {
  const value = safeString(header);
  const match = value.match(/^Bearer\s+(.+)$/i);
  return match ? safeString(match[1]) : "";
}

function constantTimeTokenEquals(suppliedToken, configuredToken) {
  const supplied = Buffer.from(safeString(suppliedToken));
  const configured = Buffer.from(safeString(configuredToken));
  return supplied.length === configured.length && supplied.length > 0 && timingSafeEqual(supplied, configured);
}

function safeString(value) {
  return String(value || "").trim();
}
