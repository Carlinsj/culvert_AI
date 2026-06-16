export function sendJson(response, payload, statusCode = 200, headers = {}) {
  const body = JSON.stringify(payload, null, 2);
  response.statusCode = statusCode;
  response.setHeader("content-type", "application/json; charset=utf-8");
  response.setHeader("cache-control", "no-store");
  for (const [key, value] of Object.entries(headers)) {
    response.setHeader(key, value);
  }
  response.end(body);
}

export function sendError(response, statusCode, message, details = undefined) {
  sendJson(response, { error: message, details }, statusCode);
}

export async function readJsonBody(request, maxBytes = 1_000_000) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > maxBytes) {
      throw new Error("Request body is too large.");
    }
  }

  return body ? JSON.parse(body) : {};
}

export function requireMethod(request, response, methods) {
  if (methods.includes(request.method)) {
    return true;
  }

  response.setHeader("allow", methods.join(", "));
  sendError(response, 405, "Method not allowed.");
  return false;
}
