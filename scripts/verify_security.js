import assert from "node:assert/strict";

import { isAuthorizedBearer, requireFeedbackWriteAuth } from "../api/_lib/auth.js";

const previousEnv = {
  CULVERT_FEEDBACK_WRITE_TOKEN: process.env.CULVERT_FEEDBACK_WRITE_TOKEN,
  CULVERT_REQUIRE_FEEDBACK_AUTH: process.env.CULVERT_REQUIRE_FEEDBACK_AUTH,
  VERCEL: process.env.VERCEL,
  VERCEL_ENV: process.env.VERCEL_ENV,
};

try {
  setAuthEnv({});
  assert.equal(requireFeedbackWriteAuth(fakeRequest(), fakeResponse()), true);

  setAuthEnv({ VERCEL: "1" });
  const missingTokenResponse = fakeResponse();
  assert.equal(requireFeedbackWriteAuth(fakeRequest(), missingTokenResponse), false);
  assert.equal(missingTokenResponse.statusCode, 500);

  setAuthEnv({ CULVERT_FEEDBACK_WRITE_TOKEN: "test-token" });
  const unauthorizedResponse = fakeResponse();
  assert.equal(requireFeedbackWriteAuth(fakeRequest(), unauthorizedResponse), false);
  assert.equal(unauthorizedResponse.statusCode, 401);
  assert.match(unauthorizedResponse.headers["www-authenticate"], /culvert-feedback/);

  assert.equal(
    requireFeedbackWriteAuth(
      fakeRequest({ authorization: "Bearer test-token" }),
      fakeResponse(),
    ),
    true,
  );
  assert.equal(isAuthorizedBearer(fakeRequest({ authorization: "Bearer test-token" }), "test-token"), true);
  assert.equal(isAuthorizedBearer(fakeRequest({ authorization: "Bearer wrong-token" }), "test-token"), false);

  console.log("Verified feedback write authentication checks.");
} finally {
  restoreEnv();
}

function setAuthEnv(values) {
  delete process.env.CULVERT_FEEDBACK_WRITE_TOKEN;
  delete process.env.CULVERT_REQUIRE_FEEDBACK_AUTH;
  delete process.env.VERCEL;
  delete process.env.VERCEL_ENV;

  for (const [key, value] of Object.entries(values)) {
    process.env[key] = value;
  }
}

function restoreEnv() {
  for (const key of Object.keys(previousEnv)) {
    if (previousEnv[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = previousEnv[key];
    }
  }
}

function fakeRequest(headers = {}) {
  return {
    headers,
  };
}

function fakeResponse() {
  return {
    headers: {},
    statusCode: 200,
    body: "",
    setHeader(key, value) {
      this.headers[key.toLowerCase()] = value;
    },
    end(body = "") {
      this.body = body;
    },
  };
}
