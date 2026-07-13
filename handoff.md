# Handoff

Generated: 2026-07-08 20:44 EDT

## Current State

A security audit was completed for the Culvert AI repo. The confirmed issue was that deployed observation mutation endpoints were unauthenticated, which allowed anyone to write/delete field observations and indirectly trigger retraining automation. That has been fixed.

The local dev server was started during verification and was healthy at:

```text
http://127.0.0.1:8080
```

## Security Fixes Applied

- Added shared Bearer-token auth in `api/_lib/auth.js`.
- Protected deployed `POST /api/observations` and `DELETE /api/observations?id=...` in `api/observations.js`.
- Updated `web/app.js` so field users are prompted for a field update token only when a write request receives `401`, then the token is stored in `sessionStorage` for that browser session.
- Hardened `server/dev-server.js`:
  - Observation writes use the same auth helper when configured or deployed.
  - Remote `/api/run-*` task endpoints now require `CULVERT_DEV_TASK_TOKEN`; loopback clients are still allowed for local development.
  - Static file path resolution now uses `path.resolve` plus `path.relative` checks.
- Added deployment security headers in `vercel.json`:
  - `Content-Security-Policy`
  - `Strict-Transport-Security`
  - `Permissions-Policy`
  - Existing `X-Content-Type-Options` and `Referrer-Policy` remain.
- Added SRI and `crossorigin="anonymous"` for Leaflet CDN CSS/JS in `web/index.html`.
- Added `scripts/verify_security.js` and wired it into `npm test`.
- Documented new env vars in `.env.example`, `README.md`, and `track.md`.

## Required Production Env Vars

Set this before deploying, otherwise deployed feedback writes fail closed:

```text
CULVERT_FEEDBACK_WRITE_TOKEN=<strong shared field-write token>
```

Optional for remote access to local/dev task endpoints:

```text
CULVERT_DEV_TASK_TOKEN=<strong dev task token>
```

Existing persistence/retraining env vars still apply:

```text
BLOB_READ_WRITE_TOKEN
VERCEL_OIDC_TOKEN
BLOB_STORE_ID
CULVERT_RETRAIN_WEBHOOK_URL
CULVERT_RETRAIN_WEBHOOK_SECRET
GITHUB_RETRAIN_TOKEN
GITHUB_REPOSITORY
CRON_SECRET
```

## Verification Completed

Commands run successfully:

```bash
npm run build
npm test
npm audit --omit=dev
scripts/python.sh -m pip_audit -r requirements.txt
```

Results:

- `npm run build`: passed; static Vercel assets and API imports verified.
- `npm test`: passed; 52 Python tests plus the new security auth check.
- `npm audit --omit=dev`: 0 vulnerabilities.
- `pip-audit`: no known Python dependency vulnerabilities.
- Browser verification against `http://127.0.0.1:8080/`: HTTP 200, meaningful content rendered, map pane and candidate list present, no console/page errors.

## Changed Files

Tracked modifications:

```text
.env.example
README.md
api/observations.js
package.json
server/dev-server.js
track.md
vercel.json
web/app.js
web/index.html
```

New files:

```text
api/_lib/auth.js
scripts/verify_security.js
handoff.md
```

## Notes For Next Operator

- Do not deploy without setting `CULVERT_FEEDBACK_WRITE_TOKEN`.
- The token is intentionally a shared field-write secret, not full user auth.
- Existing read-only endpoints remain public.
- The browser stores the write token in session storage, so field users will re-enter it each new browser session.
- The local `.env` file was not inspected or modified.
