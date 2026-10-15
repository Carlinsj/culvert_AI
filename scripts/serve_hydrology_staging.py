"""Read-only experimental map on loopback. Does not start production or train."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging.hydrology_service import analyze  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, payload, content_type="application/json"):
        body = (
            json.dumps(payload, allow_nan=False).encode()
            if content_type == "application/json"
            else payload
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        name = {
            "/": "index.html",
            "/app.js": "app.js",
            "/styles.css": "styles.css",
            "/core.js": "core.js",
        }.get(urlparse(self.path).path)
        if not name:
            return self.respond(404, {"error": "Not found"})
        mime = (
            "text/html"
            if name.endswith("html")
            else "text/css"
            if name.endswith("css")
            else "application/javascript"
        )
        return self.respond(200, (ROOT / "staging/web" / name).read_bytes(), mime)

    def do_POST(self):
        if self.path != "/api/analysis/nearby":
            return self.respond(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 4096:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            location = payload.get("location", payload)
            result = analyze(
                location["latitude"],
                location["longitude"],
                payload.get("radius_m", 1500),
                payload.get("model", "RF"),
            )
            self.respond(200, result)
        except FileNotFoundError:
            self.respond(
                503,
                {
                    "error": "Experimental model is not ready; production models are not substituted."
                },
            )
        except (ValueError, KeyError, TypeError) as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            self.log_error("Analysis failed: %r", error)
            self.respond(
                500,
                {
                    "error": "Analysis failed. See local server diagnostics; no production changes were made."
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    print(f"Experimental, read-only map: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
