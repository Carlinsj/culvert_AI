"""Isolated high-recall hybrid research map. Never deploys or fits a model."""

import argparse
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.serve_hydrology_staging import Handler as BaseHandler  # noqa: E402
from staging.hybrid_service import analyze  # noqa: E402


class Handler(BaseHandler):
    def do_GET(self):
        files = {
            "/": ("staging/hybrid_web/index.html", "text/html"),
            "/app.js": ("staging/hybrid_web/app.js", "application/javascript"),
            "/styles.css": ("staging/web/styles.css", "text/css"),
            "/core.js": ("staging/web/core.js", "application/javascript"),
        }
        target = files.get(urlparse(self.path).path)
        if target is None:
            return self.respond(404, {"error": "Not found"})
        return self.respond(200, (ROOT / target[0]).read_bytes(), target[1])

    def do_POST(self):
        if self.path != "/api/analysis/nearby":
            return self.respond(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            location = payload.get("location", payload)
            result = analyze(
                location["latitude"],
                location["longitude"],
                payload.get("radius_m", 1500),
                payload.get("policy", "dynamic_route_z"),
            )
            self.respond(200, result)
        except FileNotFoundError:
            self.respond(
                503,
                {
                    "error": "Required v2 experimental model or GIS assets are unavailable. No production model is substituted."
                },
            )
        except (ValueError, KeyError, TypeError) as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            self.log_error("Hybrid analysis failed: %r", error)
            self.respond(
                500, {"error": "Hybrid research analysis failed; no production changes were made."}
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    print(f"High-recall hybrid staging only: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
