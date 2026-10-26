"""Culvert AI local research workspace. Startup never trains or deploys."""

import argparse
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the exact existing analysis/static handler. The historical module name
# is an internal implementation detail, not a second user-facing app or route.
from scripts.serve_feature_staging import Handler  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8096, help="Local port (default: 8096)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(
            1,
            f"Could not start the research workspace on port {args.port}: {error}\n"
            "Close the existing server or pass --port with an unused local port.\n",
        )
    print(f"Culvert AI research workspace: http://127.0.0.1:{args.port}/", flush=True)
    print(
        "Research use only. Field deployment target: summer 2027, subject to validation.\n"
        "Recommendations require field verification. Stop with Ctrl+C.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
