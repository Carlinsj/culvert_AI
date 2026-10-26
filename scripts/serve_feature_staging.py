"""Separate Leaflet localization/ruler staging. Stop this server to roll back."""

import argparse
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.serve_hydrology_staging import Handler as BaseHandler  # noqa:E402
from staging.feature_service import analyze  # noqa:E402


def benchmark_payload():
    path = ROOT / "reports/experiments/anti_saturation_validation_20261017/ui_benchmark.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    payload["full_population"] = [
        {"id": "existing_baseline_rf", "label": "Original RF", "sites": 13065, "matches": 347, "recall": 347 / 847},
        {"id": "v3_physical_feature_model", "label": "v3 RF", "sites": 68302, "matches": 672, "recall": 672 / 847},
    ]
    v4_path = ROOT / "reports/experiments/physical_ranker_v4_20261019/retrieval_metrics.csv"
    if v4_path.exists():
        import pandas as pd

        metrics = pd.read_csv(v4_path)
        challenger = metrics.loc[metrics.model.eq("v4_physical_ranker")]
        for budget, row in challenger.loc[challenger.budget.ne("full_population")].set_index("budget").iterrows():
            key = str(int(float(budget)))
            if key in payload["budgets"]:
                payload["budgets"][key]["v4_physical_ranker"] = {
                    "matches": int(row.matched),
                    "recall": float(row.recall),
                    "matches_per_1000": float(row.matches_per_1000),
                    "median_distance_m": float(row.median_distance_m),
                }
        full = challenger.loc[challenger.budget.eq("full_population")].iloc[0]
        payload["full_population"].append(
            {"id": "v4_physical_ranker", "label": "v4 challenger", "sites": int(full.stops), "matches": int(full.matched), "recall": float(full.recall)}
        )
    v5_path = ROOT / "reports/experiments/baseline_preserving_v5_20261019/retrieval_metrics.csv"
    if v5_path.exists():
        import pandas as pd

        metrics = pd.read_csv(v5_path)
        challenger = metrics.loc[metrics.model.eq("original_population_v5_reranker")]
        for budget, row in challenger.loc[challenger.budget.ne("full_population")].set_index("budget").iterrows():
            key = str(int(float(budget)))
            if key in payload["budgets"]:
                payload["budgets"][key]["v5_baseline_preserving"] = {
                    "matches": int(row.matched),
                    "recall": float(row.recall),
                    "matches_per_1000": float(row.matches_per_1000),
                    "median_distance_m": float(row.median_distance_m),
                }
        full = challenger.loc[challenger.budget.eq("full_population")].iloc[0]
        payload["full_population"].append(
            {"id": "v5_baseline_preserving", "label": "v5 baseline-preserving", "sites": int(full.stops), "matches": int(full.matched), "recall": float(full.recall)}
        )
    v6_path = ROOT / "reports/experiments/final_physical_optimization_20261019/fixed_budget_metrics.csv"
    if v6_path.exists():
        import pandas as pd

        metrics = pd.read_csv(v6_path)
        challenger = metrics.loc[metrics.system.eq("v6_compact_physical")]
        for budget, row in challenger.loc[challenger.budget_label.ne("full")].set_index("budget").iterrows():
            key = str(int(float(budget)))
            if key in payload["budgets"]:
                payload["budgets"][key]["v6_compact_physical"] = {
                    "matches": int(row.matched),
                    "recall": float(row.recall),
                    "matches_per_1000": float(row.matches_per_1000),
                    "median_distance_m": float(row.median_localization_error_m),
                }
        full = challenger.loc[challenger.budget_label.eq("full")].iloc[0]
        payload["full_population"].append(
            {"id": "v6_compact_physical", "label": "v6 compact physical", "sites": int(full.predicted_sites), "matches": int(full.matched), "recall": float(full.recall)}
        )
    return payload


class Handler(BaseHandler):
    def respond(self, status, payload, content_type="application/json"):
        # Navigation/reloads can disconnect while a read-only analysis finishes.
        # Do not turn an expected closed socket into a second 500 write.
        try:
            return super().respond(status, payload, content_type)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):
        if urlparse(self.path).path == "/api/research/comparison/options":
            from staging.comparison_service import options

            return self.respond(200, options())
        if urlparse(self.path).path == "/api/research/anti-saturation":
            try:
                payload = benchmark_payload()
            except FileNotFoundError:
                return self.respond(503, {"error": "Anti-saturation benchmark is unavailable."})
            return self.respond(200, payload)
        if urlparse(self.path).path == "/api/research/inventory":
            # Explicit diagnostic overlay only. Neither inference path reads it.
            import geopandas as gpd
            from staging.hydrology_service import geojson

            path = ROOT / "reports/experiments/summer_inventory_20261016/inventory_coverage.parquet"
            if not path.exists():
                return self.respond(503, {"error": "Summer inventory diagnostic is unavailable."})
            frame = gpd.read_parquet(path)
            audit = ROOT / "reports/experiments/physical_discovery_20261016/all_959_record_diagnosis.csv"
            if not audit.exists():
                return self.respond(503, {"error": "Summer inventory coverage audit is unavailable."})
            import pandas as pd

            coverage = pd.read_csv(audit)[["record_id", "query_success"]]
            frame = frame.merge(coverage, on="record_id", validate="one_to_one")
            frame["x_m"] = frame.geometry.x
            frame["y_m"] = frame.geometry.y
            return self.respond(
                200,
                geojson(
                    frame,
                    [
                        "record_id",
                        "source_excel_row",
                        "Latitude",
                        "Longitude",
                        "Region",
                        "Route",
                        "Team",
                        "Culvert ID",
                        "supported",
                        "query_success",
                        "x_m",
                        "y_m",
                    ],
                ),
            )
        paths = {
            "/": "index.html",
            "/app.js": "app.js",
            "/core.js": "core.js",
            "/comparison.js": "comparison.js",
            "/comparison.css": "comparison.css",
            "/ruler.js": "ruler.js",
            "/styles.css": "styles.css",
        }
        name = paths.get(urlparse(self.path).path)
        if not name:
            return self.respond(404, {"error": "Not found"})
        mime = (
            "text/html"
            if name.endswith("html")
            else "text/css"
            if name.endswith("css")
            else "application/javascript"
        )
        self.respond(200, (ROOT / "staging/feature_web" / name).read_bytes(), mime)

    def do_POST(self):
        if self.path == "/api/research/comparison":
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                from staging.comparison_service import compare

                return self.respond(200, compare(payload))
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                return self.respond(400, {"error": str(error)})
            except Exception as error:
                self.log_error("Comparison failed:%r", error)
                return self.respond(500, {"error": "Comparison failed; production remains unchanged."})
        if self.path != "/api/analysis/nearby":
            return self.respond(404, {"error": "Not found"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected an object")
            location = data.get("location", data)
            if data.get("mode") == "physical_discovery":
                from staging.discovery_service import analyze as physical_analysis

                result = physical_analysis(
                    location["latitude"],
                    location["longitude"],
                    data.get("radius_m", 1500),
                    data.get("policy", "dynamic_route_z"),
                )
            else:
                result = analyze(
                    location["latitude"],
                    location["longitude"],
                    data.get("radius_m", 1500),
                    data.get("policy", "dynamic_route_z"),
                    data.get("mode", "feature"),
                    data.get("grouping", "legacy20"),
                )
            self.respond(200, result)
        except FileNotFoundError:
            self.respond(
                503,
                {"error": "Required experimental v2 assets unavailable;no frozen model fallback."},
            )
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            self.log_error("Feature analysis failed:%r", error)
            self.respond(
                500, {"error": "Experimental analysis failed;production remains unchanged."}
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()
    print(f"Feature localization staging only:http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
