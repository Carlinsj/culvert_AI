"""Static measured comparison figures; existing bundled Pillow, no new data."""

import argparse
import json
from pathlib import Path

import pandas as pd
from plot_density_controls import canvas, axes, font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    out = parser.parse_args().input
    data = json.loads((out / "spatial_plot_data.json").read_text())
    im, d = canvas(
        "Spatial distribution before and after grouping",
        "Same road geometry and extents; all points drawn. Native EPSG:32618, not a basemap.",
        1600,
        1200,
    )
    x0, y0, x1, y1 = data["bounds"]
    colors = ["#929b96", "#8055ae", "#bf7f2b", "#2463a8"]
    for i, (name, points) in enumerate(data["frames"].items()):
        left = 50 + (i % 2) * 780
        top = 135 + (i // 2) * 495
        width, height = 700, 400
        scale = min(width / (x1 - x0), height / (y1 - y0))
        offset_x = left + (width - (x1 - x0) * scale) / 2

        def pixel(p):
            return (offset_x + (p[0] - x0) * scale, top + height - (p[1] - y0) * scale)

        d.text((left, top - 35), f"{name}: {len(points):,}", font=font(22), fill=colors[i])
        for road in data["roads"]:
            d.line([pixel(p) for p in road], fill="#dde3df", width=1)
        for p in points:
            x, y = pixel(p)
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=colors[i])
        d.rectangle((left - 5, top - 5, left + width + 5, top + height + 5), outline="#e2e7e3")
    im.save(out / "spatial-distribution.png")
    groups = pd.read_csv(out / "grouping_sensitivity.csv")
    groups = groups.loc[groups.tolerance_m.eq(20)]
    im, d = canvas(
        "Grouping reduces count; stronger suppression also loses coverage",
        "Feature proposals only; one-to-one official-inventory matches within 20 m. Not field accuracy.",
    )
    tx, ty = axes(
        d, (110, 140, 1110, 710), (4000, 6600), (240, 320), "Proposed groups", "Inventory matches"
    )
    for name, color in [
        ("legacy_road_nms", "#bf7f2b"),
        ("evidence_identity", "#2463a8"),
        ("geodesic_nms", "#8055ae"),
        ("geodesic_dbscan_min1", "#8e9690"),
    ]:
        part = groups.loc[groups.method.eq(name)].sort_values("grouping_distance_m")
        pts = [(tx(r.candidates), ty(r.matches_one_to_one)) for r in part.itertuples()]
        d.line(pts, fill=color, width=3)
        for row, (x, y) in zip(part.itertuples(), pts):
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            if name == "legacy_road_nms" and row.grouping_distance_m in (5, 20, 40):
                d.text(
                    (x + 7, y - 18), str(row.grouping_distance_m) + " m", font=font(14), fill=color
                )
            if name == "evidence_identity" and row.grouping_distance_m == 20:
                d.text((x - 175, y - 27), "10–40 m: 301 matches", font=font(14), fill=color)
        index = [
            "legacy_road_nms",
            "evidence_identity",
            "geodesic_nms",
            "geodesic_dbscan_min1",
        ].index(name)
        d.text((1140, 190 + index * 60), name.replace("_", " "), font=font(16), fill=color)
    im.save(out / "grouping-tradeoff.png")
    gen = pd.read_csv(out / "candidate_generation.csv")
    im, d = canvas(
        "Sampling density is not culvert detection",
        "Controlled same-road half-offset samples. Densities use 834.809416 segment-km.",
    )
    tx, ty = axes(
        d,
        (110, 140, 1110, 710),
        (0, 110),
        (0, 705),
        "Candidates per road kilometre",
        "One-to-one inventory matches",
    )
    for stage, color in [("candidates", "#4d6278"), ("legacy20_sites", "#bf7f2b")]:
        part = gen.loc[
            gen.method.str.startswith("sampling_") & gen.stage.eq(stage) & gen.tolerance_m.eq(20)
        ].sort_values("density_per_km")
        pts = [(tx(r.density_per_km), ty(r.matches_one_to_one)) for r in part.itertuples()]
        d.line(pts, fill=color, width=3)
        for row, (x, y) in zip(part.itertuples(), pts):
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            d.text(
                (x + 8, y - 22 if stage == "candidates" else y + 7),
                row.method.replace("sampling_", ""),
                font=font(17),
                fill=color,
            )
        d.text(
            (1140, 190 + (stage != "candidates") * 60),
            stage.replace("_", " "),
            font=font(18),
            fill=color,
        )
    im.save(out / "sampling-density.png")
    print("Saved 3 measured figures.")


if __name__ == "__main__":
    main()
