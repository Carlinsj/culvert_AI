"""Measured research figures; run with bundled Python/Pillow, no map fabrication."""

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"

    def font(size):
        return ImageFont.truetype(font_path, size)

    colors = [
        "#174c3b",
        "#b65a35",
        "#325fa7",
        "#8a4794",
        "#79713d",
        "#269094",
        "#735e53",
        "#d27c27",
    ]
    with (out / "source_candidates_HYBRID-GROUPED.csv").open() as file:
        rows = list(csv.DictReader(file))
    sources = sorted({r["research_source"] for r in rows})
    image = Image.new("RGB", (1700, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 30), "What entered the hybrid queue?", font=font(36), fill="#18372b")
    draw.text(
        (60, 85),
        "Measured source distributions; inventory distance is diagnostic, not a negative label.",
        font=font(22),
        fill="#52605a",
    )
    for panel, (field, title) in enumerate(
        [
            ("score_RF", "Uncalibrated RF score"),
            ("nearest_inventory_m", "Distance to nearest summer record (m, log scale)"),
        ]
    ):
        x0, y0, w, h = 90 + panel * 825, 180, 700, 565
        draw.text((x0, y0 - 50), title, font=font(22), fill="#18372b")
        for q in [0, 0.25, 0.5, 0.75, 1]:
            y = y0 + h * (1 - q)
            draw.line((x0, y, x0 + w, y), fill="#dce3df", width=1)
            draw.text((x0 - 45, y - 12), f"{q:g}", font=font(17), fill="#52605a")
        for i, label in enumerate(sources):
            values = sorted(float(r[field]) for r in rows if r["research_source"] == label)
            if not values:
                continue
            points = []
            for j in range(0, len(values), max(1, len(values) // 500)):
                v = values[j]
                x = v if panel == 0 else min(4, math.log10(max(1, v))) / 4
                points.append((x0 + w * max(0, min(1, x)), y0 + h * (1 - (j + 1) / len(values))))
            if len(points) > 1:
                draw.line(points, fill=colors[i % len(colors)], width=3)
        ticks = (
            [(0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1, "1")]
            if panel == 0
            else [(0, "1"), (0.25, "10"), (0.5, "100"), (0.75, "1,000"), (1, "10,000")]
        )
        for x, label in ticks:
            draw.text((x0 + w * x - 15, y0 + h + 15), label, font=font(18), fill="#52605a")
        draw.text((x0, y0 - 25), "Cumulative fraction", font=font(17), fill="#52605a")
        draw.text(
            (x0, y0 + h + 48),
            "RF ranking score" if panel == 0 else "Nearest record distance (m)",
            font=font(20),
            fill="#52605a",
        )
    for i, label in enumerate(sources):
        x, y = 75 + (i % 4) * 410, 845 + (i // 4) * 40
        draw.line((x, y + 10, x + 30, y + 10), fill=colors[i % len(colors)], width=4)
        draw.text((x + 40, y), label.replace("_", " "), font=font(19), fill="#18372b")
    draw.text(
        (60, 960),
        "Prior-exposed retrospective research. Recommendations require field verification.",
        font=font(21),
        fill="#52605a",
    )
    image.save(out / "source_distributions.png")

    with (out / "comparison.csv").open() as file:
        metrics = list(csv.DictReader(file))
    selected = [
        ("BASELINE-RF", "Baseline"),
        ("FULL-HYBRID-RF", "Full hybrid"),
        ("SELECTIVE-GROUPED-corroborated-RF", "Selective grouped"),
        ("PHYSICAL-RERANK", "Selected soft diversity"),
        ("NESTED-SELECTIVE", "Nested development"),
        ("SITE-MODEL-CROSSFIT", "New site model, cross-fit"),
    ]
    image = Image.new("RGB", (1550, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (65, 30),
        "Distinct recorded culverts recovered at equal inspection budgets",
        font=font(30),
        fill="#18372b",
    )
    draw.text(
        (65, 80),
        "20 m one-to-one matching | 847 supported records | not verified precision",
        font=font(22),
        fill="#52605a",
    )
    x0, y0, w, h = 100, 160, 1070, 620
    for value in range(0, 141, 20):
        y = y0 + h * (1 - value / 140)
        draw.line((x0, y, x0 + w, y), fill="#dce3df", width=1)
        draw.text((x0 - 55, y - 12), str(value), font=font(20), fill="#52605a")
    for i, (system, label) in enumerate(selected):
        points = []
        for k in (100, 250, 500, 1000):
            row = next(r for r in metrics if r["system"] == system and r["budget"] == str(k))
            x = x0 + w * (k - 100) / 900
            y = y0 + h * (1 - int(row["matched"]) / 140)
            points.append((x, y))
        draw.line(points, fill=colors[i], width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[i])
        draw.text((1200, 210 + i * 72), label, font=font(19), fill=colors[i])
        draw.text(
            (1200, 238 + i * 72),
            f"@1000: {next(r['matched'] for r in metrics if r['system'] == system and r['budget'] == '1000')}",
            font=font(20),
            fill=colors[i],
        )
    for k in (100, 250, 500, 1000):
        draw.text(
            (x0 + w * (k - 100) / 900 - 22, y0 + h + 15), str(k), font=font(20), fill="#52605a"
        )
    draw.text((430, 840), "Distinct recommended stops", font=font(24), fill="#18372b")
    nested_count = next(
        r["matched"] for r in metrics if r["system"] == "NESTED-SELECTIVE" and r["budget"] == "1000"
    )
    draw.text(
        (65, 910),
        f"127 vs 123 is an exposed-development result; nested selection recovers {nested_count}, so keep the default.",
        font=font(23),
        fill="#18372b",
    )
    draw.text(
        (65, 955),
        "No deployment. Recommendations require field verification.",
        font=font(22),
        fill="#52605a",
    )
    image.save(out / "fixed_budget_retrieval.png")


if __name__ == "__main__":
    main()
