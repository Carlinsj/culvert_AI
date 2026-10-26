"""Render measured Rock Ledge comparison from saved native-coordinate results."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "reports/experiments/physical_discovery_20261016"


def main():
    data = json.loads((OUT / "rock_plot_data.json").read_text())
    image = Image.new("RGB", (1800, 1900), "white")
    draw = ImageDraw.Draw(image)

    def font(size):
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)

    draw.text(
        (55, 30),
        "Rock Ledge: raw evidence, site hypotheses and recorded culverts",
        font=font(31),
        fill="#18372b",
    )
    draw.text(
        (55, 80),
        "Identical 1.5 km circle; 20 m one-to-one matches. Native EPSG:32618. No basemap or inventory snapping.",
        font=font(21),
        fill="#4e6156",
    )
    cx, cy = data["center"]
    titles = {
        "baseline_raw": "Baseline computational candidates",
        "baseline_sites": "Baseline full recommended list",
        "hybrid_hypotheses": "Hybrid physical-site hypotheses",
        "hybrid_grouped": "Hybrid identity-grouped recommendations",
    }
    for j, panel in enumerate(data["panels"]):
        left = 65 + (j % 2) * 890
        top = 235 + (j // 2) * 865
        size = 705

        def pixel(x, y):
            return left + (x - cx + 1500) / 3000 * size, top + size - (y - cy + 1500) / 3000 * size

        draw.text((left, top - 90), titles[panel["name"]], font=font(25), fill="#18372b")
        draw.text(
            (left, top - 48),
            f"{panel['count']:,} positions; {panel['matched']}/7 inventory matches",
            font=font(23),
            fill="#4e6156",
        )
        draw.rectangle(
            (left, top, left + size, top + size), fill="#f4f7f4", outline="#c8d0ca", width=2
        )
        for line in data["roads"]:
            draw.line([pixel(*p) for p in line], fill="#c8d0ca", width=2)
        radius = 2 if panel["name"] == "baseline_raw" else 4
        for point in panel["points"]:
            x, y = pixel(*point)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), outline="#8055ae", width=1
            )
        for record in panel["actual"]:
            x, y = pixel(record["x"], record["y"])
            color = "#147555" if record["found"] else "#bc4936"
            for a, b, c, d in [(x - 8, y - 8, x + 8, y + 8), (x - 8, y + 8, x + 8, y - 8)]:
                draw.line((a, b, c, d), fill="white", width=8)
                draw.line((a, b, c, d), fill=color, width=3)
            draw.text(
                (x + 11, y - 16),
                str(record["row"]),
                font=font(17),
                fill="#18372b",
                stroke_width=2,
                stroke_fill="white",
            )
        x, y = pixel(cx - 1300, cy - 1300)
        x2, _ = pixel(cx - 800, cy - 1300)
        draw.line((x, y, x2, y), fill="#18372b", width=4)
        draw.text((x, y - 28), "500 m", font=font(17), fill="#18372b")
    draw.text(
        (65, 1840),
        "Crosses: workbook rows (green matched, red unmatched). Circles: hypotheses, not confirmed culverts.",
        font=font(22),
        fill="#4e6156",
    )
    draw.text(
        (65, 1872),
        "Retrospective, training-exposed research. Recommendations require field verification.",
        font=font(21),
        fill="#4e6156",
    )
    image.save(OUT / "rock_predicted_vs_recorded.png")


if __name__ == "__main__":
    main()
