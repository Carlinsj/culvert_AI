"""Render measured scientific figures with Pillow in the bundled image runtime.

No network, basemap tiles, synthetic imagery, or edits to application UI.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "uniform10": "#667085",
    "uniform_exact_budget": "#111827",
    "historical": "#ac6d19",
    "hybrid": "#8656b5",
    "crossing_only": "#117a65",
    "dem_crossings": "#1672a5",
    "mapped_crossings": "#cc526a",
}
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size=19):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def canvas(title, subtitle="", width=1500, height=860):
    im = Image.new("RGB", (width, height), "#ffffff")
    d = ImageDraw.Draw(im)
    d.text((48, 24), title, font=font(29), fill="#152c25")
    d.text((48, 67), subtitle, font=font(18), fill="#52615c")
    d.text(
        (48, height - 35),
        "Retrospective development evaluation · Recommendations require field verification.",
        font=font(16),
        fill="#65716b",
    )
    return im, d


def axes(d, bounds, xlim, ylim, xlabel, ylabel, logx=False, xticks=None):
    left_px, t, r, b = bounds

    def tx(x):
        a, z = (math.log10(xlim[0]), math.log10(xlim[1])) if logx else xlim
        value = math.log10(max(x, 1e-9)) if logx else x
        return left_px + (value - a) / (z - a) * (r - left_px)

    def ty(y):
        return b - (y - ylim[0]) / (ylim[1] - ylim[0]) * (b - t)

    for y in np.linspace(ylim[0], ylim[1], 6):
        py = ty(y)
        d.line((left_px, py, r, py), fill="#e3e8e5", width=1)
        d.text((left_px - 12, py), f"{y:.0f}", font=font(17), fill="#4a554e", anchor="rm")
    ticks = xticks if xticks is not None else np.linspace(xlim[0], xlim[1], 6)
    for x in ticks:
        px = tx(x)
        d.line((px, b, px, b + 6), fill="#4a554e", width=1)
        d.text(
            (px, b + 13),
            f"{x:g}" if abs(x) < 1 else str(int(x)),
            font=font(17),
            fill="#4a554e",
            anchor="mt",
        )
    d.line((left_px, t, left_px, b, r, b), fill="#4a554e", width=2)
    d.text(((left_px + r) / 2, b + 50), xlabel, font=font(20), fill="#24382c", anchor="mt")
    d.text((left_px, t - 30), ylabel, font=font(19), fill="#24382c")
    return tx, ty


def series(d, xs, ys, tx, ty, color, width=3):
    pts = [(tx(float(x)), ty(float(y))) for x, y in zip(xs, ys)]
    if len(pts) > 1:
        d.line(pts, fill=color, width=width)
    for x, y in pts:
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def legend(d, entries, x=1160, y=170):
    for i, (name, color) in enumerate(entries):
        py = y + i * 37
        d.line((x, py + 10, x + 27, py + 10), fill=color, width=4)
        d.text((x + 38, py), name, font=font(17), fill="#344139")


def plot_density(out, stage, name):
    data = pd.read_csv(out / "matched_budget_curves.csv")
    data = data.loc[data.stage.eq(stage)]
    im, d = canvas(
        "Coverage after controlling proposal budget",
        f"{stage.replace('_', ' ').title()} budgets · means of100 fixed seeded subsets · identical834.809km road population",
    )
    tx, ty = axes(
        d,
        (100, 155, 1100, 690),
        (0.1, 120),
        (0, 705),
        "Proposals per road kilometre (log scale)",
        "One-to-one inventory matches /705",
        True,
        [0.1, 0.3, 1, 3, 10, 30, 100],
    )
    for method, group in data.groupby("method"):
        group = group.sort_values("budget")
        series(d, group.density_per_km, group.mean_matches, tx, ty, COLORS[method])
        for row in group.itertuples():
            x = tx(row.density_per_km)
            d.line((x, ty(row.p025), x, ty(row.p975)), fill=COLORS[method], width=2)
    exact = pd.read_csv(out / "uniform_exact_budgets.csv").sort_values("budget")
    series(
        d, exact.density_per_km, exact.matches_one_to_one, tx, ty, COLORS["uniform_exact_budget"], 2
    )
    legend(d, [(m.replace("_", " "), c) for m, c in COLORS.items()])
    d.text(
        (100, 780),
        "Error bars: central95% subset outcomes. Exact-budget uniform is fresh spacing; no label-based density choice.",
        font=font(17),
        fill="#52615c",
    )
    im.save(out / "figures" / name)


def plot_tolerance(out):
    data = pd.read_csv(out / "uniform_sampling.csv")
    data = data.loc[data.scope.eq("eligible705")]
    im, d = canvas(
        "Uniform road sampling: tolerance sensitivity",
        "No ML,streams,DEM or inventory-informed proposal placement",
    )
    tx, ty = axes(
        d,
        (100, 155, 1100, 690),
        (5, 25),
        (0, 705),
        "Matching tolerance (metres)",
        "Nearby inventory records /705",
        False,
        [5, 10, 20, 25],
    )
    palette = ["#117a65", "#1672a5", "#8656b5", "#ac6d19", "#cc526a", "#667085"]
    entries = []
    for color, (spacing, group) in zip(palette, data.groupby("interval_m", sort=True)):
        group = group.sort_values("tolerance_m")
        series(d, group.tolerance_m, group.covered_any, tx, ty, color)
        entries.append((f"{spacing}m spacing", color))
    legend(d, entries)
    d.text(
        (100, 780),
        "Nearby coverage can exceed one-to-one matching capacity. See uniform_sampling.csv for both measures.",
        font=font(17),
        fill="#52615c",
    )
    im.save(out / "figures/02_tolerance.png")


def plot_histogram(out):
    data = pd.read_csv(out / "figure_random_draws.csv")
    stats = pd.read_csv(out / "ranking_random_combined.csv")
    im, d = canvas(
        "Random ranking distribution on identical hybrid sites",
        "Matched@250 in EACH of3road groups (750slots total);1,000 permutations per panel",
        height=820,
    )
    for col, policy in enumerate(["raw_rf", "dynamic_route_z"]):
        arr = data.loc[
            data.method.eq("hybrid")
            & data.policy.eq(policy)
            & data.control.eq("county")
            & data.k.eq(250),
            "matched",
        ].to_numpy()
        obs = int(
            stats.loc[
                stats.method.eq("hybrid")
                & stats.policy.eq(policy)
                & stats.control.eq("county")
                & stats.k.eq(250),
                "observed",
            ].iloc[0]
        )
        lo = min(int(arr.min()), obs) - 2
        hi = max(int(arr.max()), obs) + 2
        counts, bins = np.histogram(arr, bins=np.arange(lo - 0.5, hi + 1.5))
        left_px = 100 + col * 735
        tx, ty = axes(
            d,
            (left_px, 190, left_px + 580, 625),
            (lo, hi),
            (0, max(counts) * 1.15),
            "Inventory matches",
            "Permutation frequency",
        )
        for count, left, right in zip(counts, bins[:-1], bins[1:]):
            if count:
                d.rectangle((tx(left) + 1, ty(count), tx(right) - 1, ty(0)), fill="#9cbdb0")
        d.line((tx(obs), 190, tx(obs), 625), fill="#ba3d36", width=4)
        d.text(
            (left_px, 130),
            f"{policy}: observed{obs}, random mean{arr.mean():.2f}",
            font=font(21),
            fill="#24382c",
        )
    d.text(
        (100, 740),
        "Red line: observed policy. Randomness removes ordering only;site coordinates and matching rule are unchanged.",
        font=font(18),
        fill="#52615c",
    )
    im.save(out / "figures/03_random_distribution.png")


def plot_conditional(out):
    data = pd.read_csv(out / "ranking_random_combined.csv")
    data = data.loc[data.method.eq("hybrid") & data.k.eq(250)]
    im, d = canvas(
        "Does ranking add value within roads?",
        "Hybrid3×Top250 · random null mean and central95% outcome range · fixed representative geometry",
    )
    tx, ty = axes(
        d,
        (100, 170, 1380, 650),
        (-0.6, 5.6),
        (0, max(data.p975.max(), data.observed.max()) + 5),
        "",
        "Inventory matches",
        xticks=[],
    )
    for i, (policy, control) in enumerate(
        (p, c) for p in ["raw_rf", "dynamic_route_z"] for c in ["county", "route", "block2km"]
    ):
        row = data.loc[data.policy.eq(policy) & data.control.eq(control)].iloc[0]
        x = tx(i)
        d.line((x, ty(row.p025), x, ty(row.p975)), fill="#117a65", width=4)
        for v in [row.p025, row.p975]:
            d.line((x - 14, ty(v), x + 14, ty(v)), fill="#117a65", width=3)
        d.ellipse((x - 7, ty(row.random_mean) - 7, x + 7, ty(row.random_mean) + 7), fill="#117a65")
        d.line((x - 28, ty(row.observed), x + 28, ty(row.observed)), fill="#ba3d36", width=4)
        d.text((x, 675), control, font=font(18), fill="#344139", anchor="mt")
        d.text((x, 708), policy, font=font(16), fill="#65716b", anchor="mt")
    d.text(
        (100, 770),
        "Green: random. Red: observed. Route/block shuffles preserve the model's exact stratum at every rank slot.",
        font=font(18),
        fill="#52615c",
    )
    im.save(out / "figures/04_conditional_random.png")


def plot_routes(out):
    data = pd.read_csv(out / "table4_routes.csv", dtype={"route": str})
    routes = sorted(data.route.unique())
    im, d = canvas(
        "Route-by-route ranking contribution",
        "Hybrid3×Top250 · observed minus route-controlled random mean (not a probability)",
        height=1150,
    )
    low = min(-2, float((data.observed - data.random_mean).min()) - 1)
    high = max(2, float((data.observed - data.random_mean).max()) + 1)
    left, right, top, bottom = 170, 1300, 155, 1000

    def tx(v):
        return left + (v - low) / (high - low) * (right - left)

    for value in np.linspace(low, high, 7):
        x = tx(value)
        d.line((x, top, x, bottom), fill="#e4e9e6", width=1)
        d.text((x, bottom + 15), f"{value:.1f}", font=font(18), fill="#52615c", anchor="mt")
    d.line((tx(0), top, tx(0), bottom), fill="#65716b", width=2)
    for i, route in enumerate(routes):
        y = top + 20 + i * 37
        d.text((left - 25, y + 3), route, font=font(19), fill="#24382c", anchor="rm")
        for policy, offset, color in [("raw_rf", -7, "#ac6d19"), ("dynamic_route_z", 7, "#8656b5")]:
            row = data.loc[data.route.eq(route) & data.policy.eq(policy)].iloc[0]
            value = row.observed - row.random_mean
            d.line((tx(0), y + offset, tx(value), y + offset), fill=color, width=7)
    legend(d, [("RawRF excess", "#ac6d19"), ("ROUTE-Z excess", "#8656b5")], x=900, y=100)
    d.text(
        (170, 1070),
        "All22corridors retained, including zero-slot roads. Undefined multiplicative lift is left missing in the CSV.",
        font=font(18),
        fill="#52615c",
    )
    im.save(out / "figures/05_route_lift.png")


def plot_maps(out):
    examples = json.loads((out / "map_examples.json").read_text())
    im, d = canvas(
        "Predetermined road-map examples — no cherry-picking",
        "First3lexical route IDs;first stable segment;midpoint-centred2km square · EPSG32618 metres",
        width=1650,
        height=960,
    )
    palette = {
        "uniform10": "#a7ada9",
        "inventory": "#17251f",
        "crossing": "#1672a5",
        "raw_top250": "#c14038",
    }
    for col, item in enumerate(examples):
        left_px, t, size = 50 + col * 545, 170, 510
        xmin, ymin, xmax, ymax = item["bounds"]

        def loc(point):
            return left_px + (point[0] - xmin) / (xmax - xmin) * size, t + size - (
                point[1] - ymin
            ) / (ymax - ymin) * size

        d.rectangle(
            (left_px, t, left_px + size, t + size), fill="#f2f4ed", outline="#d5dfd6", width=2
        )
        for line in item["roads"]:
            if len(line) > 1:
                d.line([loc(p) for p in line], fill="#cbd1c6", width=4)
        for name, points in item["layers"].items():
            for point in points:
                x, y = loc(point)
                color = palette[name]
                if name == "inventory":
                    d.line((x - 6, y - 6, x + 6, y + 6), fill=color, width=3)
                    d.line((x - 6, y + 6, x + 6, y - 6), fill=color, width=3)
                elif name == "crossing":
                    d.rectangle((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                elif name == "raw_top250":
                    d.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)
                else:
                    d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        d.text((left_px, 120), f"Route{item['route']}", font=font(25), fill="#24382c")
        d.text(
            (left_px, t + size + 16),
            " / ".join(f"{k}:{len(v)}" for k, v in item["layers"].items()),
            font=font(15),
            fill="#52615c",
        )
        bar = 500 / (xmax - xmin) * size
        d.line(
            (left_px + 20, t + size - 25, left_px + 20 + bar, t + size - 25),
            fill="#24382c",
            width=3,
        )
        d.text((left_px + 20, t + size - 52), "500m", font=font(16), fill="#24382c")
        d.text((left_px + size - 35, t + 15), "N↑", font=font(20), fill="#24382c")
    legend(
        d,
        [
            ("Uniform10m (grey)", palette["uniform10"]),
            ("Inventory ×", palette["inventory"]),
            ("Crossing □", palette["crossing"]),
            ("RawRF Top250 ○", palette["raw_top250"]),
        ],
        x=80,
        y=735,
    )
    d.text(
        (550, 770),
        "Symbols show geometry/proximity only. A crossing or nearby road point is not proof of a culvert.",
        font=font(19),
        fill="#52615c",
    )
    im.save(out / "figures/06_map_examples.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    out = parser.parse_args().input
    (out / "figures").mkdir(exist_ok=True)
    plot_density(out, "candidate", "01_candidate_density.png")
    plot_density(out, "physical_site", "01b_site_density.png")
    plot_tolerance(out)
    plot_histogram(out)
    plot_conditional(out)
    plot_routes(out)
    plot_maps(out)
    print("Rendered7measured research figures")


if __name__ == "__main__":
    main()
