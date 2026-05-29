#!/usr/bin/env python3
"""Domain-composition donut (Panel A) for the FARBench overview.

Two-ring donut from benchmarks/task.md: inner = 5 domains, outer = 29 task types
(lighter shades). Percentages are written outside each domain arc; counts live in
the legend. Outputs a transparent PNG for embedding.
"""
from __future__ import annotations
import colorsys, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
TASK_MD = REPO / "benchmarks" / "task.md"
OUT = REPO / "gui" / "figs" / "domain_composition.png"

DOMAIN_ORDER = ["Computer Vision", "AI for Science", "Robotics", "NLP", "Audio/Speech"]
DOMAIN_COLOR = {
    "Computer Vision": "#6B7A45", "AI for Science": "#5A8FA0", "Robotics": "#C47832",
    "NLP": "#8B6BAE", "Audio/Speech": "#C45068",
}
RING_EDGE = "#FBF7EE"


def parse_tasks():
    rows = []
    for line in TASK_MD.read_text().splitlines():
        if line.startswith("| `"):
            c = [x.strip().strip("`") for x in line.strip().strip("|").split("|")]
            rows.append({"id": c[0], "domain": c[1], "type": c[2]})
    return rows


def lighten(hex_color, amt):
    hc = hex_color.lstrip("#")
    r, g, b = (int(hc[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l + (1 - l) * amt, s * (1 - 0.2 * amt))


def main():
    rows = parse_tasks()
    by_dom = {d: [r for r in rows if r["domain"] == d] for d in DOMAIN_ORDER}
    counts = [len(by_dom[d]) for d in DOMAIN_ORDER]
    total = sum(counts)
    HIGHLIGHT = "Computer Vision"
    EXPLODE = 0.05  # small nudge to highlight CV like a pie-of-pie callout

    fig, ax = plt.subplots(figsize=(9.0, 6.6), subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Outer ring colors + per-wedge explode (CV nudged out)
    outer_sizes, outer_colors, outer_explode = [], [], []
    cv_outer_indices = []
    idx = 0
    for d in DOMAIN_ORDER:
        ts = by_dom[d]
        for j in range(len(ts)):
            outer_sizes.append(1)
            outer_colors.append(lighten(DOMAIN_COLOR[d], 0.30 + 0.42 * (j / max(1, len(ts) - 1))))
            outer_explode.append(EXPLODE if d == HIGHLIGHT else 0.0)
            if d == HIGHLIGHT:
                cv_outer_indices.append(idx)
            idx += 1

    outer_wedges, _ = ax.pie(
        outer_sizes, radius=1.0, colors=outer_colors, startangle=90, counterclock=False,
        explode=outer_explode,
        wedgeprops=dict(width=0.26, edgecolor=RING_EDGE, linewidth=1.0))
    inner_explode = [EXPLODE if d == HIGHLIGHT else 0.0 for d in DOMAIN_ORDER]
    wedges, _ = ax.pie(
        counts, radius=0.73, colors=[DOMAIN_COLOR[d] for d in DOMAIN_ORDER],
        startangle=90, counterclock=False, explode=inner_explode,
        wedgeprops=dict(width=0.40, edgecolor=RING_EDGE, linewidth=2.4))

    # Percentage labels just outside each domain arc
    for w, d, c in zip(wedges, DOMAIN_ORDER, counts):
        ang = math.radians((w.theta1 + w.theta2) / 2)
        bump = EXPLODE if d == HIGHLIGHT else 0.0
        r = 1.16
        x, y = r * math.cos(ang) + bump * math.cos(ang), r * math.sin(ang) + bump * math.sin(ang)
        ax.text(x, y, f"{c / total * 100:.0f}%", ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#5F5144")

    ax.text(0, 0, f"{total}\ntasks", ha="center", va="center", fontsize=18,
            fontweight="bold", color="#5F5144", linespacing=1.0)

    # CV callout — list the 9 task types with leader lines, like the reference
    cv_tasks = by_dom[HIGHLIGHT]
    label_x = 1.95
    n_cv = len(cv_outer_indices)
    label_ys = [1.05 - i * (1.95 / (n_cv - 1)) for i in range(n_cv)]  # 1.05 -> -0.9
    ax.text(label_x - 0.05, 1.22, f"{HIGHLIGHT} subdomains",
            ha="left", va="bottom", fontsize=10.5, fontweight="bold", color="#5F5144")
    for k, (oi, y_l) in enumerate(zip(cv_outer_indices, label_ys)):
        w = outer_wedges[oi]
        ang = math.radians((w.theta1 + w.theta2) / 2)
        # origin at outer edge of the wedge (accounting for explode)
        r0 = 1.02 + EXPLODE
        x0, y0 = r0 * math.cos(ang), r0 * math.sin(ang)
        # leader line: from wedge edge, bend horizontally to label
        ax.annotate(
            "",
            xy=(label_x - 0.06, y_l), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-", color="#9A8C72", lw=0.7,
                            connectionstyle="angle3,angleA=0,angleB=90"),
        )
        # color swatch + task type label
        ax.scatter([label_x - 0.02], [y_l], s=24, color=outer_colors[oi],
                   edgecolor=RING_EDGE, linewidth=0.8, zorder=5)
        ax.text(label_x + 0.06, y_l, cv_tasks[k]["type"], ha="left", va="center",
                fontsize=9.0, color="#3D3225")

    # Domain legend below the donut
    handles = [Patch(facecolor=DOMAIN_COLOR[d], edgecolor=RING_EDGE,
                     label=f"{d}  ({len(by_dom[d])})") for d in DOMAIN_ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.34, -0.02), ncol=3,
              frameon=False, fontsize=10.5, handlelength=1.1, columnspacing=1.6, labelspacing=0.65)

    ax.set_xlim(-1.30, 3.05)
    ax.set_ylim(-1.50, 1.30)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(REPO / "temp_tools" / "_donut_review.png", dpi=140, facecolor="#FBF7EE",
                bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
