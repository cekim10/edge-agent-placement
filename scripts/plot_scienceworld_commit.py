#!/usr/bin/env python3
"""Figures for the ScienceWorld commit analysis.

Reads `irreversible_actions.csv` as written by
scripts/analyze_scienceworld_irreversible.py, so it can re-render without
touching the episode logs.

fig_sw_commit      commit rate varies with placement; commit *quality* does not.
fig_sw_arithmetic  mean episode score against the fraction of episodes that end
                   in a wrong commit. The points fall on a line whose slope is
                   the -100 penalty: score differences between placements are
                   composition effects of how often the agent commits, not of
                   how well it commits. This is why raw score must not be read
                   as a placement quality metric here.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from tools.paper_plot_style import BRIGHT_PALETTE, configure_plot_style, finish_paper_axes
except ImportError:
    BRIGHT_PALETTE = {
        "black": "#000000",
        "dim_gray": "#696969",
        "dark_gray": "#A9A9A9",
        "blue": "#486EE2",
        "orange": "#FFB226",
        "red": "#D94B4B",
        "white": "#FFFFFF",
    }

    def configure_plot_style() -> None:
        matplotlib.rcParams.update(
            {
                "font.family": "Times New Roman",
                "mathtext.fontset": "cm",
                "axes.linewidth": 2,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def finish_paper_axes(ax: plt.Axes) -> None:
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(2)
        ax.tick_params(axis="both", direction="in", width=2, length=5, colors="black")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        ax.set_axisbelow(True)


ROOT = Path(__file__).resolve().parents[1]

SERIES_1 = BRIGHT_PALETTE["blue"]
SERIES_2 = BRIGHT_PALETTE["orange"]
INK = BRIGHT_PALETTE["black"]
MUTED = BRIGHT_PALETTE["dim_gray"]

FIG_SIZE = (5.83, 4.78)
AXIS_LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 19
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 13

PLACEMENT_ORDER = (
    "all_cloud",
    "edge_state_abstraction",
    "edge_subgoal_planning",
    "edge_action_selection",
    "edge_progress_verification",
    "all_edge",
)
PLACEMENT_LABEL = {
    "all_cloud": "Cloud",
    "edge_state_abstraction": "S@E",
    "edge_subgoal_planning": "G@E",
    "edge_action_selection": "A@E",
    "edge_progress_verification": "V@E",
    "all_edge": "Edge",
}


def wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "irreversible_actions.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run analyze_scienceworld_irreversible.py first")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = {label: index for index, label in enumerate(PLACEMENT_ORDER)}
    rows.sort(key=lambda row: order.get(row["label"], 99))
    return rows


def figure_commit_rates(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [PLACEMENT_LABEL.get(row["label"], row["label"]) for row in rows]
    n = [int(row["n"]) for row in rows]
    commit = [float(row["commit_rate"]) for row in rows]
    wrong = [float(row["wrong_commit_rate"]) for row in rows]
    committed_n = [max(1, round(ni * ci)) for ni, ci in zip(n, commit)]

    def err(rates: list[float], counts: list[int]) -> list[list[float]]:
        # The Wilson centre shrinks toward 0.5, so at rate 1.0 the interval sits
        # entirely below the point estimate; clamp the arms at zero.
        lows, highs = [], []
        for rate, count in zip(rates, counts):
            low, high = wilson(rate, count)
            lows.append(max(0.0, rate - low))
            highs.append(max(0.0, high - rate))
        return [lows, highs]

    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    positions = range(len(labels))
    width = 0.38
    ax.bar(
        [p - width / 2 for p in positions], commit, width,
        label="commit rate", color=SERIES_1, edgecolor=INK, linewidth=1.2,
        yerr=err(commit, n),
        error_kw={"ecolor": MUTED, "elinewidth": 1.4, "capsize": 3},
        zorder=3,
    )
    ax.bar(
        [p + width / 2 for p in positions], wrong, width,
        label="wrong | committed", color=SERIES_2, edgecolor=INK, linewidth=1.2,
        yerr=err(wrong, committed_n),
        error_kw={"ecolor": MUTED, "elinewidth": 1.4, "capsize": 3},
        zorder=3,
    )
    ax.set_ylim(0, 1.12)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_ylabel("rate", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    legend = ax.legend(fontsize=LEGEND_FONT_SIZE, loc="lower left", framealpha=0.9)
    legend.get_frame().set_edgecolor(INK)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.12, top=0.95)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def figure_score_arithmetic(rows: list[dict[str, Any]], out: Path) -> None:
    x = [float(r["commit_rate"]) * float(r["wrong_commit_rate"]) for r in rows]
    y = [float(r["mean_final_score"]) for r in rows]
    labels = [PLACEMENT_LABEL.get(r["label"], r["label"]) for r in rows]

    # Ordinary least squares; the slope should land near the -100 penalty.
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    slope = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot

    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    line_x = [min(x) - 0.04, max(x) + 0.04]
    ax.plot(
        line_x, [intercept + slope * v for v in line_x],
        color=MUTED, linewidth=2, linestyle=(0, (5, 3)), zorder=2,
    )
    ax.scatter(x, y, s=140, color=SERIES_1, edgecolor=INK, linewidth=1.4, zorder=3)
    # Points closer than a tolerance share one annotation; four placements sit
    # in a tight cluster at the top of the commit range and per-point labels
    # would print on top of each other.
    clusters: list[dict[str, Any]] = []
    for xi, yi, label in sorted(zip(x, y, labels)):
        for cluster in clusters:
            if abs(xi - cluster["x"]) < 0.03 and abs(yi - cluster["y"]) < 6:
                cluster["labels"].append(label)
                break
        else:
            clusters.append({"x": xi, "y": yi, "labels": [label]})
    mid_x = (min(x) + max(x)) / 2
    for cluster in clusters:
        # Keep labels inside the axes: a point in the right half is labelled to
        # its left, a point in the left half below-right.
        left_half = cluster["x"] <= mid_x
        ax.annotate(
            ", ".join(cluster["labels"]), (cluster["x"], cluster["y"]),
            textcoords="offset points",
            xytext=(10, 8) if left_half else (-14, 8),
            ha="left" if left_half else "right",
            fontsize=ANNOTATION_FONT_SIZE, color=INK,
        )
    ax.annotate(
        f"slope = {slope:.0f}\n$R^2$ = {r2:.3f}",
        xy=(0.04, 0.08), xycoords="axes fraction",
        fontsize=LEGEND_FONT_SIZE, color=INK,
    )
    ax.set_xlabel("P(wrong commit) per episode", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=6)
    ax.set_ylabel("mean episode score", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=6)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.16, top=0.95)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((ROOT / "outputs").glob("scienceworld_observation1_*"))
        if not candidates:
            print("no scienceworld_observation1_* run directories found")
            return 1
        run_dir = candidates[-1]

    rows = load_rows(run_dir)
    outdir = args.outdir or run_dir
    outdir.mkdir(parents=True, exist_ok=True)
    figure_commit_rates(rows, outdir / "fig_sw_commit")
    figure_score_arithmetic(rows, outdir / "fig_sw_arithmetic")
    print(f"wrote {outdir}/fig_sw_commit.[pdf|png]")
    print(f"wrote {outdir}/fig_sw_arithmetic.[pdf|png]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
