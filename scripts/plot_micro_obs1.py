#!/usr/bin/env python3
"""Figures for Observation 1 of the access-control microbenchmark.

Separate figures, each answering one question a reader will ask:

fig1  Does stage placement matter, and does it matter asymmetrically?
      Per-stage accuracy by placement.

fig2  How often does edge placement create unrequested destructive operations?

fig3  Does the effect hold across the frozen difficulty grid, or only in some
      cells? Reported as a grid rather than a mean so cells that disagree stay
      visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ec_llm_matplotlib")

EC_LLM_ROOT = Path("/Users/cekim/Desktop/EC-LLM")
if EC_LLM_ROOT.exists() and str(EC_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(EC_LLM_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

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
DEEMPHASIS = BRIGHT_PALETTE["dim_gray"]
SURFACE = BRIGHT_PALETTE["white"]
INK_PRIMARY = BRIGHT_PALETTE["black"]
INK_MUTED = BRIGHT_PALETTE["dim_gray"]

SEQUENTIAL_BLUE = [
    BRIGHT_PALETTE["white"],
    "#dbe8ff",
    "#93b7ff",
    BRIGHT_PALETTE["blue"],
    "#163a7a",
]

PLACEMENT_ORDER = ("all_cloud", "edge_classify", "edge_plan", "all_edge")
PLACEMENT_LABEL = {
    "all_cloud": "Cloud",
    "edge_classify": "C@Edge",
    "edge_plan": "P@Edge",
    "all_edge": "Edge",
}
# Placements whose plan stage runs on the edge tier.
PLAN_ON_EDGE = {"edge_plan", "all_edge"}

CELL_ORDER = (("easy", "easy"), ("easy", "hard"), ("hard", "easy"), ("hard", "hard"))

FIG_SIZE = (5.83, 4.78)
AXIS_LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 19
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 13


def wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval; steadier than normal approximation near 0 and 1."""
    if n == 0:
        return (0.0, 0.0)
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
        / (1 + z * z / n)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _err(rates: list[float], counts: list[int]) -> list[list[float]]:
    lows, highs = [], []
    for rate, n in zip(rates, counts):
        low, high = wilson(rate, n)
        lows.append(rate - low)
        highs.append(high - rate)
    return [lows, highs]


def _style_axes(ax: plt.Axes, *, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    finish_paper_axes(ax)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE, pad=8, colors=INK_PRIMARY)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE, pad=5, colors=INK_PRIMARY)
    ax.set_ylabel(ylabel, color=INK_PRIMARY, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    ax.yaxis.set_label_coords(-0.11, 0.45)


def _by_label(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["label"]: entry for entry in metrics}


def _load_metrics_from_cell_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    metrics = []
    for label in PLACEMENT_ORDER:
        if label not in grouped:
            continue
        cells = grouped[label]
        n = sum(int(row["n"]) for row in cells)

        def weighted_mean(key: str) -> float:
            return sum(float(row.get(key) or 0.0) * int(row["n"]) for row in cells) / n

        entry = {
            "label": label,
            "n": n,
            "ok_n": sum(int(row["ok_n"]) for row in cells),
            "error_n": sum(int(row["error_n"]) for row in cells),
            "classification_accuracy": weighted_mean("classification_accuracy"),
            "plan_exact_rate": weighted_mean("plan_exact_rate"),
            "commit_success_rate": weighted_mean("commit_success_rate"),
            "end_to_end_success_rate": weighted_mean("end_to_end_success_rate"),
            "schema_violation_rate": weighted_mean("schema_violation_rate"),
            "plan_partial_recall": weighted_mean("plan_partial_recall"),
            "unrequested_destructive_rate": weighted_mean("unrequested_destructive_rate"),
            "mean_unrequested_destructive_ops": weighted_mean("mean_unrequested_destructive_ops"),
            # Added after this loader was first written; absent in older runs.
            "mean_expected_destructive_ops": weighted_mean("mean_expected_destructive_ops"),
            "mean_effective_destructive_ops": weighted_mean("mean_effective_destructive_ops"),
            "placement": cells[0]["placement"].split(","),
            "by_cell": [
                {
                    "a_level": row["a_level"],
                    "b_level": row["b_level"],
                    "n": int(row["n"]),
                    "ok_n": int(row["ok_n"]),
                    "error_n": int(row["error_n"]),
                    "classification_accuracy": float(row["classification_accuracy"]),
                    "plan_exact_rate": float(row["plan_exact_rate"]),
                    "commit_success_rate": float(row["commit_success_rate"]),
                    "end_to_end_success_rate": float(row["end_to_end_success_rate"]),
                    "schema_violation_rate": float(row["schema_violation_rate"]),
                    "plan_partial_recall": float(row["plan_partial_recall"]),
                    "unrequested_destructive_rate": float(row["unrequested_destructive_rate"]),
                    "mean_unrequested_destructive_ops": float(row["mean_unrequested_destructive_ops"]),
                    "mean_expected_destructive_ops": float(row.get("mean_expected_destructive_ops") or 0.0),
                    "mean_effective_destructive_ops": float(row.get("mean_effective_destructive_ops") or 0.0),
                }
                for row in cells
            ],
        }
        metrics.append(entry)
    return metrics


def figure_stage_rates(metrics: list[dict[str, Any]], out: Path) -> None:
    data = _by_label(metrics)
    labels = [key for key in PLACEMENT_ORDER if key in data]
    counts = [data[key]["n"] for key in labels]
    classify = [data[key]["classification_accuracy"] for key in labels]
    plan = [data[key]["plan_exact_rate"] for key in labels]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor(SURFACE)

    positions = range(len(labels))
    width = 0.36
    gap = 0.02  # surface gap between adjacent bars
    left = [p - width / 2 - gap / 2 for p in positions]
    right = [p + width / 2 + gap / 2 for p in positions]

    ax.bar(
        left, classify, width, label="classification correct",
        color=SERIES_1, edgecolor=INK_PRIMARY, linewidth=1.2, yerr=_err(classify, counts),
        error_kw={"ecolor": INK_PRIMARY, "elinewidth": 1.6, "capsize": 4},
        zorder=3,
    )
    ax.bar(
        right, plan, width, label="plan exact",
        color=SERIES_2, edgecolor=INK_PRIMARY, linewidth=1.2, yerr=_err(plan, counts),
        error_kw={"ecolor": INK_PRIMARY, "elinewidth": 1.6, "capsize": 4},
        zorder=3,
    )
    _style_axes(ax, ylabel="rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([PLACEMENT_LABEL[key] for key in labels])
    ax.set_title(
        "stage sensitivity",
        color=INK_PRIMARY, fontsize=TITLE_FONT_SIZE, loc="left", pad=8,
    )
    legend = ax.legend(
        frameon=False, fontsize=LEGEND_FONT_SIZE, loc="lower center", ncols=2,
        bbox_to_anchor=(0.50, -0.34), handlelength=0.85, columnspacing=0.8,
    )
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.30, top=0.88)
    fig.text(0.98, 0.03, "error bars: 95% Wilson", ha="right", fontsize=ANNOTATION_FONT_SIZE, color=INK_MUTED)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def figure_destructive_ops(metrics: list[dict[str, Any]], out: Path) -> None:
    data = _by_label(metrics)
    labels = [key for key in PLACEMENT_ORDER if key in data]
    destructive = [
        data[key].get("mean_effective_destructive_ops")
        or data[key]["mean_unrequested_destructive_ops"]
        for key in labels
    ]
    # The legitimate irreversible workload. It is a property of the request set,
    # not of the placement, so it is one reference line rather than a series --
    # and without it "0.95 incorrect ops" has no scale.
    ground_truth = max(
        data[key].get("mean_expected_destructive_ops", 0.0) for key in labels
    )
    positions = range(len(labels))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor(SURFACE)

    # Emphasis, not categorical: the claim is about the two bars where the plan
    # stage sits on the edge tier, so the others recede.
    colors = [SERIES_1 if key in PLAN_ON_EDGE else DEEMPHASIS for key in labels]
    ax.bar(
        list(positions),
        destructive,
        0.62,
        color=colors,
        edgecolor=INK_PRIMARY,
        linewidth=1.2,
        zorder=3,
    )
    for pos, value in zip(positions, destructive):
        ax.text(
            pos, value + 0.04, f"{value:.2f}",
            ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE, color=INK_PRIMARY,
        )
    if ground_truth > 0:
        ax.axhline(
            ground_truth, color=INK_PRIMARY, linestyle=(0, (4, 3)),
            linewidth=1.1, zorder=4,
        )
        ax.text(
            -0.42, ground_truth, "correctly requested", va="bottom", ha="left",
            fontsize=ANNOTATION_FONT_SIZE, color=INK_PRIMARY, zorder=4,
        )
    _style_axes(ax, ylabel="irreversible ops per request")
    ceiling = max([*destructive, ground_truth]) if destructive else 1
    ax.set_ylim(0, ceiling * 1.30)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([PLACEMENT_LABEL[key] for key in labels])
    ax.set_title(
        "incorrect irreversible operations",
        color=INK_PRIMARY, fontsize=TITLE_FONT_SIZE, loc="left", pad=8,
    )
    ax.text(
        0.97, 0.90, "blue: plan on edge",
        transform=ax.transAxes, ha="right", fontsize=ANNOTATION_FONT_SIZE, color=INK_MUTED,
    )

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.88)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def figure_difficulty_grid(metrics: list[dict[str, Any]], out: Path) -> None:
    data = _by_label(metrics)
    labels = [key for key in PLACEMENT_ORDER if key in data]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

    matrix: list[list[float]] = []
    for key in labels:
        cells = {(row["a_level"], row["b_level"]): row for row in data[key]["by_cell"]}
        matrix.append(
            [cells[cell]["end_to_end_success_rate"] if cell in cells else float("nan")
             for cell in CELL_ORDER]
        )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor(SURFACE)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            if math.isnan(value):
                continue
            # Ink flips on the dark end of the ramp so every cell stays legible.
            ax.text(
                col, row, f"{value:.2f}", ha="center", va="center", fontsize=ANNOTATION_FONT_SIZE,
                color="#ffffff" if value > 0.62 else INK_PRIMARY,
            )

    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels([f"A:{a}\nB:{b}" for a, b in CELL_ORDER], fontsize=16)
    ax.set_xlabel(
        "A = classify difficulty,  B = plan difficulty",
        color=INK_MUTED, fontsize=16, labelpad=9,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        [PLACEMENT_LABEL[key].replace("\n", " ") for key in labels], fontsize=17
    )
    ax.tick_params(colors=INK_PRIMARY, length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "success across frozen difficulty grid",
        color=INK_PRIMARY, fontsize=TITLE_FONT_SIZE, loc="left", pad=8,
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.outline.set_linewidth(1.2)
    bar.ax.tick_params(colors=INK_PRIMARY, length=4, width=1.2, labelsize=16)

    fig.subplots_adjust(left=0.18, right=0.91, bottom=0.22, top=0.88)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def write_table(metrics: list[dict[str, Any]], out: Path) -> None:
    """Table view, so nothing in the figures is carried by color alone."""
    data = _by_label(metrics)
    lines = [
        "| placement | n | classify | plan exact | e2e | schema | incorrect irrev. ops/req |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in PLACEMENT_ORDER:
        if key not in data:
            continue
        entry = data[key]
        low, high = wilson(entry["end_to_end_success_rate"], entry["n"])
        lines.append(
            f"| {key} | {entry['n']} | {entry['classification_accuracy']:.3f} | "
            f"{entry['plan_exact_rate']:.3f} | "
            f"{entry['end_to_end_success_rate']:.3f} [{low:.2f}, {high:.2f}] | "
            f"{entry['schema_violation_rate']:.3f} | "
            f"{entry.get('mean_effective_destructive_ops') or entry['mean_unrequested_destructive_ops']:.2f} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_plot_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--cell-csv", type=Path)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    if args.cell_csv is not None:
        metrics = _load_metrics_from_cell_csv(args.cell_csv)
        outdir = args.outdir or args.cell_csv.parent
    else:
        run_dir = args.run_dir
        if run_dir is None:
            candidates = sorted((ROOT / "outputs").glob("micro_obs1_*"))
            if not candidates:
                print("no micro_obs1_* run directories found")
                return 1
            run_dir = candidates[-1]

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"missing {metrics_path}")
            return 1
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        outdir = args.outdir or run_dir

    outdir.mkdir(parents=True, exist_ok=True)
    figure_stage_rates(metrics, outdir / "fig1_stage_rates")
    figure_destructive_ops(metrics, outdir / "fig2_destructive_ops")
    figure_difficulty_grid(metrics, outdir / "fig3_difficulty_grid")
    write_table(metrics, outdir / "obs1_table.md")
    print(f"wrote {outdir}/fig1_stage_rates.[pdf|png]")
    print(f"wrote {outdir}/fig2_destructive_ops.[pdf|png]")
    print(f"wrote {outdir}/fig3_difficulty_grid.[pdf|png]")
    print(f"wrote {outdir}/obs1_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
