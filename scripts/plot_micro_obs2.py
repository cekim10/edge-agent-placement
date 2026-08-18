#!/usr/bin/env python3
"""Figures for Observation 2 (verification as a decision variable).

Reads metrics.csv from a micro_obs2 run directory.

fig_obs2_tradeoff  detection, false rejects, and unsafe commits per verifier.
                   The edge verifier's headline recall is bought by rejecting
                   most correct plans; the failure mode of a weak verifier is
                   blocked work, not approved damage.
fig_obs2_latency   projected pipeline latency against RTT. Only the cloud
                   verifier pays a network crossing under the all-edge
                   placement, so its line is the only one with slope 1.
"""

from __future__ import annotations

import argparse
import csv
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

BLUE = BRIGHT_PALETTE["blue"]
ORANGE = BRIGHT_PALETTE["orange"]
RED = BRIGHT_PALETTE["red"]
GRAY = BRIGHT_PALETTE["dim_gray"]
INK = BRIGHT_PALETTE["black"]

FIG_SIZE = (5.83, 4.78)
AXIS_LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 19
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 13

VARIANT_ORDER = ("none", "rule_local", "llm_edge", "llm_cloud")
VARIANT_LABEL = {
    "none": "None",
    "rule_local": "Rule",
    "llm_edge": "LLM@E",
    "llm_cloud": "LLM@C",
}
LINE_STYLES = {
    "none": (0, (1, 2)),
    "rule_local": (0, (5, 3)),
    "llm_edge": (0, (3, 2, 1, 2)),
    "llm_cloud": "solid",
}


def load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    rows.sort(key=lambda row: order.get(row["variant"], 99))
    return rows


def figure_tradeoff(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [VARIANT_LABEL.get(r["variant"], r["variant"]) for r in rows]
    detection = [float(r["detection_recall"]) for r in rows]
    false_reject = [float(r["false_reject_rate"]) for r in rows]
    unsafe = [float(r["unsafe_commit_rate"]) for r in rows]

    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    positions = range(len(labels))
    width = 0.27
    series = (
        ("detection", detection, BLUE, -width),
        ("false reject", false_reject, ORANGE, 0.0),
        ("unsafe commit", unsafe, RED, width),
    )
    for name, values, color, offset in series:
        ax.bar(
            [p + offset for p in positions], values, width,
            label=name, color=color, edgecolor=INK, linewidth=1.2, zorder=3,
        )
        for p, v in zip(positions, values):
            ax.text(
                p + offset, v + 0.02, f"{v:.2f}".lstrip("0") if v else "0",
                ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE, color=INK,
            )
    ax.set_ylim(0, 1.18)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_ylabel("rate", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    legend = ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left", ncols=1, framealpha=0.9)
    legend.get_frame().set_edgecolor(INK)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.12, top=0.95)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def figure_latency(rows: list[dict[str, Any]], out: Path, rtt_ms_max: float) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    rtts = [rtt_ms_max * i / 100 for i in range(101)]
    for row in rows:
        base = float(row["mean_latency_s"])
        crossings = float(row["mean_cloud_calls"])
        variant = row["variant"]
        ax.plot(
            rtts,
            [base + crossings * rtt / 1000.0 for rtt in rtts],
            label=VARIANT_LABEL.get(variant, variant),
            color=BLUE if variant == "llm_cloud" else GRAY,
            linewidth=3 if variant == "llm_cloud" else 2,
            linestyle=LINE_STYLES.get(variant, "solid"),
            zorder=3,
        )
    ax.set_xlabel("edge-cloud RTT (ms)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=6)
    ax.set_ylabel("pipeline latency (s)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=6)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    legend = ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left", framealpha=0.9)
    legend.get_frame().set_edgecolor(INK)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.95)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--rtt-ms-max", type=float, default=500.0)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((ROOT / "outputs").glob("micro_obs2_*"))
        if not candidates:
            print("no micro_obs2_* run directories found")
            return 1
        run_dir = candidates[-1]

    rows = load_metrics(run_dir)
    outdir = args.outdir or run_dir
    outdir.mkdir(parents=True, exist_ok=True)
    figure_tradeoff(rows, outdir / "fig_obs2_tradeoff")
    figure_latency(rows, outdir / "fig_obs2_latency", args.rtt_ms_max)
    print(f"wrote {outdir}/fig_obs2_tradeoff.[pdf|png]")
    print(f"wrote {outdir}/fig_obs2_latency.[pdf|png]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
