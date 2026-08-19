#!/usr/bin/env python3
"""Figures for Observation 3 (the commit barrier).

fig_obs3_recovery  What happens to the commits verification rejected, split by
                   recoverability class. This is the robust claim: it is
                   conditional on a rejection having occurred, so it does not
                   depend on how often the verifier rejects, which varies with
                   planner quality and injection rate.

fig_obs3_latency   Latency against commit cost, per class, for both policies.
                   Speculation is *fastest* on irreversible operations -- it
                   never pays for compensation because compensation refuses --
                   and it is the only configuration that leaves damage. A
                   scheduler optimising latency alone picks it.

Reads metrics.csv and latency_vs_rtt.csv from one or more micro_obs3 run
directories; runs nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
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

BLUE = BRIGHT_PALETTE["blue"]
ORANGE = BRIGHT_PALETTE["orange"]
RED = BRIGHT_PALETTE["red"]
GRAY = BRIGHT_PALETTE["dim_gray"]
INK = BRIGHT_PALETTE["black"]

FIG_SIZE = (5.83, 4.78)
WIDE_SIZE = (8.6, 3.6)
AXIS_LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 19
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 13

CLASS_ORDER = ("reversible", "compensable", "irreversible")
CLASS_LABEL = {
    "reversible": "Reversible",
    "compensable": "Compensable",
    "irreversible": "Irreversible",
}


def wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    latency: list[dict[str, str]] = []
    latency_path = run_dir / "latency_vs_rtt.csv"
    if latency_path.exists():
        with latency_path.open(encoding="utf-8", newline="") as handle:
            latency = list(csv.DictReader(handle))
    label = manifest.get("placement", run_dir.name)
    return {"dir": run_dir, "label": label, "metrics": metrics, "latency": latency}


def figure_recovery(runs: list[dict[str, Any]], out: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    positions = range(len(CLASS_ORDER))
    width = 0.8 / max(1, len(runs))

    for index, run in enumerate(runs):
        rows = {
            r["recoverability"]: r
            for r in run["metrics"]
            if r["policy"] == "speculative"
        }
        offset = (index - (len(runs) - 1) / 2) * width
        rates: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        counts: list[int] = []
        for recoverability in CLASS_ORDER:
            row = rows.get(recoverability)
            invoked = int(row["compensation_invoked_n"]) if row else 0
            rate = float(row["recovery_success_rate"]) if row else float("nan")
            rate = 0.0 if math.isnan(rate) else rate
            low, high = wilson(rate, invoked)
            rates.append(rate)
            lows.append(max(0.0, rate - low))
            highs.append(max(0.0, high - rate))
            counts.append(invoked)
        ax.bar(
            [p + offset for p in positions], rates, width * 0.92,
            label=run["label"], color=BLUE if index == 0 else ORANGE,
            edgecolor=INK, linewidth=1.2,
            yerr=[lows, highs],
            error_kw={"ecolor": GRAY, "elinewidth": 1.4, "capsize": 3},
            zorder=3,
        )
        # n is the number of rejected commits compensation ran on, not the cell
        # size: the claim is conditional on a rejection having happened. Printed
        # above each bar's own error bar so the legend cannot cover it.
        for p, rate, high, count in zip(positions, rates, highs, counts):
            ax.text(
                p + offset, (rate + high + 0.03) if count else 0.03,
                f"n={count}" if count else "none",
                ha="center", va="bottom",
                fontsize=ANNOTATION_FONT_SIZE, color=INK,
            )

    ax.set_ylim(0, 1.28)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([CLASS_LABEL[c] for c in CLASS_ORDER], fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_ylabel("recovered | rejected", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    legend = ax.legend(
        fontsize=LEGEND_FONT_SIZE, loc="lower left", ncols=2, framealpha=0.9,
        bbox_to_anchor=(0.0, -0.30),
    )
    legend.get_frame().set_edgecolor(INK)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.26, top=0.95)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def figure_latency(run: dict[str, Any], out: Path) -> None:
    """Latency difference between the policies against commit cost.

    Plotted as speculative minus conservative rather than as two absolute
    curves: the policy effect is tens of milliseconds on a ~4 s pipeline, so on
    an absolute axis the curves overlap and the sign -- the only thing that
    decides which policy to pick -- is invisible. Below zero means speculation
    is faster.
    """
    if not run["latency"] or "commit_ms" not in run["latency"][0]:
        print(f"  (no commit-cost sweep in {run['dir'].name}; skipping latency figure)")
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    damage = {
        r["recoverability"]: float(r["policy_induced_damage_rate"])
        for r in run["metrics"]
        if r["policy"] == "speculative"
    }
    colours = {"reversible": GRAY, "compensable": ORANGE, "irreversible": RED}

    for recoverability in CLASS_ORDER:
        by_commit: dict[float, dict[str, float]] = {}
        for row in run["latency"]:
            if row["recoverability"] != recoverability or float(row["rtt_ms"]) != 0.0:
                continue
            by_commit.setdefault(float(row["commit_ms"]), {})[row["policy"]] = float(
                row["mean_latency_s"]
            )
        points = sorted(
            (commit, pair["speculative"] - pair["conservative"])
            for commit, pair in by_commit.items()
            if {"speculative", "conservative"} <= pair.keys()
        )
        if not points:
            continue
        harm = damage.get(recoverability, 0.0)
        ax.plot(
            [p[0] for p in points], [p[1] for p in points],
            label=f"{CLASS_LABEL[recoverability]}  (damage {harm:.2f})",
            color=colours[recoverability], linewidth=2.8,
            linestyle="solid" if harm > 0 else (0, (5, 3)),
            marker="o", markersize=6, zorder=3,
        )

    ax.axhline(0.0, color=INK, linewidth=1.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("commit cost (ms)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=6)
    ax.set_ylabel("speculative $-$ conservative (s)",
                  fontsize=AXIS_LABEL_FONT_SIZE - 5, labelpad=6)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.text(
        0.03, 0.06, "below 0: speculation faster",
        transform=ax.transAxes, fontsize=ANNOTATION_FONT_SIZE, color=INK,
    )
    legend = ax.legend(fontsize=ANNOTATION_FONT_SIZE + 1, loc="upper left", framealpha=0.9)
    legend.get_frame().set_edgecolor(INK)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.16, top=0.95)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    runs = [load(d) for d in args.run_dir if (d / "metrics.csv").exists()]
    if not runs:
        print("no micro_obs3 run directories with metrics.csv")
        return 1
    outdir = args.outdir or runs[-1]["dir"]
    outdir.mkdir(parents=True, exist_ok=True)

    figure_recovery(runs, outdir / "fig_obs3_recovery")
    print(f"wrote {outdir}/fig_obs3_recovery.[pdf|png]")
    figure_latency(runs[-1], outdir / "fig_obs3_latency")
    print(f"wrote {outdir}/fig_obs3_latency.[pdf|png]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
