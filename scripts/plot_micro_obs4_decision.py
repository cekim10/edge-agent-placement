#!/usr/bin/env python3
"""Observation 4: the decision map for the commit policy.

One panel per recoverability class. Colour is how much latency speculation
saves; the black contour is where the two policies tie; hatching marks the
region where speculation is not an option regardless of what the latency says.

The surface is derived, the parameters in it are measured
--------------------------------------------------------
Speculation's saving over the conservative order is

    saving(a, c) = c * (a - m),      m = (1 - a) * r

with `a` the approval rate, `c` the commit round trip, and `r` the rate at which
compensation succeeds once it runs. `r` and the damage rate come from the
Observation 3 runs; `a` and `c` are swept so the map covers operating points the
experiments did not visit. Measured operating points are drawn on top, so the
derived surface can be checked against them.

Two consequences are worth reading off the map rather than the algebra:

  * the tie contour is vertical. The sign of the saving does not depend on the
    commit cost -- only its magnitude does. Choosing a policy is therefore a
    question about the approval rate, which placement controls, not about how
    expensive the commit is.
  * the class where speculation saves the most is the class where it is
    forbidden. A scheduler that reads only the colour picks exactly the panel
    that is hatched out.
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
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

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
        ax.set_axisbelow(True)


ROOT = Path(__file__).resolve().parents[1]

BLUE = BRIGHT_PALETTE["blue"]
ORANGE = BRIGHT_PALETTE["orange"]
INK = BRIGHT_PALETTE["black"]
GRAY = BRIGHT_PALETTE["dim_gray"]

CLASS_ORDER = ("reversible", "compensable", "irreversible")
CLASS_LABEL = {
    "reversible": "Reversible",
    "compensable": "Compensable",
    "irreversible": "Irreversible",
}

WIDE_SIZE = (9.4, 3.8)
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 12

# Speculation slower (orange) -> tie (near-white) -> speculation faster (blue).
DIVERGING = LinearSegmentedColormap.from_list(
    "spec_saving", [ORANGE, "#F7F3EC", BLUE]
)

# A class is treated as off-limits for speculation when its measured permanent
# damage exceeds this. The point is not the exact threshold: irreversible sits
# at 0.60 and the others at 0.00, so any threshold in between separates them.
DAMAGE_THRESHOLD = 0.05


def load_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["policy"] == "speculative"]
        runs.append({"dir": run_dir, "manifest": manifest, "rows": rows})
    return runs


def class_parameters(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recovery rate, damage rate and measured operating points, per class.

    `r` and the damage rate are taken from the run with the most compensation
    invocations: a class's recoverability is only observable through commits
    that were actually rejected, so the run that produced the most rejections
    measures it best.
    """
    params: dict[str, dict[str, Any]] = {}
    for recoverability in CLASS_ORDER:
        best: dict[str, Any] | None = None
        points: list[tuple[float, str]] = []
        for run in runs:
            row = next(
                (r for r in run["rows"] if r["recoverability"] == recoverability), None
            )
            if row is None:
                continue
            label = (
                f"{run['manifest'].get('placement', '?')}"
                f" / inj {run['manifest'].get('inject_rate', '?')}"
            )
            points.append((float(row["approved_rate"]), label))
            invoked = int(row["compensation_invoked_n"])
            if best is None or invoked > int(best["compensation_invoked_n"]):
                best = row
        if best is None:
            continue
        recovery = float(best["recovery_success_rate"])
        params[recoverability] = {
            "recovery": 0.0 if math.isnan(recovery) else recovery,
            "damage": float(best["policy_induced_damage_rate"]),
            "invoked": int(best["compensation_invoked_n"]),
            "points": points,
        }
    return params


def figure_decision_map(params: dict[str, dict[str, Any]], out: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=WIDE_SIZE, sharey=True)
    approvals = np.linspace(0.0, 1.0, 201)
    commits_ms = np.logspace(1, 3, 161)
    grid_a, grid_c = np.meshgrid(approvals, commits_ms)

    surfaces = {}
    for recoverability in CLASS_ORDER:
        entry = params.get(recoverability)
        if entry is None:
            continue
        recovery = entry["recovery"]
        compensated = (1.0 - grid_a) * recovery
        surfaces[recoverability] = (grid_c / 1000.0) * (grid_a - compensated)
    limit = max(float(np.abs(s).max()) for s in surfaces.values()) if surfaces else 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    mesh = None
    for ax, recoverability in zip(axes, CLASS_ORDER):
        entry = params.get(recoverability)
        if entry is None:
            ax.set_visible(False)
            continue
        saving = surfaces[recoverability]
        mesh = ax.pcolormesh(
            grid_a, grid_c, saving, cmap=DIVERGING, norm=norm, shading="auto",
        )
        ax.contour(grid_a, grid_c, saving, levels=[0.0], colors=[INK], linewidths=2.2)

        unsafe = entry["damage"] > DAMAGE_THRESHOLD
        if unsafe:
            ax.contourf(
                grid_a, grid_c, np.ones_like(saving), levels=[0.5, 1.5],
                colors="none", hatches=["///"],
            )
            ax.text(
                0.5, 0.80,
                f"speculation unavailable\n{entry['damage']:.0%} permanent damage",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=ANNOTATION_FONT_SIZE, color=INK,
                bbox={"facecolor": "white", "edgecolor": INK, "alpha": 0.88},
            )

        for approval, label in entry["points"]:
            ax.plot(
                [approval] * 2, [commits_ms[0], commits_ms[-1]],
                color=INK, linewidth=1.0, linestyle=(0, (1, 3)), zorder=4,
            )
            ax.plot(
                approval, commits_ms[len(commits_ms) // 2], marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.6,
                zorder=5,
            )

        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        # Three ticks only: adjacent panels otherwise print 1.00 and 0.00 on top
        # of each other at the shared boundary.
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_title(
            f"{CLASS_LABEL[recoverability]}   $r$={entry['recovery']:.2f}",
            fontsize=AXIS_LABEL_FONT_SIZE - 2, color=INK, pad=6,
        )
        ax.set_xlabel("approval rate $a$", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=4)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        finish_paper_axes(ax)

    axes[0].set_ylabel("commit cost (ms)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=4)
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=list(axes), fraction=0.026, pad=0.035)
        bar.set_label(
            "speculation saving (s)", fontsize=AXIS_LABEL_FONT_SIZE - 2, labelpad=6
        )
        bar.ax.tick_params(labelsize=TICK_FONT_SIZE - 2)
        bar.outline.set_linewidth(1.5)
    fig.subplots_adjust(left=0.075, right=0.855, bottom=0.19, top=0.86, wspace=0.17)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    runs = load_runs(list(args.run_dir))
    if not runs:
        print("no micro_obs3 run directories with metrics.csv")
        return 1
    params = class_parameters(runs)
    outdir = args.outdir or runs[-1]["dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    figure_decision_map(params, outdir / "fig_obs4_decision")

    print(f"wrote {outdir}/fig_obs4_decision.[pdf|png]")
    print(f"\n{'class':<14}{'r':>6}{'damage':>9}{'tie at a':>10}  measured a")
    print("-" * 62)
    for recoverability in CLASS_ORDER:
        entry = params.get(recoverability)
        if entry is None:
            continue
        recovery = entry["recovery"]
        tie = recovery / (1.0 + recovery)
        points = ", ".join(f"{a:.2f}" for a, _ in entry["points"])
        print(
            f"{CLASS_LABEL[recoverability]:<14}{recovery:>6.2f}{entry['damage']:>9.2f}"
            f"{tie:>10.2f}  {points}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
