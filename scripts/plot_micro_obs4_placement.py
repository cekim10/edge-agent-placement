#!/usr/bin/env python3
"""Observation 4, motivation: placement moves the approval rate across the
profitability boundary.

One panel per recoverability class. The points are the measured approval rate
`a` at each placement; the dashed line is the tie point `a*`, above which
speculating is the faster policy. The claim the figure makes is not that
speculation is good or bad -- it is that the *same* class flips which policy
wins depending on where the stages ran. Policy selection therefore cannot be
made statically at design time; it has to read a quantity that placement sets.

Both curves are measured; only the map between them is algebra
---------------------------------------------------------------
`a` is measured, one run per placement, with plan injection switched OFF so the
approval rate reflects the planner's own quality -- the thing placement controls
-- and nothing else.

`a* = r / (1 + r)` is derived from `r`, the rate at which compensation actually
pays its round trip once a commit is rejected, and it holds while the commit
costs no more than verification (with `c > v` the tie moves to
`a* = 1 - v/(c(1+r))`; see compare_micro_obs3.py). `r` is measured in the *same*
run as the `a` it is compared against, per placement.

Measuring `r` locally is not fussiness. An earlier version of this figure
borrowed one `r` per class from a high-injection run, on the reasoning that
recovery is a property of the compensation mechanism. That reasoning is wrong
for irreversible operations. Compensation there refuses exactly when the
committed plan contained an irreversible op, so `r` is a property of the *error
distribution* -- and the error distribution is what placement changes. Injected
faults preserve the op type and yield r=0.16; genuine planner faults change it
and yield r=0.47 to 0.77, moving a* from 0.14 to 0.43. Borrowing put the line in
the wrong place.

Both curves carry Wilson intervals: `a` directly, `a*` by mapping the interval
on `r` through r/(1+r), which is monotone. A placement is only called for one
policy when the two intervals do not overlap.

Where every commit was approved there are no rejections, `r` is unmeasurable and
`a*` is undefined. That is not a gap in the data: with nothing ever rejected,
compensation never runs and speculation wins for any `r` whatsoever. Those
placements are drawn without an a* marker and labelled.
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
        ax.set_axisbelow(True)


ROOT = Path(__file__).resolve().parents[1]

BLUE = BRIGHT_PALETTE["blue"]
ORANGE = BRIGHT_PALETTE["orange"]
RED = BRIGHT_PALETTE["red"]
GRAY = BRIGHT_PALETTE["dim_gray"]
INK = BRIGHT_PALETTE["black"]

CLASS_ORDER = ("reversible", "compensable", "irreversible")
CLASS_LABEL = {
    "reversible": "Reversible",
    "compensable": "Compensable",
    "irreversible": "Irreversible",
}

# Ordered by how much of the pipeline sits on the edge, so the x axis reads as a
# single monotone "more edge" direction and any trend in `a` is visible as slope.
PLACEMENT_ORDER = ("all_cloud", "edge_classify", "edge_plan", "all_edge")
PLACEMENT_LABEL = {
    "all_cloud": "all\ncloud",
    "edge_classify": "edge\nclassify",
    "edge_plan": "edge\nplan",
    "all_edge": "all\nedge",
}

WIDE_SIZE = (9.4, 3.6)
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 13
TITLE_FONT_SIZE = 16
ANNOTATION_FONT_SIZE = 11


def wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def read_metrics(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["policy"] == "speculative"]
    return manifest, rows


def normalise_placement(name: str) -> str:
    """The runner takes `classify`/`plan`; the outputs name them `edge_*`."""
    if name in ("classify", "plan"):
        return f"edge_{name}"
    return name


def load_cells(run_dirs: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """cell[class][placement] -> approval, recovery and the counts behind them."""
    cells: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in CLASS_ORDER}
    for run_dir in run_dirs:
        if not (run_dir / "metrics.csv").exists():
            print(f"skipping {run_dir}: no metrics.csv")
            continue
        manifest, rows = read_metrics(run_dir)
        placement = normalise_placement(str(manifest.get("placement", "")))
        if placement not in PLACEMENT_ORDER:
            print(f"skipping {run_dir}: unknown placement {placement!r}")
            continue
        rate = manifest.get("inject_rate")
        if rate not in (None, 0, 0.0):
            print(
                f"  warning: {run_dir.name} has inject_rate={rate}; its approval rate"
                " is capped by injection, not set by the planner"
            )
        for row in rows:
            recoverability = row["recoverability"]
            if recoverability not in cells:
                continue
            recovery = float(row["recovery_success_rate"])
            invoked = int(row["compensation_invoked_n"])
            cells[recoverability][placement] = {
                "a": float(row["approved_rate"]),
                "n": int(row["n"]),
                "r": None if (invoked == 0 or math.isnan(recovery)) else recovery,
                "invoked": invoked,
            }
    return cells


def tie_point(entry: dict[str, Any]) -> tuple[float, float, float] | None:
    """(a*, low, high) from the measured recovery rate, or None if unmeasurable.

    r/(1+r) is increasing in r, so the interval on `a*` is the interval on `r`
    mapped through the same expression.
    """
    r = entry["r"]
    if r is None:
        return None
    low, high = wilson(r, entry["invoked"])
    return (r / (1 + r), low / (1 + low), high / (1 + high))


def figure_single(
    cells: dict[str, dict[str, dict[str, Any]]], recoverability: str, out: Path
) -> None:
    """One class, full width: the bridge between placement and policy.

    The three-panel version answers "does this hold across recoverability
    classes". This one answers the single question a reader of the motivation
    needs settled -- whether moving one stage can carry a workflow across the
    line where the optimal commit policy changes -- and it answers it with the
    class where both sides of the crossing are resolved.
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    points = cells.get(recoverability, {})
    positions = list(range(len(PLACEMENT_ORDER)))

    ties = [(i, tie_point(points[p]))
            for i, p in enumerate(PLACEMENT_ORDER) if p in points]
    tie_x = [i for i, t in ties if t is not None]
    tie_y = [t[0] for _i, t in ties if t is not None]
    tie_lo = [t[1] for _i, t in ties if t is not None]
    tie_hi = [t[2] for _i, t in ties if t is not None]

    # Name the two half-planes rather than leaving the reader to derive which
    # side means what. The shading is the weaker cue and the words are the
    # stronger one, so the figure survives being read quickly.
    ceiling = 1.16
    if tie_x:
        ax.fill_between(tie_x, tie_y, ceiling, color=BLUE, alpha=0.10, zorder=0)
        ax.fill_between(tie_x, 0.0, tie_y, color=ORANGE, alpha=0.10, zorder=0)
        ax.fill_between(tie_x, tie_lo, tie_hi, color=INK, alpha=0.16, zorder=1)
        ax.plot(tie_x, tie_y, color=INK, linestyle="--", linewidth=2.2, zorder=2)
        ax.text(
            len(positions) - 0.6, tie_y[-1] + 0.05, "speculate", ha="right",
            va="bottom", fontsize=ANNOTATION_FONT_SIZE + 2, color=BLUE, style="italic",
        )
        ax.text(
            len(positions) - 0.6, tie_y[-1] - 0.05, "verify first", ha="right",
            va="top", fontsize=ANNOTATION_FONT_SIZE + 2, color=ORANGE, style="italic",
        )
        ax.text(
            -0.4, tie_y[0] + 0.02, f"$a^*$ = {tie_y[0]:.2f}", ha="left", va="bottom",
            fontsize=ANNOTATION_FONT_SIZE + 1, color=INK,
        )

    xs, ys, los, his = [], [], [], []
    for index, placement in enumerate(PLACEMENT_ORDER):
        entry = points.get(placement)
        if entry is None:
            continue
        low, high = wilson(entry["a"], entry["n"])
        xs.append(index); ys.append(entry["a"])
        los.append(max(0.0, entry["a"] - low)); his.append(max(0.0, high - entry["a"]))
    ax.plot(xs, ys, color=GRAY, linewidth=2.0, zorder=3)
    for x, y, lo, hi in zip(xs, ys, los, his):
        tie = tie_point(points[PLACEMENT_ORDER[x]])
        above = tie is None or y > tie[0]
        ax.errorbar(
            x, y, yerr=[[lo], [hi]], fmt="o", markersize=11,
            color=BLUE if above else ORANGE,
            markerfacecolor=(BLUE if above else "white"),
            markeredgecolor=BLUE if above else ORANGE, markeredgewidth=2.4,
            ecolor=GRAY, elinewidth=1.8, capsize=4, zorder=4,
        )
        # n varies by a factor of four across placements, which is most of why
        # the intervals differ in width. Hiding that would make the two wide
        # points look like noisier measurements of the same thing.
        ax.text(
            x, y - lo - 0.045, f"n={points[PLACEMENT_ORDER[x]]['n']}",
            ha="center", va="top", fontsize=ANNOTATION_FONT_SIZE - 1, color=GRAY,
        )

    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0.0, ceiling)
    ax.set_xticks(positions)
    ax.set_xticklabels([PLACEMENT_LABEL[p] for p in PLACEMENT_ORDER],
                       fontsize=TICK_FONT_SIZE + 1)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE + 1)
    ax.set_ylabel("approval rate $a$", fontsize=AXIS_LABEL_FONT_SIZE + 1, labelpad=8)
    ax.set_xlabel("stage placement", fontsize=AXIS_LABEL_FONT_SIZE + 1, labelpad=8)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.165, top=0.965)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")


def figure(cells: dict[str, dict[str, dict[str, Any]]], out: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=WIDE_SIZE, sharey=True)
    positions = list(range(len(PLACEMENT_ORDER)))

    for ax, recoverability in zip(axes, CLASS_ORDER):
        points = cells.get(recoverability, {})

        # a* first and underneath: it is the reference the measurement is read
        # against, not a second result competing for attention.
        ties = [(i, tie_point(points[p]))
                for i, p in enumerate(PLACEMENT_ORDER) if p in points]
        tie_x = [i for i, t in ties if t is not None]
        tie_y = [t[0] for _i, t in ties if t is not None]
        tie_lo = [t[1] for _i, t in ties if t is not None]
        tie_hi = [t[2] for _i, t in ties if t is not None]
        if tie_x:
            ax.fill_between(tie_x, tie_lo, tie_hi, color=INK, alpha=0.13, zorder=1)
            ax.plot(
                tie_x, tie_y, color=INK, linestyle="--", linewidth=2.0,
                marker="_", markersize=13, markeredgewidth=2.0,
                label="$a^*$ (tie point)", zorder=2,
            )
        for index, placement in enumerate(PLACEMENT_ORDER):
            entry = points.get(placement)
            if entry is not None and entry["r"] is None:
                ax.text(
                    index, 0.06, "no\nreject", ha="center", va="bottom",
                    fontsize=ANNOTATION_FONT_SIZE - 1, color=GRAY, style="italic",
                )

        xs, ys, los, his = [], [], [], []
        for index, placement in enumerate(PLACEMENT_ORDER):
            entry = points.get(placement)
            if entry is None:
                continue
            low, high = wilson(entry["a"], entry["n"])
            xs.append(index)
            ys.append(entry["a"])
            los.append(max(0.0, entry["a"] - low))
            his.append(max(0.0, high - entry["a"]))
        if xs:
            ax.plot(xs, ys, color=GRAY, linewidth=1.8, zorder=3,
                    label="$a$ (approval rate)")
            for x, y, lo, hi in zip(xs, ys, los, his):
                tie = tie_point(points[PLACEMENT_ORDER[x]])
                # Filled above the tie point, hollow below, so which policy wins
                # survives greyscale printing and a colour-blind reader.
                above = tie is None or y > tie[0]
                ax.errorbar(
                    x, y, yerr=[[lo], [hi]], fmt="o", markersize=9,
                    color=BLUE if above else ORANGE,
                    markerfacecolor=(BLUE if above else "white"),
                    markeredgecolor=BLUE if above else ORANGE,
                    markeredgewidth=2.2,
                    ecolor=GRAY, elinewidth=1.6, capsize=3, zorder=4,
                )

        ax.set_xlim(-0.5, len(positions) - 0.5)
        ax.set_ylim(0.0, 1.22)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [PLACEMENT_LABEL[p] for p in PLACEMENT_ORDER], fontsize=TICK_FONT_SIZE
        )
        ax.set_title(CLASS_LABEL[recoverability], fontsize=TITLE_FONT_SIZE, pad=8)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        finish_paper_axes(ax)

    axes[0].set_ylabel("rate", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    legend = axes[0].legend(fontsize=ANNOTATION_FONT_SIZE, loc="upper right",
                            framealpha=0.92)
    legend.get_frame().set_edgecolor(INK)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.88, wspace=0.10)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")


def table(cells: dict[str, dict[str, dict[str, Any]]]) -> None:
    header = (
        f"{'class':<14}{'placement':<15}{'a':>6}{'  a 95% CI':>15}"
        f"{'r':>7}{'a*':>7}{'  a* 95% CI':>15}   verdict"
    )
    print("\n" + header)
    print("-" * len(header))
    for recoverability in CLASS_ORDER:
        for placement in PLACEMENT_ORDER:
            entry = cells.get(recoverability, {}).get(placement)
            if entry is None:
                continue
            a_lo, a_hi = wilson(entry["a"], entry["n"])
            tie = tie_point(entry)
            if tie is None:
                print(
                    f"{recoverability:<14}{placement:<15}{entry['a']:>6.2f}"
                    f"   [{a_lo:.2f}, {a_hi:.2f}]{'--':>7}{'--':>7}"
                    f"{'  no rejections':>15}   spec (any r)"
                )
                continue
            star, s_lo, s_hi = tie
            if a_lo > s_hi:
                verdict = "spec (resolved)"
            elif a_hi < s_lo:
                verdict = "cons (resolved)"
            else:
                verdict = "UNRESOLVED"
            print(
                f"{recoverability:<14}{placement:<15}{entry['a']:>6.2f}"
                f"   [{a_lo:.2f}, {a_hi:.2f}]{entry['r']:>7.2f}{star:>7.2f}"
                f"   [{s_lo:.2f}, {s_hi:.2f}]   {verdict}"
            )
        print()
    print(
        "Verdict is resolved only when the interval on the measured approval rate\n"
        "and the interval on the derived tie point do not overlap. A class whose\n"
        "rows contain both 'spec' and 'cons' is one where placement decides the\n"
        "policy."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir", type=Path, nargs="+",
        help="zero-injection micro_obs3 runs, one per placement",
    )
    parser.add_argument(
        "--only-class", choices=CLASS_ORDER, default=None,
        help="draw one class full width as the motivation bridge figure. "
             "compensable is the one whose crossing is resolved on both sides.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
    )
    args = parser.parse_args()

    cells = load_cells(args.run_dir)
    if not any(cells.values()):
        print("no usable runs")
        return 1
    default_name = (
        f"fig_obs4_boundary_{args.only_class}" if args.only_class
        else "fig_obs4_placement_boundary"
    )
    out = args.out or (ROOT / "figures" / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.only_class:
        figure_single(cells, args.only_class, out)
    else:
        figure(cells, out)
    table(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
