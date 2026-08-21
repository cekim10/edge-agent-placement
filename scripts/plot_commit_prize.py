#!/usr/bin/env python3
"""What speculation is worth against the commit cost, at measured and at fixed approval.

The saving is not monotone in the commit cost, which is the opposite of what the
speculative story assumes. Overlapping commit with verification hides at most
`min(v, c)`; past `c = v` there is no verification left to hide behind, while a
rejected commit still pays a second round trip to compensate. The two terms
cross, so the benefit window is closed on both sides and peaks near `c = v`.

Why two panels
--------------
The three classes do not share an approval rate -- placement and task difficulty
set it separately for each -- so a single panel of measured curves varies `a` and
recoverability at once, and the reader cannot tell which one shaped the curve.
That confound produced a wrong claim: read from measured curves alone,
irreversible appeared to have no upper bound at all, its saving growing without
limit in `c`. It does not. The saving fails to close only when `a = 1.00`
exactly, because only then is compensation never invoked; the class whose
measured approval happened to be 1.00 was the class that looked unbounded.

The right panel removes the confound by reweighting every class to one approval
rate, leaving the recovery rate `r` as the only difference between them. The
ordering survives -- irreversible peaks highest and closes latest, since
compensation refuses at no cost -- but as a matter of degree, not of kind.

The reweighting
---------------
Approved and rejected records are averaged separately and recombined at the
target rate, so no record is invented and `r` stays as measured within the
rejected subset. It needs both subsets to be non-empty: a class where every
commit was approved cannot be reweighted downward, which is exactly the case
that produced the retracted claim.

`a` defaults to 0.60 because it sits above the tie point of all three classes
(0.49, 0.50, 0.38), so every class is in the regime where a latency model would
consider speculating -- which is the regime the comparison is about.

Measured and derived
--------------------
Every point is composed from measured per-record components -- upstream stages,
verification, approval and compensation outcomes. Only `c` is swept, being a
deployment parameter rather than a property of the workload, and only `a` is
reweighted. The marked commit cost comes from measure_commit_cost.py against a
real durable service.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from tools.paper_plot_style import BRIGHT_PALETTE, configure_plot_style, finish_paper_axes
except ImportError:
    BRIGHT_PALETTE = {
        "black": "#000000", "dim_gray": "#696969", "dark_gray": "#A9A9A9",
        "blue": "#486EE2", "orange": "#FFB226", "red": "#D94B4B", "white": "#FFFFFF",
    }

    def configure_plot_style() -> None:
        matplotlib.rcParams.update({
            "font.family": "Times New Roman", "mathtext.fontset": "cm",
            "axes.linewidth": 2, "pdf.fonttype": 42, "ps.fonttype": 42,
        })

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
CLASS_LABEL = {"reversible": "Reversible", "compensable": "Compensable",
               "irreversible": "Irreversible"}
CLASS_COLOUR = {"reversible": GRAY, "compensable": BLUE, "irreversible": RED}
CLASS_DASH = {"reversible": (4, 2), "compensable": (), "irreversible": (1, 1.6)}

FIG_SIZE = (10.2, 4.5)
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 12


def load(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for recoverability in CLASS_ORDER:
        path = run_dir / f"{recoverability}_speculative.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.open() if line.strip()]
        out[recoverability] = [r for r in rows if r.get("ok")]
    return out


def saving_fraction(
    records: list[dict[str, Any]], commit_s: float, approval: float | None = None
) -> float:
    """(conservative - speculative) / conservative, composed per record.

    With `approval` set, the approved and rejected subsets are averaged apart and
    recombined at that rate instead of at the measured one.
    """
    def conservative(record: dict[str, Any], approved: bool) -> float:
        return (
            record["components"]["t_stages"]
            + record["components"]["t_verify"]
            + (commit_s if approved else 0.0)
        )

    def speculative(record: dict[str, Any]) -> float:
        compensated = bool(
            record["score"]["compensation_invoked"]
            and record["score"]["compensation_applied"]
        )
        return (
            record["components"]["t_stages"]
            + max(commit_s, record["components"]["t_verify"])
            + (commit_s if compensated else 0.0)
        )

    if approval is None:
        cons = sum(conservative(r, r["score"]["approved"]) for r in records)
        spec = sum(speculative(r) for r in records)
        return (cons - spec) / cons

    approved = [r for r in records if r["score"]["approved"]]
    rejected = [r for r in records if not r["score"]["approved"]]
    if not approved or not rejected:
        return float("nan")
    cons = (
        approval * statistics.mean(conservative(r, True) for r in approved)
        + (1 - approval) * statistics.mean(conservative(r, False) for r in rejected)
    )
    spec = (
        approval * statistics.mean(speculative(r) for r in approved)
        + (1 - approval) * statistics.mean(speculative(r) for r in rejected)
    )
    return (cons - spec) / cons


def curve_summary(
    records: list[dict[str, Any]], commits: np.ndarray, approval: float | None
) -> tuple[np.ndarray, int, float | None] | None:
    """Savings over the sweep, the peak index, and where it crosses back to zero."""
    savings = np.array([saving_fraction(records, c, approval) for c in commits])
    if np.isnan(savings).all():
        return None
    peak = int(np.nanargmax(savings))
    after = np.where(savings[peak:] < 0)[0]
    zero = float(commits[peak + after[0]]) if len(after) else None
    return savings, peak, zero


def figure(
    cells: dict[str, list[dict[str, Any]]],
    marks: list[tuple[str, float]],
    approval: float,
    out: Path,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    commits = np.logspace(-4, 1.1, 400)  # 0.1 ms .. ~12 s

    panels = (
        (axes[0], None, "as measured"),
        (axes[1], approval, f"reweighted to $a$ = {approval:.2f}"),
    )
    summary: list[str] = []
    for ax, target, title in panels:
        ax.axhline(0.0, color=INK, linewidth=1.4, zorder=2)
        for recoverability in CLASS_ORDER:
            records = cells.get(recoverability)
            if not records:
                continue
            result = curve_summary(records, commits, target)
            if result is None:
                # Every commit approved: there is no rejected subset to reweight
                # against. Saying so on the panel is the honest version of the
                # claim this figure had to retract.
                ax.text(
                    2e-4, 26.5, f"{CLASS_LABEL[recoverability]}: every commit\n"
                    "approved, cannot reweight",
                    fontsize=ANNOTATION_FONT_SIZE - 1, color=CLASS_COLOUR[recoverability],
                    va="top",
                )
                continue
            savings, peak, zero = result
            ax.plot(
                commits, savings * 100, color=CLASS_COLOUR[recoverability],
                linewidth=2.6, dashes=CLASS_DASH[recoverability],
                label=CLASS_LABEL[recoverability], zorder=4,
            )
            ax.plot(
                commits[peak], savings[peak] * 100, marker="o", markersize=8,
                color=CLASS_COLOUR[recoverability], markeredgecolor=INK,
                markeredgewidth=1.4, zorder=5,
            )
            verify = statistics.mean(r["components"]["t_verify"] for r in records)
            a = statistics.mean(float(r["score"]["approved"]) for r in records)
            summary.append(
                f"{title:<24}{CLASS_LABEL[recoverability]:<14}"
                f"a={target if target is not None else a:.2f}  "
                f"peak {savings[peak]*100:6.2f}% at c={commits[peak]:5.3f}s "
                f"(v={verify:.3f}s)"
                + (f", back to zero at c={zero:.3f}s" if zero is not None
                   else ", never turns negative")
            )

        for index, (label, commit_s) in enumerate(marks):
            ax.axvline(commit_s, color=ORANGE, linewidth=2.2, dashes=(3, 2), zorder=3)
            ax.annotate(
                label, xy=(commit_s, 0.0), xycoords=("data", "axes fraction"),
                xytext=(6, 10 + index * 15), textcoords="offset points",
                fontsize=ANNOTATION_FONT_SIZE, color=ORANGE, ha="left", va="bottom",
            )

        ax.set_xscale("log")
        ax.set_xlim(1e-4, 12)
        # Clipped: past the crossing the curves dive well below -20%, which would
        # flatten the region the paper argues about. The crossings stay visible,
        # which is the part that matters -- that the window closes, not how fast.
        ax.set_ylim(-14, 30)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=8)
        ax.set_xlabel("commit round trip $c$ (s, log scale)",
                      fontsize=AXIS_LABEL_FONT_SIZE, labelpad=6)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        finish_paper_axes(ax)

    axes[0].set_ylabel("end-to-end latency saved (%)",
                       fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    legend = axes[1].legend(fontsize=LEGEND_FONT_SIZE, loc="upper left",
                            framealpha=0.94)
    legend.get_frame().set_edgecolor(INK)
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.155, top=0.90, wspace=0.07)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}\n")
    for line in summary:
        print("  " + line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--cost-json", type=Path, action="append", default=None,
        help="output of measure_commit_cost.py; its un-injected total becomes "
             "the measured mark. Repeatable.",
    )
    parser.add_argument(
        "--approval", type=float, default=0.60,
        help="approval rate to reweight every class to in the right panel. The "
             "default sits above all three tie points, so each class is in the "
             "regime where a latency model would consider speculating.",
    )
    parser.add_argument(
        "--mark-ms", default=None,
        help="comma-separated commit costs in ms to mark, when the cost JSON "
             "lives on the machine that measured it rather than this one",
    )
    parser.add_argument("--out", type=Path,
                        default=ROOT / "figures" / "fig_commit_prize")
    args = parser.parse_args()

    cells = load(args.run_dir)
    if not cells:
        print(f"no speculative records in {args.run_dir}")
        return 1

    marks: list[tuple[str, float]] = []
    for path in args.cost_json or []:
        data = json.loads(path.read_text())
        for label, terms in data.get("deployments", {}).items():
            if "injected" in label:
                continue  # injected rows are counterfactual, not deployments
            marks.append((f"measured $c$ = {terms['total']['mean']*1000:.2f} ms",
                          terms["total"]["mean"]))
    for raw in str(args.mark_ms or "").split(","):
        if raw.strip():
            value = float(raw)
            marks.append((f"measured $c$ = {value:.2f} ms", value / 1000.0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure(cells, marks, args.approval, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
