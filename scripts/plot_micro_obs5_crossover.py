#!/usr/bin/env python3
"""Where the commit policy flips: approval rate, commit cost, and RTT.

Runs nothing. Every number comes from the per-record components already stored
in the Observation 3 runs; the approval rate is swept by reweighting those
records, not by re-running the pipeline.

The reweighting
---------------
Per record, with `v` its measured verification latency and `c` the commit round
trip:

    delta = max(c, v + rtt) - (v + rtt) + c * (compensated - approved)

`approved` and `compensated` are measured per record. Splitting the records into
the approved and rejected groups and averaging within each gives the mean delta
at any approval rate `a`:

    E[delta | a] = a * mean_approved(...) + (1 - a) * mean_rejected(...)

This is a counterfactual over the measured component distribution -- "what if
the verifier approved this often" -- not a new measurement. The runs' own
operating points are drawn on top so the curve can be checked where it was
actually observed.

Why RTT does not move the boundary here
---------------------------------------
While `c <= v + rtt` the bracket term vanishes and

    E[delta | a] = c * (m - a),      m = (1 - a) * r

whose sign is independent of both `c` and `rtt`. Verification crosses the
network under both policies, so the crossing cancels. RTT only enters once the
commit is more expensive than verification: then speculation hides the
verification behind the commit, and `a*` starts to fall as RTT grows. Our commit
sweep stopped at 1 s against a ~1.1-1.8 s verification, which is why the earlier
RTT sweep produced parallel lines. The second figure sweeps far enough to cross
that boundary.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
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
        ax.grid(alpha=0.18, linewidth=0.7)
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
CLASS_COLOUR = {"reversible": GRAY, "compensable": ORANGE, "irreversible": RED}

FIG_SIZE = (5.83, 4.78)
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 12


def load_class_records(run_dirs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Per class, the speculative records pooled across runs."""
    pooled: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_ORDER}
    for run_dir in run_dirs:
        for path in sorted(glob.glob(str(run_dir / "*_speculative.jsonl"))):
            recoverability = Path(path).name.replace("_speculative.jsonl", "")
            if recoverability not in pooled:
                continue
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("ok"):
                    continue
                pooled[recoverability].append(
                    {
                        "verify": float(record["components"]["t_verify"]),
                        "approved": bool(record["score"]["approved"]),
                        "compensated": bool(record["score"]["compensation_applied"]),
                        "run": run_dir.name,
                    }
                )
    return pooled


def measured_points(run_dirs: list[Path]) -> dict[str, list[tuple[float, str]]]:
    """Each run's own approval rate per class, for overlaying on the curve."""
    points: dict[str, list[tuple[float, str]]] = {name: [] for name in CLASS_ORDER}
    for run_dir in run_dirs:
        for path in sorted(glob.glob(str(run_dir / "*_speculative.jsonl"))):
            recoverability = Path(path).name.replace("_speculative.jsonl", "")
            if recoverability not in points:
                continue
            rows = [
                json.loads(line)
                for line in open(path, encoding="utf-8")
                if line.strip()
            ]
            ok = [r for r in rows if r.get("ok")]
            if not ok:
                continue
            approval = sum(1 for r in ok if r["score"]["approved"]) / len(ok)
            points[recoverability].append((approval, run_dir.name))
    return points


def delta_at(
    records: list[dict[str, Any]], approval: float, commit_s: float, rtt_s: float
) -> float:
    """Mean latency delta if the verifier approved at rate `approval`."""
    approved = [r for r in records if r["approved"]]
    rejected = [r for r in records if not r["approved"]]
    if not approved or not rejected:
        return float("nan")

    def bracket(record: dict[str, Any]) -> float:
        verify = record["verify"] + rtt_s
        return max(commit_s, verify) - verify

    mean_approved = sum(bracket(r) for r in approved) / len(approved) - commit_s
    mean_rejected = sum(
        bracket(r) + (commit_s if r["compensated"] else 0.0) for r in rejected
    ) / len(rejected)
    return approval * mean_approved + (1.0 - approval) * mean_rejected


def crossover_approval(
    records: list[dict[str, Any]], commit_s: float, rtt_s: float
) -> float | None:
    """Approval rate where the two policies tie, by bisection on the reweighting."""
    low, high = 0.001, 0.999
    delta_low = delta_at(records, low, commit_s, rtt_s)
    delta_high = delta_at(records, high, commit_s, rtt_s)
    if any(math.isnan(v) for v in (delta_low, delta_high)) or delta_low * delta_high > 0:
        return None
    for _ in range(60):
        mid = (low + high) / 2
        if delta_at(records, low, commit_s, rtt_s) * delta_at(records, mid, commit_s, rtt_s) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def figure_approval(
    pooled: dict[str, list[dict[str, Any]]],
    points: dict[str, list[tuple[float, str]]],
    out: Path,
    commit_ms: float,
    rtt_ms: float,
) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    approvals = np.linspace(0.02, 0.98, 193)
    commit_s, rtt_s = commit_ms / 1000.0, rtt_ms / 1000.0

    for recoverability in CLASS_ORDER:
        records = pooled.get(recoverability) or []
        if not records:
            continue
        curve = [delta_at(records, a, commit_s, rtt_s) for a in approvals]
        ax.plot(
            approvals, curve, color=CLASS_COLOUR[recoverability], linewidth=2.8,
            label=CLASS_LABEL[recoverability], zorder=3,
        )
        tie = crossover_approval(records, commit_s, rtt_s)
        if tie is not None:
            ax.axvline(
                tie, color=CLASS_COLOUR[recoverability], linewidth=1.4,
                linestyle=(0, (2, 3)), zorder=2,
            )
            ax.annotate(
                f"$a^*$={tie:.2f}", xy=(tie, 0), xytext=(4, 8),
                textcoords="offset points", rotation=90,
                fontsize=ANNOTATION_FONT_SIZE, color=CLASS_COLOUR[recoverability],
            )
        for approval, _run in points.get(recoverability, []):
            ax.plot(
                approval, delta_at(records, approval, commit_s, rtt_s),
                marker="o", markersize=8, markerfacecolor="white",
                markeredgecolor=CLASS_COLOUR[recoverability], markeredgewidth=1.8,
                zorder=4,
            )

    ax.axhline(0.0, color=INK, linewidth=1.8, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("approval rate $a$", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=5)
    ax.set_ylabel("speculative $-$ conservative (s)",
                  fontsize=AXIS_LABEL_FONT_SIZE - 3, labelpad=5)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.text(
        0.03, 0.05, f"commit {commit_ms:.0f} ms, RTT {rtt_ms:.0f} ms\nbelow 0: speculation faster",
        transform=ax.transAxes, fontsize=ANNOTATION_FONT_SIZE, color=INK,
    )
    legend = ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper right", framealpha=0.9)
    legend.get_frame().set_edgecolor(INK)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.14, top=0.96)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def figure_rtt(
    pooled: dict[str, list[dict[str, Any]]],
    out: Path,
    commit_ms_list: list[float],
    rtt_ms: np.ndarray,
) -> None:
    """How the tie point moves with RTT, for commit costs either side of verify."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    records = pooled.get("compensable") or []
    if not records:
        print("  (no compensable records; skipping RTT figure)")
        return
    mean_verify = sum(r["verify"] for r in records) / len(records)

    styles = ["solid", (0, (5, 3)), (0, (3, 2, 1, 2)), (0, (1, 2))]
    for index, commit_ms in enumerate(sorted(commit_ms_list)):
        ties = [
            crossover_approval(records, commit_ms / 1000.0, rtt / 1000.0)
            for rtt in rtt_ms
        ]
        xs = [rtt for rtt, tie in zip(rtt_ms, ties) if tie is not None]
        ys = [tie for tie in ties if tie is not None]
        if not xs:
            continue
        beyond = commit_ms / 1000.0 > mean_verify
        ax.plot(
            xs, ys, linewidth=2.8 if beyond else 2.0,
            color=BLUE if beyond else GRAY,
            linestyle=styles[index % len(styles)],
            label=f"commit {commit_ms:.0f} ms" + (" (> verify)" if beyond else ""),
            zorder=3,
        )

    ax.set_xlabel("edge$-$cloud RTT (ms)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=5)
    ax.set_ylabel("tie approval rate $a^*$", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=5)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.text(
        0.03, 0.06,
        f"compensable, mean verify {mean_verify:.2f} s\n"
        "flat while commit < verify",
        transform=ax.transAxes, fontsize=ANNOTATION_FONT_SIZE, color=INK,
    )
    legend = ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper right", framealpha=0.9)
    legend.get_frame().set_edgecolor(INK)
    finish_paper_axes(ax)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.96)
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--commit-ms", type=float, default=1000.0)
    parser.add_argument("--rtt-ms", type=float, default=0.0)
    parser.add_argument("--rtt-commit-ms", default="500,1000,3000,6000")
    args = parser.parse_args()

    run_dirs = [d for d in args.run_dir if d.exists()]
    pooled = load_class_records(run_dirs)
    points = measured_points(run_dirs)
    if not any(pooled.values()):
        print("no speculative records found")
        return 1

    outdir = args.outdir or run_dirs[-1]
    outdir.mkdir(parents=True, exist_ok=True)
    figure_approval(pooled, points, outdir / "fig_obs5_approval", args.commit_ms, args.rtt_ms)
    figure_rtt(
        pooled, outdir / "fig_obs5_rtt",
        [float(v) for v in args.rtt_commit_ms.split(",") if v.strip()],
        np.linspace(0.0, 500.0, 51),
    )
    print(f"wrote {outdir}/fig_obs5_approval.[pdf|png]")
    print(f"wrote {outdir}/fig_obs5_rtt.[pdf|png]")

    print(f"\ntie approval rate at commit {args.commit_ms:.0f} ms, RTT {args.rtt_ms:.0f} ms")
    print(f"{'class':<14}{'n':>5}{'verify (s)':>12}{'a*':>8}   measured a")
    print("-" * 60)
    for recoverability in CLASS_ORDER:
        records = pooled.get(recoverability) or []
        if not records:
            continue
        tie = crossover_approval(records, args.commit_ms / 1000.0, args.rtt_ms / 1000.0)
        verify = sum(r["verify"] for r in records) / len(records)
        observed = ", ".join(f"{a:.2f}" for a, _ in points.get(recoverability, []))
        print(
            f"{CLASS_LABEL[recoverability]:<14}{len(records):>5}{verify:>12.2f}"
            f"{('n/a' if tie is None else f'{tie:.2f}'):>8}   {observed}"
        )

    print("\ntie approval rate vs RTT (compensable)")
    header = f"{'commit (ms)':<14}" + "".join(f"{r:>9.0f}" for r in (0, 100, 250, 500))
    print(header)
    print("-" * len(header))
    records = pooled.get("compensable") or []
    for commit_ms in [float(v) for v in args.rtt_commit_ms.split(",") if v.strip()]:
        cells = []
        for rtt in (0, 100, 250, 500):
            tie = crossover_approval(records, commit_ms / 1000.0, rtt / 1000.0)
            cells.append("n/a" if tie is None else f"{tie:.2f}")
        print(f"{commit_ms:<14.0f}" + "".join(f"{c:>9}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
