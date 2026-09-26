"""Every benchmark figure, generated from a results directory. Never hand-edited.

    uv run python scripts/plot_benchmark.py results/canonical

Writes into ``docs/figures/``:

    bench_gap_vs_n.png              optimality gap vs N, per solver
    bench_runtime_vs_n.png          time to solution vs N, log scale
    qubo_penalty_sweep.png          QUBO feasibility and quality vs penalty weight
    best_sequence_raan_altitude.png the best-known route on the plane that costs

Needs the ``physics`` extra (matplotlib): ``uv sync --all-extras``.

Every number plotted here comes out of the CSVs in the results directory. The
script never solves anything -- if a figure and the README disagree, the figure
is stale and the fix is to re-run it, not to edit it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

#: Fixed hue order, assigned per solver and never cycled. Validated for the
#: adjacent pairlist in light mode (worst adjacent CVD dE 9.1, normal-vision
#: 19.6). Three of these sit below 3:1 contrast on a light surface, so every
#: series is also direct-labelled -- that is the relief, not a nicety.
SOLVER_COLOURS = {
    "exact": "#2a78d6",
    "greedy": "#eb6834",
    "cpsat": "#1baf7a",
    "ortools": "#eda100",
    "localsearch": "#e87ba4",
    "sa-perm": "#008300",
    "sa-qubo": "#4a3aa7",
    "qaoa": "#e34948",
}
#: Secondary encoding, so identity never rests on hue alone.
SOLVER_MARKERS = {
    "exact": "o",
    "greedy": "s",
    "cpsat": "D",
    "ortools": "^",
    "localsearch": "v",
    "sa-perm": "P",
    "sa-qubo": "X",
    "qaoa": "*",
}
#: ``brute`` is omitted on purpose: it is the test oracle for ``exact``, agrees
#: with it wherever it runs, and a ninth hue would be a generated one.
FIGURE_SOLVERS = tuple(SOLVER_COLOURS)

INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8a85"
GRID = "#d8d8d2"

VARIANTS = (("static", "static costs C[i,j]"), ("td", "time-slotted costs C[t,i,j]"))


def figure_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "figures"


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _style(ax) -> None:
    ax.grid(alpha=0.35, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def _direct_label(ax, x, y, text: str, colour: str) -> None:
    """Label at the end of a series. Text in ink, identity carried by the mark."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": colour, "lw": 0.9, "alpha": 0.9},
    )


def _series(summary: list[dict[str, str]], solver: str, variant: str, key: str):
    """Points for one solver on one cost shape. Refusals are NOT points.

    A refused run still carries a runtime -- microseconds, because refusing is
    instant -- so plotting every row would draw ``qaoa`` and ``ortools`` as the
    fastest solvers on exactly the instances they could not solve. A miss is a
    gap in the line, which is the honest shape for it.
    """
    rows = [r for r in summary if r["solver"] == solver and r["variant"] == variant]
    rows = [r for r in rows if int(r["feasible_runs"]) > 0]
    rows = [r for r in rows if _number(r[key]) is not None]
    rows.sort(key=lambda r: int(r["n"]))
    return (
        [int(r["n"]) for r in rows],
        [_number(r[key]) for r in rows],
        [_number(r.get("gap_std_pct")) or 0.0 for r in rows],
        rows,
    )


def figure_gap_vs_n(summary: list[dict[str, str]], out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, (variant, subtitle) in zip(axes, VARIANTS, strict=True):
        tied = []
        for solver in FIGURE_SOLVERS:
            ns, gaps, spread, _ = _series(summary, solver, variant, "gap_mean_pct")
            if not ns:
                continue
            colour = SOLVER_COLOURS[solver]
            ax.errorbar(
                ns,
                gaps,
                yerr=spread,
                color=colour,
                marker=SOLVER_MARKERS[solver],
                markersize=6,
                linewidth=2,
                capsize=3,
                elinewidth=1,
                label=solver,
            )
            # Direct-label only the series that separate from the reference.
            # Five solvers sit on exactly 0.00 here; stacking five labels on one
            # point hides the two that actually have something to say.
            if max(gaps) > 0.1:
                _direct_label(ax, ns[-1], gaps[-1], solver, colour)
            else:
                tied.append(solver)
        if tied:
            ax.annotate(
                "+0.00% at every N (tie the reference): " + ", ".join(tied),
                xy=(0.5, -0.155),
                xycoords="axes fraction",
                fontsize=8,
                color=INK_SOFT,
                ha="center",
            )

        # Beyond N=18 Held-Karp cannot run, so the reference is the best result
        # anything in the run achieved. That is not an optimality gap, and the
        # figure has to say so where a reader is looking.
        best_known = sorted({int(r["n"]) for r in summary if r["reference_kind"] == "best-known"})
        if best_known:
            ax.axvspan(min(best_known) - 0.6, max(best_known) + 0.6, color="#f3f1ea", zorder=0)
            ax.annotate(
                f"N={min(best_known)}: no exact oracle,\ngap is vs best-known",
                xy=(0.975, 0.80),
                xycoords="axes fraction",
                fontsize=8,
                color=INK_SOFT,
                ha="right",
            )
        _style(ax)
        # Linear would put 121% and 2.8% on the same axis and make the second
        # invisible -- and the 2-5% band is where every classical solver lives.
        # symlog keeps the exact zeros (which log cannot) and still separates
        # single digits from three.
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100])
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(-0.15, 220)
        ax.set_xlabel("N (objects)", fontsize=10, color=INK_SOFT)
        ax.set_title(subtitle, fontsize=10, color=INK)
        ax.margins(x=0.18)

    axes[0].set_ylabel("gap above the reference (%, symlog below 1)", fontsize=10, color=INK_SOFT)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncols=2)
    fig.suptitle(
        "Solution quality: mean gap over seeds, error bars are 1 s.d. across seeds",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out / "bench_gap_vs_n.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def figure_runtime_vs_n(summary: list[dict[str, str]], out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, (variant, subtitle) in zip(axes, VARIANTS, strict=True):
        ends = []
        for solver in FIGURE_SOLVERS:
            ns, runtimes, _, rows = _series(summary, solver, variant, "runtime_mean_s")
            if not ns:
                continue
            colour = SOLVER_COLOURS[solver]
            spread = [_number(r["runtime_std_s"]) or 0.0 for r in rows]
            ax.errorbar(
                ns,
                runtimes,
                yerr=spread,
                color=colour,
                marker=SOLVER_MARKERS[solver],
                markersize=6,
                linewidth=2,
                capsize=3,
                elinewidth=1,
                label=solver,
            )
            ends.append((runtimes[-1], ns[-1], solver, colour))

        # Runtimes here span seven decades, so several series finish within a
        # label-height of each other. Push the labels apart in log space; the
        # markers stay where the data is.
        ends.sort(reverse=True)
        previous = None
        for value, x_end, solver, colour in ends:
            label_y = value if previous is None else min(value, previous / 2.2)
            _direct_label(ax, x_end, label_y, solver, colour)
            previous = label_y
        _style(ax)
        ax.set_yscale("log")
        ax.set_xlabel("N (objects)", fontsize=10, color=INK_SOFT)
        ax.set_title(subtitle, fontsize=10, color=INK)
        ax.margins(x=0.18)

    axes[0].set_ylabel("time to solution (s, log scale)", fontsize=10, color=INK_SOFT)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncols=2)
    fig.suptitle(
        "Time to solution. A line stops where the solver refused the instance", fontsize=12
    )
    # The caveats without which any two of these lines are incomparable.
    fig.text(
        0.5,
        0.055,
        "qaoa is CLASSICAL STATEVECTOR SIMULATION TIME, not quantum runtime, and it only "
        "reaches N=4.\n"
        "ortools and cpsat burn a configured time limit: a knob, not a measurement.  "
        "sa-perm is numpy against sa-qubo's compiled C++ at the same proposal count.\n"
        "Laptop wall clock, so treat these as orders of magnitude rather than as "
        "a ranking of implementations.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK_SOFT,
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = out / "bench_runtime_vs_n.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def figure_penalty(sweep: list[dict[str, str]], out: Path) -> Path | None:
    """Feasibility and quality against penalty weight, one line per instance.

    Two panels rather than two y-axes on one: a dual-axis chart invites the
    reader to see a crossing point that is an artefact of the scaling.
    """
    import matplotlib.pyplot as plt

    rows = [r for r in sweep if _number(r.get("penalty_factor")) is not None]
    if not rows:
        return None

    instances = sorted({r["instance"] for r in rows}, key=lambda s: int(s.split("_")[0][1:]))
    hues = list(SOLVER_COLOURS.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for index, label in enumerate(instances):
        colour = hues[index % len(hues)]
        subset = [r for r in rows if r["instance"] == label]
        factors = sorted({_number(r["penalty_factor"]) for r in subset})
        for (
            ax,
            key,
        ) in ((axes[0], "feasibility_rate"), (axes[1], "gap_pct")):
            means, spread = [], []
            for factor in factors:
                values = [
                    _number(r[key])
                    for r in subset
                    if _number(r["penalty_factor"]) == factor and _number(r[key]) is not None
                ]
                means.append(float(np.mean(values)) if values else np.nan)
                spread.append(float(np.std(values)) if values else 0.0)
            ax.errorbar(
                factors,
                means,
                yerr=spread,
                color=colour,
                marker="o",
                markersize=5,
                linewidth=2,
                capsize=3,
                elinewidth=1,
                label=label,
            )

    for ax, title, ylabel in (
        (axes[0], "raw feasibility of the samples", "fraction of reads that were permutations"),
        (axes[1], "quality of the best repaired read", "gap above the reference (%)"),
    ):
        _style(ax)
        ax.set_xscale("log")
        # Label the factors that were actually swept, not decades: the whole
        # sweep lives between 0.5 and 8, where "2 x 10^0" says nothing.
        all_factors = sorted({_number(r["penalty_factor"]) for r in rows})
        ax.set_xticks(all_factors)
        ax.set_xticklabels([f"{x:g}" for x in all_factors], fontsize=8)
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("penalty weight / path upper bound", fontsize=10, color=INK_SOFT)
        ax.set_ylabel(ylabel, fontsize=10, color=INK_SOFT)
        ax.set_title(title, fontsize=10, color=INK)
        # Left of 1.0 the bound is not provably sufficient; 1.1 is the shipped
        # default. Both are claims the sweep exists to test.
        ax.axvline(1.0, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.axvline(1.1, color=INK_MUTED, linestyle=":", linewidth=1)
        ax.annotate(
            "penalty provably\nsufficient from here ->",
            xy=(0.30, 0.94),
            xycoords="axes fraction",
            fontsize=7.5,
            color=INK_SOFT,
            ha="right",
            va="top",
        )
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, title="instance")
    fig.suptitle(
        "QUBO penalty weight: what a bigger penalty actually buys "
        "(time-dependent instances, mean over seeds)",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    path = out / "qubo_penalty_sweep.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def figure_best_sequence(
    results: list[dict[str, str]], label: str, snapshot_name: str, out: Path
) -> Path | None:
    """The best-known route for one realistic instance, on RAAN vs altitude.

    Drawn on the plane that costs delta-v rather than on altitude alone: 1 deg
    of RAAN is about 0.131 km/s here against 0.053 km/s per 100 km of altitude,
    so a route that looks like a detour vertically may be the cheap one.
    """
    import matplotlib.pyplot as plt

    from dextrivia.costs.realistic import mean_elements
    from dextrivia.propagation import R_EARTH_WGS72_KM
    from dextrivia.snapshots import Snapshot, default_snapshot_dir

    candidates = [
        r
        for r in results
        if r["instance"] == label and r["feasible"] == "True" and _number(r["total_dv_kms"])
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: _number(r["total_dv_kms"]))
    order = [int(i) for i in best["sequence_norad"].split()]

    snapshot = Snapshot.load(default_snapshot_dir() / snapshot_name)
    epoch = snapshot.median_epoch()
    elements = {o.norad_id: mean_elements(o.line1, o.line2, epoch) for o in snapshot.objects}
    altitude = {k: e.a_km - R_EARTH_WGS72_KM for k, e in elements.items()}
    raan = {k: np.degrees(e.raan_rad) % 360.0 for k, e in elements.items()}

    fig, (context, zoom) = plt.subplots(
        1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    all_raan = [raan[k] for k in elements]
    all_alt = [altitude[k] for k in elements]
    xs = [raan[k] for k in order]
    ys = [altitude[k] for k in order]

    # Left: where in the cloud this cluster is. plane-cluster selection keeps
    # the objects inside a ~10 deg window of node, so on the full 360 deg axis
    # the whole mission collapses to a line -- which is itself the point.
    context.scatter(all_raan, all_alt, s=20, c="#e6e5df", edgecolors=GRID, linewidths=0.5)
    pad_x, pad_y = 12.0, 45.0
    context.add_patch(
        plt.Rectangle(
            (min(xs) - pad_x, min(ys) - pad_y),
            (max(xs) - min(xs)) + 2 * pad_x,
            (max(ys) - min(ys)) + 2 * pad_y,
            fill=False,
            edgecolor=SOLVER_COLOURS["exact"],
            linewidth=1.6,
        )
    )
    context.annotate(
        "the instance,\nshown right",
        xy=(min(xs) - pad_x, max(ys) + pad_y),
        xytext=(-6, 14),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SOFT,
        ha="right",
    )
    _style(context)
    context.set_xlim(0, 360)
    context.set_xticks(range(0, 361, 60))
    context.set_xlabel("RAAN at epoch (deg)", fontsize=10, color=INK_SOFT)
    context.set_ylabel("mean altitude (km)", fontsize=10, color=INK_SOFT)
    context.set_title(f"all {len(snapshot)} catalogued objects", fontsize=10, color=INK)

    # Right: the route itself.
    zoom.scatter(all_raan, all_alt, s=26, c="#e6e5df", edgecolors=GRID, linewidths=0.5)
    for step, (x0, y0, x1, y1) in enumerate(zip(xs, ys, xs[1:], ys[1:], strict=False)):
        zoom.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": SOLVER_COLOURS["exact"],
                "lw": 1.7,
                "alpha": 0.9,
                "shrinkA": 8,
                "shrinkB": 8,
                "connectionstyle": "arc3,rad=0.12",
            },
            zorder=2,
        )
        zoom.annotate(
            f"{step + 1}",
            xy=((x0 + x1) / 2, (y0 + y1) / 2),
            fontsize=7,
            color=INK_SOFT,
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.1", "fc": "white", "ec": GRID, "lw": 0.6},
            zorder=3,
        )

    zoom.scatter(xs, ys, s=64, facecolors="white", edgecolors=INK, linewidths=1.3, zorder=4)
    zoom.scatter(
        xs[0],
        ys[0],
        s=170,
        marker="*",
        color="#008300",
        edgecolors=INK,
        linewidths=0.8,
        zorder=5,
        label="start",
    )
    zoom.scatter(
        xs[-1],
        ys[-1],
        s=110,
        marker="X",
        color="#e34948",
        edgecolors=INK,
        linewidths=0.8,
        zorder=5,
        label="end",
    )
    # Cycle the label side so ten NORAD ids inside a 10 deg window stay legible.
    # Two directions is not enough here -- the cluster puts several objects
    # within a label-width of each other.
    offsets = (
        (13, 0, "left", "center"),
        (0, 12, "center", "bottom"),
        (-13, 0, "right", "center"),
        (0, -13, "center", "top"),
    )
    for index, (norad, x, y) in enumerate(zip(order, xs, ys, strict=True)):
        dx, dy, ha, va = offsets[index % len(offsets)]
        zoom.annotate(
            str(norad),
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_SOFT,
            ha=ha,
            va=va,
            zorder=6,
        )

    _style(zoom)
    zoom.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    zoom.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    zoom.set_xlabel("RAAN at epoch (deg)", fontsize=10, color=INK_SOFT)
    zoom.set_ylabel("mean altitude (km)", fontsize=10, color=INK_SOFT)
    zoom.legend(loc="lower right", fontsize=8, frameon=False)
    kind = best["reference_kind"]
    provenance = "matches the Held-Karp optimum" if kind == "exact" else "no exact oracle at this N"
    zoom.set_title(f"the route, in visiting order ({provenance})", fontsize=10, color=INK)

    fig.suptitle(
        f"{label}: best-known route, {float(best['total_dv_kms']):.4f} km/s over "
        f"{len(order) - 1} legs, found by {best['solver']}",
        fontsize=12,
        color=INK,
    )
    # The cloud is drawn at the epoch; the mission is not flown at the epoch.
    fig.text(
        0.5,
        0.02,
        "Nodes are plotted at the snapshot epoch, but leg k departs 30 days after leg k-1. "
        "Over the 270 days of this mission\nevery node regresses about -0.45 deg/day -- near "
        "enough common-mode that the picture stays fair, which is itself a finding.",
        ha="center",
        fontsize=8,
        color=INK_SOFT,
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    path = out / "best_sequence_raan_altitude.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="a directory written by `dextrivia bench`")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: docs/figures")
    parser.add_argument(
        "--sequence-instance",
        default="n10_td30d",
        help="which instance the route figure draws (default n10_td30d)",
    )
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")  # scripted, headless, deterministic

    results_dir = Path(args.results)
    summary = read_rows(results_dir / "summary.csv")
    results = read_rows(results_dir / "results.csv")
    sweep = read_rows(results_dir / "penalty_sweep.csv")
    manifest = json.loads((results_dir / "manifest.json").read_text())
    snapshot_name = (manifest.get("snapshots") or ["iridium33_20260402.json"])[0]

    out = args.out_dir or figure_dir()
    out.mkdir(parents=True, exist_ok=True)

    written = [
        figure_gap_vs_n(summary, out),
        figure_runtime_vs_n(summary, out),
        figure_penalty(sweep, out),
        figure_best_sequence(results, args.sequence_instance, snapshot_name, out),
    ]
    for path in written:
        print(f"wrote {path}" if path else "skipped a figure: no rows for it in this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
