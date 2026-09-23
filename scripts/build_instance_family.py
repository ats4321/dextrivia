"""Generate the versioned plane-cluster instance family into ``data/instances/``.

    uv run python scripts/build_instance_family.py            # build + report
    uv run python scripts/build_instance_family.py --check    # rebuild nothing, just report

One file per (N, time-variant) pair, named::

    <snapshot-stem>_planecluster-<version>_n<N>_<static|td<delta>d>.npz

Every file records the snapshot filename, the selection rule and its window,
the epoch, the cost model and -- for the time-slotted ones -- the per-leg
duration and the leg-to-wall-clock mapping. The gap table this prints is the
benchmark headline: greedy against the Held-Karp optimum, plus what a servicer
would pay by simply visiting in altitude order (the strategy that was optimal
under the old coplanar cost model).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dextrivia.core import ProblemInstance
from dextrivia.costs.realistic import ImpulsiveCostModel, mean_elements
from dextrivia.costs.selection import build_cluster_instance
from dextrivia.snapshots import Snapshot, default_snapshot_dir
from dextrivia.solvers import ExactSolver, GreedySolver

#: The snapshot the committed family is defined against. Named, not "newest":
#: a committed benchmark artifact must not change when someone runs `fetch`.
FAMILY_SNAPSHOT = "iridium33_20260402.json"
FAMILY_VERSION = "v1"
FAMILY_SIZES = (4, 5, 8, 10, 15, 20)
DEFAULT_DELTA_DAYS = 30.0


def instance_dir() -> Path:
    return default_snapshot_dir().parent / "instances"


def family_name(snapshot_stem: str, n: int, delta_days: float | None, version: str) -> str:
    variant = "static" if delta_days is None else f"td{delta_days:g}d"
    return f"{snapshot_stem}_planecluster-{version}_n{n}_{variant}.npz"


def altitude_order(instance: ProblemInstance, snapshot: Snapshot) -> tuple[int, ...]:
    """Indices sorted by mean altitude -- the optimum under the coplanar model."""
    by_id = {o.norad_id: o for o in snapshot.objects}
    radii = [
        mean_elements(by_id[i].line1, by_id[i].line2, instance.epoch).a_km
        for i in instance.norad_ids
    ]
    return tuple(int(i) for i in np.argsort(radii))


def report_row(instance: ProblemInstance, snapshot: Snapshot) -> dict[str, object]:
    greedy = GreedySolver().solve(instance)
    exact = ExactSolver().solve(instance)
    alt = instance.path_cost(altitude_order(instance, snapshot))
    row: dict[str, object] = {
        "n": instance.n,
        "variant": "time-dependent" if instance.time_dependent else "static",
        "greedy": greedy.total_dv_kms,
        "altitude": alt,
        "exact": exact.total_dv_kms if exact.feasible else None,
        "exact_note": None if exact.feasible else exact.metadata.get("reason"),
        "runtime_s": exact.runtime_s,
    }
    if exact.feasible:
        row["gap_pct"] = (greedy.total_dv_kms - exact.total_dv_kms) / exact.total_dv_kms * 100.0
        row["alt_excess_pct"] = (alt - exact.total_dv_kms) / exact.total_dv_kms * 100.0
        row["exact_is_altitude_order"] = exact.sequence in (
            altitude_order(instance, snapshot),
            altitude_order(instance, snapshot)[::-1],
        )
    return row


def print_table(rows: list[dict[str, object]]) -> None:
    print()
    print("| N | variant | exact (km/s) | greedy (km/s) | gap | altitude order (km/s) | vs exact |")
    print("|---|---------|--------------|---------------|-----|-----------------------|----------|")
    for r in rows:
        if r["exact"] is None:
            exact_cell, gap_cell, alt_cell = f"_{r['exact_note']}_", "-", "-"
        else:
            exact_cell = f"{r['exact']:.4f}"
            gap_cell = f"{r['gap_pct']:+.2f}%"
            alt_cell = f"{r['alt_excess_pct']:+.1f}%"
        print(
            f"| {r['n']} | {r['variant']} | {exact_cell} | {r['greedy']:.4f} | {gap_cell} "
            f"| {r['altitude']:.4f} | {alt_cell} |"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=FAMILY_SNAPSHOT, help=f"default {FAMILY_SNAPSHOT}")
    parser.add_argument("--version", default=FAMILY_VERSION)
    parser.add_argument("--delta-days", type=float, default=DEFAULT_DELTA_DAYS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report on the instances already on disk instead of rebuilding them",
    )
    args = parser.parse_args(argv)

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        snapshot_path = default_snapshot_dir() / args.snapshot
    snapshot = Snapshot.load(snapshot_path)
    out_dir = args.out_dir or instance_dir()

    print(f"snapshot  {snapshot_path.name} ({len(snapshot)} objects)")
    print(f"epoch     {snapshot.median_epoch().isoformat()} (snapshot median TLE epoch)")
    print(f"delta     {args.delta_days:g} days per leg (transfer + rendezvous + capture)")

    rows = []
    for n in FAMILY_SIZES:
        for delta in (None, args.delta_days):
            path = out_dir / family_name(snapshot_path.stem, n, delta, args.version)
            if args.check:
                instance = ProblemInstance.load(path)
            else:
                instance = build_cluster_instance(
                    snapshot,
                    n,
                    ImpulsiveCostModel(delta_per_leg_days=delta),
                    family_version=args.version,
                )
                instance.save(path)
                print(f"wrote {path.name}  costs {instance.costs.shape}")
            rows.append(report_row(instance, snapshot))

    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
