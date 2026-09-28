"""Generate a versioned plane-cluster instance family.

    uv run python scripts/build_instance_family.py                 # v2: build + report
    uv run python scripts/build_instance_family.py --version v1    # rebuild v1 (costs to ~1e-12)
    uv run python scripts/build_instance_family.py --check         # report on files on disk

File names::

    <snapshot-stem>_planecluster-<version>_n<N>_<static|td<delta>d>.npz

v1 lives flat in ``data/instances/`` (the README's canonical run cites it); v2
lives in ``data/instances/planecluster-v2/`` so ``dextrivia bench`` does not mix
the two. Every file is also reproducible from the CLI -- the command is printed
next to each one.

v1 (frozen): ``iridium33_20260402``, N in {4,5,8,10,15,20}, static and 30 d
per leg. Four of its time-dependent instances propagate beyond the validity
horizon; they stay committed unchanged for the README's reproducibility.

v2: the fresh snapshot, and every time-dependent instance inside the
``VALIDITY_HORIZON_DAYS`` (see ``docs/physics.md`` section 9-10):

* Per-leg durations {3, 7, 14, 30} days, each kept for an N only if its
  propagation span stays inside the horizon. 7 d covers one full phasing cycle
  at a 50 km altitude difference (~6.5 d, section 10); 3 d is an aggressive
  schedule, 14 and 30 d relaxed ones.
* N in {25, 30, 40} use the smallest RAAN window in {15, 20, 25, 30} deg in
  which the densest seed holds 40 objects -- chosen by object COUNT, never by
  delta-v or by which solver does well.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dextrivia.core import ProblemInstance
from dextrivia.costs.realistic import ImpulsiveCostModel, mean_elements
from dextrivia.costs.selection import (
    ClusterWindow,
    build_cluster_instance,
    family_path,
    select_plane_cluster,
)
from dextrivia.costs.validity import VALIDITY_HORIZON_DAYS, PropagationHorizonWarning
from dextrivia.snapshots import Snapshot, default_snapshot_dir
from dextrivia.solvers import ExactSolver, GreedySolver, LocalSearchSolver

LARGE_WINDOW_CANDIDATES_DEG = (15.0, 20.0, 25.0, 30.0)


@dataclass(frozen=True)
class Family:
    snapshot: str
    sizes: tuple[int, ...]
    deltas: tuple[float | None, ...]
    large_sizes: tuple[int, ...] = ()
    enforce_horizon: bool = True


FAMILIES = {
    "v1": Family(
        snapshot="iridium33_20260402.json",
        sizes=(4, 5, 8, 10, 15, 20),
        deltas=(None, 30.0),
        enforce_horizon=False,  # frozen before the horizon existed
    ),
    "v2": Family(
        snapshot="iridium33_20260928.json",
        sizes=(4, 5, 8, 10, 15, 20, 25, 30, 40),
        deltas=(None, 3.0, 7.0, 14.0, 30.0),
        large_sizes=(25, 30, 40),
    ),
}


def instance_dir() -> Path:
    return default_snapshot_dir().parent / "instances"


def large_window(snapshot: Snapshot, n: int) -> ClusterWindow:
    """Smallest candidate RAAN window whose densest seed holds ``n`` objects."""
    for raan in LARGE_WINDOW_CANDIDATES_DEG:
        window = ClusterWindow(raan_window_deg=raan)
        try:
            select_plane_cluster(snapshot, n, window=window)
        except ValueError:
            continue
        return window
    raise ValueError(f"no candidate window holds {n} objects")


def altitude_order(instance: ProblemInstance, snapshot: Snapshot) -> tuple[int, ...]:
    """Indices sorted by mean altitude -- the optimum under the coplanar model."""
    by_id = {o.norad_id: o for o in snapshot.objects}
    radii = [
        mean_elements(by_id[i].line1, by_id[i].line2, instance.epoch).a_km
        for i in instance.norad_ids
    ]
    return tuple(int(i) for i in np.argsort(radii))


def slot_drift_pct(instance: ProblemInstance) -> float | None:
    """Mean |C[last] - C[0]| over pairs, as % of mean C[0]: how time-dependent it really is."""
    if not instance.time_dependent:
        return None
    first, last = instance.leg_costs(0), instance.leg_costs(instance.n - 2)
    mask = ~np.eye(instance.n, dtype=bool)
    return float(np.abs(last - first)[mask].mean() / first[mask].mean() * 100.0)


def report_row(instance: ProblemInstance, snapshot: Snapshot) -> dict[str, object]:
    greedy = GreedySolver().solve(instance)
    local = LocalSearchSolver().solve(instance)
    exact = ExactSolver().solve(instance)
    meta = instance.metadata
    row: dict[str, object] = {
        "n": instance.n,
        "delta": meta.get("delta_per_leg_days"),
        "raan_window": meta.get("raan_window_deg"),
        "span": meta.get("propagation_span_days"),
        "drift_pct": slot_drift_pct(instance),
        "greedy": greedy.total_dv_kms,
        "local": local.total_dv_kms,
        "altitude": instance.path_cost(altitude_order(instance, snapshot)),
        "exact": exact.total_dv_kms if exact.feasible else None,
    }
    reference = row["exact"] if exact.feasible else min(greedy.total_dv_kms, local.total_dv_kms)
    row["reference_kind"] = "exact" if exact.feasible else "best-known"
    for key in ("greedy", "local", "altitude"):
        gap = (row[key] - reference) / reference * 100.0
        row[f"{key}_gap"] = 0.0 if abs(gap) < 1e-9 else gap  # float summation order
    return row


def print_table(rows: list[dict[str, object]]) -> None:
    print()
    print(
        "| N | delta (d) | RAAN window | span (d) | slot drift | reference (km/s) "
        "| greedy | localsearch | altitude order |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        delta = "static" if r["delta"] is None else f"{r['delta']:g}"
        drift = "-" if r["drift_pct"] is None else f"{r['drift_pct']:.2f}%"
        ref = r["exact"] if r["exact"] is not None else min(r["greedy"], r["local"])
        print(
            f"| {r['n']} | {delta} | {r['raan_window']:g} deg | {r['span']:.1f} | {drift} "
            f"| {ref:.4f} *{r['reference_kind']}* | {r['greedy_gap']:+.2f}% "
            f"| {r['local_gap']:+.2f}% | {r['altitude_gap']:+.1f}% |"
        )
    print()


def cli_command(snapshot: str, n: int, delta: float | None, window: ClusterWindow, version: str):
    parts = [
        "uv run dextrivia build",
        f"--snapshot data/snapshots/{snapshot}",
        "--select plane-cluster --cost-model impulsive-plane",
        f"--n {n}",
    ]
    if delta is not None:
        parts.append(f"--delta-days {delta:g}")
    if window != ClusterWindow():
        parts.append(f"--raan-window {window.raan_window_deg:g}")
    parts.append(f"--family-version {version}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v2", choices=sorted(FAMILIES))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report on the instances already on disk instead of rebuilding them",
    )
    args = parser.parse_args(argv)

    family = FAMILIES[args.version]
    snapshot = Snapshot.load(default_snapshot_dir() / family.snapshot)
    stem = Path(family.snapshot).stem
    out_dir = args.out_dir or instance_dir()
    print(f"snapshot  {family.snapshot} ({len(snapshot)} objects)")
    print(f"epoch     {snapshot.median_epoch().isoformat()} (snapshot median TLE epoch)")

    rows, skipped = [], []
    for n in family.sizes:
        window = (
            large_window(snapshot, max(family.large_sizes)) if n in family.large_sizes else None
        )
        for delta in family.deltas:
            path = family_path(out_dir, stem, n, delta, args.version)
            if args.check:
                if not path.exists():
                    continue
                instance = ProblemInstance.load(path)
            elif family.enforce_horizon and (n - 2) * (delta or 0.0) > VALIDITY_HORIZON_DAYS:
                # the last departure alone is past the horizon; TLE age only adds
                skipped.append((n, delta, (n - 2) * delta))
                continue
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", PropagationHorizonWarning)
                    instance = build_cluster_instance(
                        snapshot,
                        n,
                        ImpulsiveCostModel(delta_per_leg_days=delta),
                        window=window,
                        family_version=args.version,
                    )
                if family.enforce_horizon and instance.metadata["exceeds_validity_horizon"]:
                    skipped.append((n, delta, instance.metadata["propagation_span_days"]))
                    continue
                for w in caught:
                    print(f"  warning: {w.message}")
                path.parent.mkdir(parents=True, exist_ok=True)
                instance.save(path)
                print(f"wrote {path.name}  costs {instance.costs.shape}")
                command = cli_command(
                    family.snapshot, n, delta, window or ClusterWindow(), args.version
                )
                print(f"  $ {command}")
            rows.append(report_row(instance, snapshot))

    for n, delta, span in skipped:
        print(f"skipped n={n} delta={delta:g} d: span >= {span:.1f} d exceeds the horizon")
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
