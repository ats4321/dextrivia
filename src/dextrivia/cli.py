"""Command line interface: ``dextrivia fetch | build | solve | bench``.

argparse rather than typer: it is stdlib, and this CLI is four verbs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dextrivia import bench as bench_module
from dextrivia.core import ProblemInstance
from dextrivia.costs import COST_MODELS, ClusterWindow, HohmannCostModel
from dextrivia.costs.selection import PLANE_CLUSTER_RULE, build_cluster_instance, family_path
from dextrivia.instances import build_instance
from dextrivia.snapshots import (
    DEFAULT_GROUP,
    SELECTION_RULES,
    Snapshot,
    default_snapshot_dir,
    fetch_snapshot,
    group_stem,
    parse_snapshot_name,
)
from dextrivia.solvers import SOLVERS


def default_instance_dir() -> Path:
    return default_snapshot_dir().parent / "instances"


def latest_snapshot(group: str | None = DEFAULT_GROUP, snapshot_dir: Path | None = None) -> Path:
    """Newest snapshot of ``group`` (all groups if ``group`` is None).

    Ordered by the date encoded in the filename, not by alphabetical order and
    not by ``fetched_utc`` -- the committed legacy snapshot has a null
    ``fetched_utc``, and a group whose stem sorts early (``cosmos2251``) must
    not out-rank a newer one that sorts late (``iridium33``).

    Same-date ties go to the larger suffix: ``_new_snapshot_path`` writes the
    bare ``<stem>_<date>.json`` first and only then falls back to the
    ``<stem>_<date>T<time>Z.json`` form, so the suffixed file is the later one.
    """
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else default_snapshot_dir()
    wanted = group_stem(group) if group else None

    candidates = []
    for path in snapshot_dir.glob("*.json"):
        parsed = parse_snapshot_name(path)
        if parsed is None:
            continue
        stem, date, suffix = parsed
        if wanted is None or stem == wanted:
            candidates.append(((date, suffix, path.name), path))

    if not candidates:
        scope = f" for group {group!r}" if group else ""
        raise SystemExit(f"no snapshots{scope} in {snapshot_dir}; run `dextrivia fetch` first")
    return max(candidates)[1]


def _cmd_fetch(args: argparse.Namespace) -> int:
    path = fetch_snapshot(group=args.group, out_dir=args.out_dir)
    snapshot = Snapshot.load(path)
    print(f"wrote {path}")
    print(f"  objects       {len(snapshot)}")
    fetched = snapshot.fetched_utc.isoformat() if snapshot.fetched_utc else "not recorded"
    print(f"  fetched (UTC) {fetched}")
    print(f"  median epoch  {snapshot.median_epoch().isoformat()}")
    return 0


def _cost_model(args: argparse.Namespace):
    if args.cost_model == HohmannCostModel.name:
        if args.delta_days is not None:
            raise ValueError("--delta-days needs a plane-aware cost model; hohmann is static")
        return HohmannCostModel()
    return COST_MODELS[args.cost_model](delta_per_leg_days=args.delta_days)


def _default_out(args: argparse.Namespace, snapshot_path: Path, cost_model) -> Path:
    if args.select == PLANE_CLUSTER_RULE and args.family_version:
        if args.cost_model != "impulsive-plane":
            raise ValueError("--family-version names the impulsive family; pass --out instead")
        return family_path(
            default_instance_dir(), snapshot_path.stem, args.n, args.delta_days, args.family_version
        )
    if args.select == PLANE_CLUSTER_RULE:
        name = f"{snapshot_path.stem}_n{args.n}_{PLANE_CLUSTER_RULE}_{cost_model.name}.npz"
        return default_instance_dir() / name
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    return default_instance_dir() / f"{snapshot_path.stem}_n{args.n}_{args.select}{suffix}.npz"


def _cmd_build(args: argparse.Namespace) -> int:
    snapshot_path = Path(args.snapshot) if args.snapshot else latest_snapshot()
    snapshot = Snapshot.load(snapshot_path)
    epoch = datetime.fromisoformat(args.epoch) if args.epoch else None
    if epoch is not None and epoch.tzinfo is None:
        raise ValueError("--epoch must carry a UTC offset, e.g. 2026-04-02T07:05:22+00:00")
    cost_model = _cost_model(args)

    if args.select == PLANE_CLUSTER_RULE:
        window = ClusterWindow(
            raan_window_deg=args.raan_window,
            inc_window_deg=args.inc_window,
            alt_band_km=args.alt_band,
        )
        instance = build_cluster_instance(
            snapshot,
            args.n,
            cost_model,
            epoch=epoch,
            window=window,
            seed_norad=args.seed_norad,
            family_version=args.family_version,
        )
    else:
        if args.delta_days is not None or args.cost_model != HohmannCostModel.name:
            raise ValueError(f"--select {args.select} builds hohmann costs only; use plane-cluster")
        instance = build_instance(snapshot, n=args.n, rule=args.select, seed=args.seed, epoch=epoch)

    out = Path(args.out) if args.out else _default_out(args, snapshot_path, cost_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out = instance.save(out)

    print(f"wrote {out}")
    print(f"  snapshot   {snapshot_path.name} ({len(snapshot)} objects)")
    if args.select == PLANE_CLUSTER_RULE:
        meta = instance.metadata
        print(
            f"  selection  {args.select} n={args.n} seed_norad={meta['cluster_seed_norad']} "
            f"raan<={meta['raan_window_deg']:g} inc<={meta['inc_window_deg']:g} "
            f"alt<={meta['alt_band_km']:g}km"
        )
    else:
        print(f"  selection  {args.select} n={args.n} seed={args.seed}")
    print(f"  epoch      {instance.epoch.isoformat()} ({instance.metadata['epoch_source']})")
    print(f"  cost model {instance.metadata['cost_model']}, costs {instance.costs.shape}")
    print(f"  norad ids  {', '.join(str(i) for i in instance.norad_ids)}")
    meta = instance.metadata
    for record in meta.get("excluded_decayed", ()):
        norad, error, at = record["norad_id"], record["sgp4_error"], record["at"]
        print(f"  excluded   {norad} (SGP4 error {error} at {at})")
    status = "EXCEEDS" if meta["exceeds_validity_horizon"] else "within"
    print(
        f"  horizon    propagates {meta['propagation_span_days']:.1f} d from TLE epochs, "
        f"{status} the {meta['validity_horizon_days']:g}-day validity horizon"
    )
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    instance = ProblemInstance.load(args.instance)
    solution = SOLVERS[args.solver]().solve(instance, seed=args.seed)

    print(f"solver     {solution.solver_name}")
    print(f"instance   {args.instance} (N={instance.n}, {instance.metadata.get('cost_model')})")
    print(f"epoch      {instance.epoch.isoformat()}")
    if not solution.feasible:
        print(f"INFEASIBLE {solution.metadata.get('reason')}")
        return 1
    print(f"total dv   {solution.total_dv_kms:.6f} km/s")
    print(f"runtime    {solution.runtime_s * 1e3:.1f} ms")
    print("sequence   (NORAD id, leg delta-v km/s)")
    for step, index in enumerate(solution.sequence):
        if step == 0:
            print(f"  {step + 1:2d}. {instance.norad_ids[index]:>6}   start")
        else:
            leg = instance.leg_cost(step - 1, solution.sequence[step - 1], index)
            print(f"  {step + 1:2d}. {instance.norad_ids[index]:>6}   +{leg:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dextrivia", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download a new TLE snapshot (never overwrites)")
    fetch.add_argument("--group", default=DEFAULT_GROUP, help="Celestrak GROUP name")
    fetch.add_argument("--out-dir", type=Path, default=None, help="default: data/snapshots")
    fetch.set_defaults(func=_cmd_fetch)

    build = sub.add_parser("build", help="build a problem instance from a snapshot")
    build.add_argument(
        "--snapshot",
        default=None,
        help=f"default: newest {DEFAULT_GROUP} snapshot in data/snapshots",
    )
    build.add_argument("--n", type=int, default=10, help="number of objects (default 10)")
    build.add_argument("--select", choices=(*SELECTION_RULES, PLANE_CLUSTER_RULE), default="first")
    build.add_argument("--seed", type=int, default=None, help="required for --select random")
    build.add_argument("--epoch", default=None, help="ISO UTC; default snapshot median TLE epoch")
    build.add_argument("--out", default=None, help="output .npz path")
    build.add_argument(
        "--cost-model",
        choices=sorted(COST_MODELS),
        default=HohmannCostModel.name,
        help="default hohmann; impulsive-plane and edelbaum need --select plane-cluster",
    )
    build.add_argument(
        "--delta-days",
        type=float,
        default=None,
        help="per-leg duration; makes costs time-slotted C[t,i,j] (default: static)",
    )
    defaults = ClusterWindow()
    cluster = build.add_argument_group("plane-cluster window (docs/physics.md section 6)")
    cluster.add_argument("--raan-window", type=float, default=defaults.raan_window_deg)
    cluster.add_argument("--inc-window", type=float, default=defaults.inc_window_deg)
    cluster.add_argument("--alt-band", type=float, default=defaults.alt_band_km)
    cluster.add_argument("--seed-norad", type=int, default=None, help="pin the cluster seed")
    cluster.add_argument(
        "--family-version",
        default=None,
        help="record a family version and use the family filename, e.g. v2",
    )
    build.set_defaults(func=_cmd_build)

    solve = sub.add_parser("solve", help="solve a built instance")
    solve.add_argument("--instance", required=True, help="path to a built .npz instance")
    solve.add_argument("--solver", choices=sorted(SOLVERS), default="greedy")
    solve.add_argument("--seed", type=int, default=None)
    solve.set_defaults(func=_cmd_solve)

    bench = sub.add_parser("bench", help="run every solver over the instance family")
    bench_module.add_arguments(bench)
    bench.set_defaults(func=bench_module.main)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
