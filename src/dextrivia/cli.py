"""Command line interface: ``dextrivia fetch | build | solve``.

argparse rather than typer: it is stdlib, and this CLI is three verbs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dextrivia.core import ProblemInstance
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


def _cmd_build(args: argparse.Namespace) -> int:
    snapshot_path = Path(args.snapshot) if args.snapshot else latest_snapshot()
    snapshot = Snapshot.load(snapshot_path)
    epoch = datetime.fromisoformat(args.epoch) if args.epoch else None
    instance = build_instance(snapshot, n=args.n, rule=args.select, seed=args.seed, epoch=epoch)

    if args.out:
        out = Path(args.out)
    else:
        suffix = f"_seed{args.seed}" if args.seed is not None else ""
        out = default_instance_dir() / (f"{snapshot_path.stem}_n{args.n}_{args.select}{suffix}.npz")
    out = instance.save(out)

    print(f"wrote {out}")
    print(f"  snapshot   {snapshot_path.name} ({len(snapshot)} objects)")
    print(f"  selection  {args.select} n={args.n} seed={args.seed}")
    print(f"  epoch      {instance.epoch.isoformat()} ({instance.metadata['epoch_source']})")
    print(f"  cost model {instance.metadata['cost_model']}, costs {instance.costs.shape}")
    print(f"  norad ids  {', '.join(str(i) for i in instance.norad_ids)}")
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
    build.add_argument("--select", choices=SELECTION_RULES, default="first")
    build.add_argument("--seed", type=int, default=None, help="required for --select random")
    build.add_argument("--epoch", default=None, help="ISO UTC; default snapshot median TLE epoch")
    build.add_argument("--out", default=None, help="output .npz path")
    build.set_defaults(func=_cmd_build)

    solve = sub.add_parser("solve", help="solve a built instance")
    solve.add_argument("--instance", required=True, help="path to a built .npz instance")
    solve.add_argument("--solver", choices=sorted(SOLVERS), default="greedy")
    solve.add_argument("--seed", type=int, default=None)
    solve.set_defaults(func=_cmd_solve)

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
