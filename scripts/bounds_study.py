"""Where do the MIP solvers stop being useful? Evidence for CPSAT_MAX_N / HIGHS_MAX_N.

    uv run python scripts/bounds_study.py            # writes docs/data/bounds_study.json
    uv run python scripts/bounds_study.py --ablation # cpsat warm start / circuit on or off

Each (solver, instance) runs in its own process so peak RSS is that run's alone.
Records wall time, model-build overhead (wall minus the solver's own clock, for
cpsat), peak memory, the certified bound and the gap to the best path known.
Instances: v2 n40, v3 n40/n50/n60, static and time-dependent where they exist.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTANCES = ROOT / "data" / "instances"
CASES = [
    "planecluster-v2/iridium33_20260928_planecluster-v2_n40_static.npz",
    "planecluster-v2/iridium33_20260928_planecluster-v2_n40_td3d.npz",
    "collisionpair-v3/iridium33_20260928+cosmos2251_20260930_collisionpair-v3_n40_static.npz",
    "collisionpair-v3/iridium33_20260928+cosmos2251_20260930_collisionpair-v3_n40_td3d.npz",
    "collisionpair-v3/iridium33_20260928+cosmos2251_20260930_collisionpair-v3_n50_static.npz",
    "collisionpair-v3/iridium33_20260928+cosmos2251_20260930_collisionpair-v3_n50_td3d.npz",
    "collisionpair-v3/iridium33_20260928+cosmos2251_20260930_collisionpair-v3_n60_static.npz",
]

CHILD = """
import json, resource, sys, time
from dextrivia.core import ProblemInstance
from dextrivia.solvers import CPSATSolver, HiGHSSolver, IteratedLocalSearchSolver
path, name, limit = sys.argv[1], sys.argv[2], float(sys.argv[3])
options = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
inst = ProblemInstance.load(path)
solver = {"cpsat": CPSATSolver, "highs": HiGHSSolver}[name](
    time_limit_s=limit, max_n=10**6, **options
)
t = time.perf_counter()
s = solver.solve(inst, seed=1)
wall = time.perf_counter() - t
ils = IteratedLocalSearchSolver().solve(inst, seed=1).total_dv_kms
m = s.metadata
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({
    "n": inst.n, "time_dependent": inst.time_dependent, "solver": name,
    "time_limit_s": limit, "wall_s": wall,
    "solver_clock_s": m.get("solver_wall_time_s"),
    "peak_rss_mb": rss / (1 << 20) if sys.platform == "darwin" else rss / 1024,
    "feasible": s.feasible, "dv_kms": s.total_dv_kms if s.feasible else None,
    "lower_bound_kms": m.get("lower_bound_kms"), "proven_optimal": m.get("proven_optimal"),
    "ils_seed1_dv_kms": ils, "reason": m.get("reason"), "options": options,
    "load_average_1min": __import__("os").getloadavg()[0],
}))
"""


#: The CP-SAT ablation: warm start and redundant circuit, each on and off.
ABLATION_CASES = [
    "iridium33_20260402_planecluster-v1_n20_td30d.npz",
    "planecluster-v2/iridium33_20260928_planecluster-v2_n20_td7d.npz",
    "planecluster-v2/iridium33_20260928_planecluster-v2_n30_static.npz",
    "planecluster-v2/iridium33_20260928_planecluster-v2_n40_td3d.npz",
]
ABLATION_OPTIONS = [
    {"warm_start": False, "circuit": False},
    {"warm_start": True, "circuit": False},
    {"warm_start": True, "circuit": True},
]


def main() -> int:
    ablation = "--ablation" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--ablation"]
    limit = float(args[0]) if args else (30.0 if ablation else 60.0)
    plan = (
        [(c, "cpsat", o) for c in ABLATION_CASES for o in ABLATION_OPTIONS]
        if ablation
        else [(c, name, {}) for c in CASES for name in ("cpsat", "highs")]
    )
    rows = []
    for case, name, options in plan:
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD,
                str(INSTANCES / case),
                name,
                str(limit),
                json.dumps(options),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        row = {"instance": Path(case).name, **json.loads(out.stdout.strip().splitlines()[-1])}
        best = min(v for v in (row["dv_kms"], row["ils_seed1_dv_kms"]) if v is not None)
        bound = row["lower_bound_kms"]
        row["gap_to_best_pct"] = (best - bound) / best * 100 if bound is not None else None
        rows.append(row)
        shown = "-" if bound is None else f"{bound:.4f} ({row['gap_to_best_pct']:.2f}%)"
        print(
            f"{row['instance'][-22:]:<22} {name:<6} wall {row['wall_s']:6.1f}s "
            f"rss {row['peak_rss_mb']:7.0f} MB  bound {shown} {options or ''}",
            flush=True,
        )
    target = ROOT / "docs" / "data" / ("cpsat_ablation.json" if ablation else "bounds_study.json")
    target.write_text(json.dumps({"time_limit_s": limit, "runs": rows}, indent=2) + "\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
