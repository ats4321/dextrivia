"""Re-measure the whole planecluster-v1 family with every solver, many seeds.

    uv run python scripts/benchmark_family.py                    # full run
    uv run python scripts/benchmark_family.py --seeds 3 --quick  # smoke run

Reports MEANS AND SPREADS, never single runs. A stochastic solver quoted from
one seed is an anecdote; the whole point of this script is that the headline
numbers in CLAUDE.md and docs/qubo.md stop being anecdotes.

Deterministic solvers are run once and labelled, rather than averaged over
seeds they ignore -- reporting a standard deviation of 0.000 over five
identical runs would imply a precision that was never measured.

The reference is Held-Karp where it reaches (N <= 18) and CP-SAT above that.
CP-SAT is the only solver here that can say whether its answer is PROVEN
optimal, so the table distinguishes "matched the reference" from "proved it".
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from dextrivia.core import ProblemInstance, Solution
from dextrivia.snapshots import default_snapshot_dir
from dextrivia.solvers import (
    CPSATSolver,
    ExactSolver,
    GreedySolver,
    LocalSearchSolver,
    ORToolsRoutingSolver,
    PermutationAnnealingSolver,
    QAOASolver,
)
from dextrivia.solvers.quantum_annealing import SimulatedAnnealingSolver

FAMILY_SIZES = (4, 5, 8, 10, 15, 20)
VARIANTS = ("static", "td30d")

#: Deterministic solvers: same input, same output. Run once, labelled as such.
DETERMINISTIC = {"greedy", "exact", "localsearch", "ortools", "cpsat"}


def instance_path(n: int, variant: str) -> Path:
    return (
        default_snapshot_dir().parent
        / "instances"
        / f"iridium33_20260402_planecluster-v1_n{n}_{variant}.npz"
    )


def solver_bank(budget_s: float, mip_s: float) -> dict[str, object]:
    """One place where every solver's budget is set, so they stay comparable."""
    return {
        "greedy": GreedySolver(),
        "localsearch": LocalSearchSolver(),
        "exact": ExactSolver(),
        "cpsat": CPSATSolver(time_limit_s=mip_s),
        "ortools": ORToolsRoutingSolver(time_limit_s=budget_s),
        # Wall-clock matched against sa-qubo below; see the module docstring of
        # dextrivia.solvers.permutation for why wall clock and not iterations.
        "sa-perm": PermutationAnnealingSolver(time_limit_s=budget_s, restarts=4),
        "sa-qubo": SimulatedAnnealingSolver(num_reads=1000, num_sweeps=2000),
        "qaoa": QAOASolver(),
    }


def summarise(name: str, runs: list[Solution], reference: float | None) -> dict[str, object]:
    feasible = [r for r in runs if r.feasible]
    if not feasible:
        return {
            "solver": name,
            "feasible": False,
            "reason": runs[0].metadata.get("reason", "unknown"),
            "runs": len(runs),
        }

    costs = [r.total_dv_kms for r in feasible]
    runtimes = [r.runtime_s for r in feasible]
    record: dict[str, object] = {
        "solver": name,
        "feasible": True,
        "runs": len(feasible),
        "deterministic": name in DETERMINISTIC,
        "best_dv_kms": min(costs),
        "mean_dv_kms": statistics.fmean(costs),
        "worst_dv_kms": max(costs),
        "stdev_dv_kms": statistics.stdev(costs) if len(costs) > 1 else None,
        "mean_runtime_s": statistics.fmean(runtimes),
    }
    if reference:
        record["mean_gap_pct"] = 100.0 * (record["mean_dv_kms"] / reference - 1.0)
        record["best_gap_pct"] = 100.0 * (record["best_dv_kms"] / reference - 1.0)
        record["worst_gap_pct"] = 100.0 * (record["worst_dv_kms"] / reference - 1.0)
    if name == "cpsat":
        record["proven_optimal"] = feasible[0].metadata.get("proven_optimal")
        record["lower_bound_kms"] = feasible[0].metadata.get("lower_bound_kms")
    if name in {"sa-qubo", "qaoa"}:
        record["feasibility_rate"] = statistics.fmean(
            [r.metadata.get("feasibility_rate", 0.0) for r in feasible]
        )
    if name == "sa-perm":
        record["mean_iterations"] = statistics.fmean([r.metadata["iterations"] for r in feasible])
    return record


def reference_for(instance: ProblemInstance, mip_s: float) -> dict[str, object]:
    """Held-Karp where it reaches, else CP-SAT. Says which, and whether proved."""
    exact = ExactSolver().solve(instance)
    if exact.feasible:
        return {
            "value": exact.total_dv_kms,
            "source": "held-karp",
            "proven": True,
            "lower_bound": exact.total_dv_kms,
        }
    mip = CPSATSolver(time_limit_s=mip_s).solve(instance)
    if mip.feasible:
        return {
            "value": mip.total_dv_kms,
            "source": "cpsat",
            "proven": bool(mip.metadata["proven_optimal"]),
            "lower_bound": mip.metadata["lower_bound_kms"],
        }
    return {"value": None, "source": "none", "proven": False, "lower_bound": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=2.0, help="wall-clock seconds per run")
    parser.add_argument("--mip-seconds", type=float, default=120.0)
    parser.add_argument("--quick", action="store_true", help="N<=10 only, skip qaoa")
    parser.add_argument("--out", type=Path, default=Path("docs/data/benchmark_family.json"))
    args = parser.parse_args()

    sizes = tuple(n for n in FAMILY_SIZES if n <= 10) if args.quick else FAMILY_SIZES
    seeds = list(range(args.seeds))
    results = []
    started = time.time()

    for variant in VARIANTS:
        for n in sizes:
            path = instance_path(n, variant)
            if not path.exists():
                print(f"skip {path.name}: not built")
                continue
            instance = ProblemInstance.load(path)
            reference = reference_for(instance, args.mip_seconds)
            print(
                f"\n=== {path.name}  N={n}  td={instance.time_dependent} "
                f"ref={reference['source']} "
                f"{'' if reference['value'] is None else round(reference['value'], 4)} "
                f"proven={reference['proven']} ===",
                flush=True,
            )

            bank = solver_bank(args.budget, args.mip_seconds)
            if args.quick:
                bank.pop("qaoa", None)

            rows = []
            for name, solver in bank.items():
                run_seeds = [seeds[0]] if name in DETERMINISTIC else seeds
                runs = [solver.solve(instance, seed=s) for s in run_seeds]
                row = summarise(name, runs, reference["value"])
                rows.append(row)
                if row["feasible"]:
                    gap = row.get("mean_gap_pct")
                    spread = row.get("stdev_dv_kms")
                    print(
                        f"  {name:<12} mean {row['mean_dv_kms']:.4f} km/s"
                        + (f"  gap {gap:+.2f}%" if gap is not None else "")
                        + (f"  sd {spread:.4f}" if spread else "")
                        + f"  {row['mean_runtime_s'] * 1e3:.0f} ms"
                        + ("  [det]" if row["deterministic"] else f"  [{len(run_seeds)} seeds]"),
                        flush=True,
                    )
                else:
                    print(f"  {name:<12} REFUSED: {row['reason'][:70]}", flush=True)

            results.append(
                {
                    "instance": path.name,
                    "n": n,
                    "variant": variant,
                    "time_dependent": instance.time_dependent,
                    "reference": reference,
                    "solvers": rows,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "budget_s": args.budget,
                "mip_seconds": args.mip_seconds,
                "elapsed_s": time.time() - started,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}  ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
