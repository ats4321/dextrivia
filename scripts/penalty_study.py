"""Sweep the QUBO penalty safety factor across ALL time-dependent instances.

    uv run python scripts/penalty_study.py --seeds 5

DEFAULT_PENALTY_SAFETY was originally calibrated on ONE instance, and a later
spot check on n15_td30d found the shipped value of 1.1 to be the worst of four
factors there. Neither observation is grounds for changing a default: one
instance is not a study, and the ordering on that instance was not monotone,
which is what sampler noise looks like.

So this script exists to answer one question with enough data to act on it:
**does any single safety factor win consistently across the whole
time-dependent family?** If yes, change the default. If no, keep it and
document the variation. Tuning on whichever instance is in front of you is how
a benchmark ends up measuring its own tuning.

Every factor above 1.0 is provably sufficient (see docs/qubo.md section 3);
factors at or below 1.0 are swept too, because the bound is worst-case and the
interesting question is how much the guarantee actually costs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from dextrivia.core import ProblemInstance
from dextrivia.qubo import build_qubo, path_upper_bound, summarize_samples
from dextrivia.snapshots import default_snapshot_dir
from dextrivia.solvers import CPSATSolver, ExactSolver
from dextrivia.solvers.quantum_annealing import SimulatedAnnealingSolver

FAMILY_SIZES = (4, 5, 8, 10, 15, 20)
FACTORS = (0.5, 1.01, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0)

#: Two factors within this many percentage points of each other are a tie, not
#: a ranking. Without it the easy instances (where everything scores +0.00%)
#: vote for whichever factor sorts first.
TIE_PCT = 0.5


def instance_path(n: int) -> Path:
    return (
        default_snapshot_dir().parent
        / "instances"
        / f"iridium33_20260402_planecluster-v1_n{n}_td30d.npz"
    )


def reference(instance: ProblemInstance, mip_s: float) -> float | None:
    exact = ExactSolver().solve(instance)
    if exact.feasible:
        return exact.total_dv_kms
    mip = CPSATSolver(time_limit_s=mip_s).solve(instance)
    return mip.total_dv_kms if mip.feasible else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--reads", type=int, default=1000)
    parser.add_argument("--sweeps", type=int, default=2000)
    parser.add_argument("--mip-seconds", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=Path("docs/data/penalty_study.json"))
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    sampler = SimulatedAnnealingSolver(num_reads=args.reads, num_sweeps=args.sweeps)
    results = []
    # factor -> list of per-instance mean gaps, for the "consistent winner" test
    by_factor: dict[float, list[float]] = defaultdict(list)

    for n in FAMILY_SIZES:
        path = instance_path(n)
        if not path.exists():
            print(f"skip {path.name}: not built")
            continue
        instance = ProblemInstance.load(path)
        ref = reference(instance, args.mip_seconds)
        bound = path_upper_bound(instance)
        print(f"\n=== N={n} td30d  reference {ref:.4f} km/s  greedy UB {bound:.4f} ===", flush=True)

        for factor in FACTORS:
            gaps, rates = [], []
            for seed in seeds:
                qubo = build_qubo(instance, penalty=factor * bound)
                samples = sampler.sample(qubo, seed=seed)
                summary = summarize_samples(instance, qubo, samples)
                gaps.append(100.0 * (summary["best_repaired_dv_kms"] / ref - 1.0))
                rates.append(summary["feasibility_rate"])

            mean_gap = statistics.fmean(gaps)
            record = {
                "n": n,
                "factor": factor,
                "penalty": factor * bound,
                "provably_sufficient": factor > 1.0,
                "mean_gap_pct": mean_gap,
                "best_gap_pct": min(gaps),
                "worst_gap_pct": max(gaps),
                "stdev_gap_pct": statistics.stdev(gaps) if len(gaps) > 1 else None,
                "mean_feasibility": statistics.fmean(rates),
                "seeds": len(seeds),
            }
            results.append(record)
            by_factor[factor].append(mean_gap)
            sd = record["stdev_gap_pct"]
            print(
                f"  factor {factor:<5} gap {mean_gap:+7.2f}%"
                f"  (best {min(gaps):+.2f} worst {max(gaps):+.2f}"
                + (f" sd {sd:.2f}" if sd else "")
                + f")  feas {statistics.fmean(rates):.2f}"
                + ("" if factor > 1.0 else "   [NOT provably sufficient]"),
                flush=True,
            )

    # Does one factor win consistently? Count per-instance wins among the
    # provably sufficient factors only -- a default that voids the bound is not
    # a candidate no matter how well it scores.
    #
    # TIES COUNT FOR NOBODY. On the easy instances every factor scores +0.00%,
    # and a plain argmin hands the win to whichever factor happens to sort
    # first -- which manufactures a "consistent winner" out of instances that
    # discriminate nothing. An instance only votes if some factor beats the
    # rest by more than TIE_PCT.
    sufficient = [f for f in FACTORS if f > 1.0]
    wins: dict[float, int] = dict.fromkeys(sufficient, 0)
    instances = sorted({r["n"] for r in results})
    discriminating = []
    for n in instances:
        rows = [r for r in results if r["n"] == n and r["provably_sufficient"]]
        if not rows:
            continue
        ranked = sorted(rows, key=lambda r: r["mean_gap_pct"])
        if len(ranked) > 1 and ranked[1]["mean_gap_pct"] - ranked[0]["mean_gap_pct"] <= TIE_PCT:
            continue  # tied at the top: this instance does not discriminate
        discriminating.append(n)
        wins[ranked[0]["factor"]] += 1

    total = len(discriminating)
    print(
        f"\n=== wins among provably sufficient factors ===\n"
        f"  {total}/{len(instances)} instances discriminate "
        f"(N={discriminating}); the rest tie within {TIE_PCT}pp and vote for nobody"
    )
    for factor, count in sorted(wins.items(), key=lambda kv: -kv[1]):
        means = by_factor[factor]
        overall = statistics.fmean(means) if means else float("nan")
        print(f"  factor {factor:<5} wins {count}/{total}  mean-of-means {overall:+.2f}%")

    best_factor, best_wins = max(wins.items(), key=lambda kv: kv[1])
    if total and best_wins > total / 2:
        verdict = f"factor {best_factor} wins {best_wins}/{total} discriminating instances"
    else:
        verdict = (
            f"NO consistent winner: {total} discriminating instances, "
            f"best factor {best_factor} takes only {best_wins}. Keep the current default."
        )
    print(f"\nVERDICT: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "reads": args.reads,
                "sweeps": args.sweeps,
                "factors": list(FACTORS),
                "wins": {str(k): v for k, v in wins.items()},
                "discriminating_instances": discriminating,
                "tie_pct": TIE_PCT,
                "verdict": verdict,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
