"""Exact / bounded MIP for open-path sequencing, via OR-Tools CP-SAT.

This exists because the benchmark ran out of oracles exactly where it got
interesting. Held-Karp stops at N=18, and ``ortools`` routing refuses
time-slotted costs outright (its arc-cost callback sees only ``(from, to)``).
That left N=15 and N=20 on the time-dependent instances -- the only ones with
any headroom -- with nothing credible to compare against.

Formulation: the time-indexed assignment model, which is the same encoding the
QUBO uses, minus the penalties.

    x[i,p] in {0,1}     object i is visited p-th
    y[i,j,p] in {0,1}   leg p flies i -> j          (i != j, p in 0..N-2)

    sum_p x[i,p] = 1      for every object i
    sum_i x[i,p] = 1      for every position p
    sum_ij y[i,j,p] = 1   exactly one leg at each position

    y[i,j,p] <= x[i,p]
    y[i,j,p] <= x[j,p+1]
    y[i,j,p] >= x[i,p] + x[j,p+1] - 1

    minimise sum_p sum_ij C_p[i,j] * y[i,j,p]

The three inequalities are the standard linearisation of the product
``x[i,p] * x[j,p+1]``. The last one is implied by the others here (the one-hot
rows already force y to the single (i,j) that is actually flown), but it is
stated anyway: it costs nothing, and a linearisation that relies on an implicit
argument is the kind of thing that breaks silently when someone relaxes a
constraint later.

N-1 legs, not N. There is no wrap-around term and no depot.

CP-SAT is integral, so costs are scaled to integers. The delta-v that gets
REPORTED is recomputed from the unrounded instance, so the rounding only ever
costs search quality -- never accuracy of the number.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution

__all__ = ["CPSATSolver", "COST_SCALE", "CPSAT_MAX_N"]

#: km/s -> mm/s, as in ``ortools_routing``. int64 has ample room at N=20.
COST_SCALE = 1_000_000

#: ~N**3 booleans: 7220 at N=20, 56k at N=40. Not a correctness wall, a
#: model-build-time one. Raise it deliberately.
CPSAT_MAX_N = 40


def _infeasible(name: str, reason: str, t0: float, **metadata: Any) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=time.perf_counter() - t0,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason, **metadata},
    )


class CPSATSolver:
    """Time-indexed MIP. Proves optimality when it can, bounds it when it cannot.

    Unlike every heuristic in this package, a result here carries a *lower
    bound*. ``metadata["proven_optimal"]`` says whether the search closed the
    gap; ``metadata["gap"]`` says how far it got if not. A heuristic that
    happens to be optimal and a solver that has proved it are different claims,
    and the benchmark needs to be able to tell them apart.
    """

    name = "cpsat"

    def __init__(
        self,
        time_limit_s: float = 60.0,
        workers: int = 8,
        max_n: int = CPSAT_MAX_N,
    ) -> None:
        self.time_limit_s = float(time_limit_s)
        self.workers = int(workers)
        self.max_n = int(max_n)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        if instance.n > self.max_n:
            return _infeasible(self.name, f"N={instance.n} exceeds cpsat limit {self.max_n}", t0)
        try:
            return self._run(instance, seed, t0)
        except ImportError as exc:
            return _infeasible(self.name, f"backend unavailable: {exc}", t0)

    def _run(self, instance: ProblemInstance, seed: int | None, t0: float) -> Solution:
        from ortools.sat.python import cp_model

        n = instance.n
        model = cp_model.CpModel()

        x = {(i, p): model.NewBoolVar(f"x_{i}_{p}") for i in range(n) for p in range(n)}
        for i in range(n):
            model.AddExactlyOne(x[i, p] for p in range(n))
        for p in range(n):
            model.AddExactlyOne(x[i, p] for i in range(n))

        y: dict[tuple[int, int, int], Any] = {}
        objective = []
        for p in range(n - 1):
            legs = np.rint(instance.leg_costs(p) * COST_SCALE).astype(np.int64)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue  # an object cannot be its own successor
                    var = model.NewBoolVar(f"y_{i}_{j}_{p}")
                    y[i, j, p] = var
                    model.Add(var <= x[i, p])
                    model.Add(var <= x[j, p + 1])
                    model.Add(var >= x[i, p] + x[j, p + 1] - 1)
                    objective.append(int(legs[i, j]) * var)
            model.AddExactlyOne(y[i, j, p] for i in range(n) for j in range(n) if i != j)

        model.Minimize(sum(objective))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_workers = self.workers
        if seed is not None:
            solver.parameters.random_seed = int(seed)
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return _infeasible(
                self.name,
                f"no solution within {self.time_limit_s}s (status {solver.StatusName(status)})",
                t0,
                status=solver.StatusName(status),
            )

        sequence = []
        for p in range(n):
            sequence.append(next(i for i in range(n) if solver.Value(x[i, p])))

        bound_kms = solver.BestObjectiveBound() / COST_SCALE
        # Report the cost of the sequence under the real (unrounded) costs.
        total = instance.path_cost(sequence)
        # The bound is computed on the rounded integer costs and the total is
        # not, so at a proven optimum the two can cross by up to half a scale
        # unit and produce a nonsensical negative gap. Clamp, and keep the raw
        # bound in the metadata so the crossing stays visible if it ever grows.
        gap = max(0.0, (total - bound_kms) / total) if total > 0 else 0.0

        return Solution(
            sequence=tuple(sequence),
            total_dv_kms=total,
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                "proven_optimal": status == cp_model.OPTIMAL,
                "status": solver.StatusName(status),
                "lower_bound_kms": bound_kms,
                "gap": gap,
                "scaled_objective": int(solver.ObjectiveValue()),
                "cost_scale": COST_SCALE,
                "num_variables": n * n + (n - 1) * n * (n - 1),
                "time_limit_s": self.time_limit_s,
                "workers": self.workers,
                "solver_wall_time_s": solver.WallTime(),
                "seed": seed,
            },
        )
