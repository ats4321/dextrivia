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
costs search quality -- never accuracy of the number. The BOUND is another
matter: it is a bound on the rounded problem, and each of the N-1 rounded legs
can sit up to half a scale unit above its true cost. ``lower_bound_kms``
therefore subtracts ``(N-1) * 0.5 / COST_SCALE`` to stay a bound on the real
problem; the unadjusted value is kept as ``raw_bound_kms``.

Warm start: the model is hinted with the ``localsearch`` sequence
(``AddHint``). Without it, a 30 s run at N=30 was measured returning a path
more than three times worse than greedy. The hint is a starting point, not a
constraint -- CP-SAT may still find something better. Whether the result
at least matches the hint is recorded in ``beats_or_ties_hint``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution

__all__ = ["CPSATSolver", "COST_SCALE", "CPSAT_MAX_N"]

#: km/s -> mm/s, as in ``ortools_routing``. int64 has ample room at N=20.
COST_SCALE = 1_000_000

#: ~N**3 booleans: 7220 at N=20, 56k at N=40, 205k at N=60. Not a
#: correctness wall, a memory and bound-quality one. Measured
#: (docs/data/bounds_study.json, 60 s): N=50 static proved optimal at 2.8 GB
#: peak; N=60 static needed 4.3 GB and left a 20% gap where highs left 7.7%
#: on 1.1 GB. So 50, and highs covers N=60.
CPSAT_MAX_N = 50


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
        warm_start: bool = True,
        circuit: bool = True,
    ) -> None:
        self.time_limit_s = float(time_limit_s)
        self.workers = int(workers)
        self.max_n = int(max_n)
        self.warm_start = bool(warm_start)
        self.circuit = bool(circuit)

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

        if self.circuit:
            # Redundant, and the reason the bound is worth reading. Summing legs
            # over positions gives an arc z[i,j]; closing the path through a
            # depot (d -> first, last -> d) makes those arcs one Hamiltonian
            # circuit. AddCircuit hands CP-SAT's LP the subtour cuts that the
            # time-indexed relaxation lacks on its own.
            depot = n
            arcs = []
            z_vars: dict[tuple[int, int], Any] = {}
            for i in range(n):
                arcs.append((depot, i, x[i, 0]))
                arcs.append((i, depot, x[i, n - 1]))
                for j in range(n):
                    if i != j:
                        z = z_vars[i, j] = model.NewBoolVar(f"z_{i}_{j}")
                        model.Add(z == sum(y[i, j, p] for p in range(n - 1)))
                        arcs.append((i, j, z))
            model.AddCircuit(arcs)

        hint = None
        if self.warm_start:
            from dextrivia.solvers.permutation import LocalSearchSolver

            hint = LocalSearchSolver().solve(instance)
            position = {obj: p for p, obj in enumerate(hint.sequence)}
            for (i, p), var in x.items():
                model.AddHint(var, position[i] == p)
            for (i, j, p), var in y.items():
                model.AddHint(var, position[i] == p and position[j] == p + 1)
            # Hint the circuit's arcs too: CP-SAT is only guaranteed to load a
            # hint as its first solution when the hint is complete.
            if self.circuit:
                for (i, j), var in z_vars.items():
                    model.AddHint(var, position[j] == position[i] + 1)

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

        raw_bound_kms = solver.BestObjectiveBound() / COST_SCALE
        bound_kms = raw_bound_kms - (n - 1) * 0.5 / COST_SCALE
        # Report the cost of the sequence under the real (unrounded) costs.
        total = instance.path_cost(sequence)
        # With the rounding margin subtracted the bound cannot exceed the true
        # optimum, but clamp anyway: a negative gap would be a bug to surface,
        # not a number to print.
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
                "raw_bound_kms": raw_bound_kms,
                "bound_certification": (
                    "CP-SAT best objective bound on mm/s-rounded costs, minus "
                    "(N-1)*0.5 mm/s rounding margin"
                ),
                "warm_start": "localsearch" if hint else None,
                "circuit_constraint": self.circuit,
                "hint_dv_kms": hint.total_dv_kms if hint else None,
                "beats_or_ties_hint": (bool(total <= hint.total_dv_kms + 1e-9) if hint else None),
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
