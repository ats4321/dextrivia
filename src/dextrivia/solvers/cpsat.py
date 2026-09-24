"""CP-SAT reference model. The strong classical bar, including on time-slotted costs.

``ortools_routing`` is a routing model, so its arc-cost callback sees only
``(from, to)`` and cannot express ``C[t,i,j]`` -- which left the two instances
with any headroom (``n8_td30d``, ``n15_td30d``) with no strong classical
baseline at all above Held-Karp range (CLAUDE.md limitations 8 and 11). A
position-indexed CP-SAT model has no such blind spot: the leg variable is
indexed by position, which is exactly what a time slot is.

Model
-----
``x[i,p]`` is 1 when object ``i`` is visited ``p``-th (one per row, one per
column -- a permutation matrix, the same encoding the QUBO uses). Leg variables
``y[p,i,j]`` are forced on by ``y[p,i,j] >= x[i,p] + x[j,p+1] - 1`` and carry
the objective. There are ``N-1`` leg positions, not ``N``: open path, no return
leg.

Not an oracle by default and not a heuristic either: CP-SAT either *proves*
optimality or runs out of time. ``proven_optimal`` in the metadata says which
happened, and a time-limited answer is never reported as an optimum. Validated
against Held-Karp in ``tests/test_bench_solvers.py``.

Arc costs are int64, so delta-v in km/s is scaled to mm/s exactly as
``ortools_routing`` does; the reported delta-v is recomputed from the unrounded
instance, so rounding costs search quality and never accuracy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.qubo import library_versions
from dextrivia.solvers.greedy import GreedySolver

__all__ = ["CPSATSolver", "COST_SCALE"]

#: km/s -> mm/s, matching ``ortools_routing.COST_SCALE`` so the two models round
#: identically and a difference between them is never a unit artefact.
COST_SCALE = 1_000_000

#: CP-SAT proves optimality in milliseconds up to about N=12 here and burns the
#: whole limit above it. Long enough to be a fair bar, short enough to benchmark.
#: Like OR-Tools' guided local search, this is a configured knob and not a
#: measurement -- quote it whenever a CP-SAT runtime is quoted.
DEFAULT_TIME_LIMIT_S = 10.0

_PACKAGES = ("ortools", "numpy")


def _infeasible(name: str, reason: str, t0: float, **metadata: Any) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=time.perf_counter() - t0,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason, "versions": library_versions(*_PACKAGES), **metadata},
    )


class CPSATSolver:
    """Position-indexed CP-SAT model. Handles static and time-slotted costs.

    ``seed`` is passed to CP-SAT's own ``random_seed``. With more than one
    worker the search is not bit-reproducible even so, which is why
    ``num_workers`` is recorded alongside it.
    """

    name = "cpsat"

    def __init__(self, time_limit_s: float = DEFAULT_TIME_LIMIT_S, num_workers: int = 8) -> None:
        self.time_limit_s = float(time_limit_s)
        self.num_workers = int(num_workers)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        try:
            return self._run(instance, seed, t0)
        except ImportError as exc:
            return _infeasible(self.name, f"backend unavailable: {exc}", t0)

    def _run(self, instance: ProblemInstance, seed: int | None, t0: float) -> Solution:
        from ortools.sat.python import cp_model

        n = instance.n
        legs = [np.rint(instance.leg_costs(p) * COST_SCALE).astype(np.int64) for p in range(n - 1)]

        model = cp_model.CpModel()
        x = [[model.new_bool_var(f"x_{i}_{p}") for p in range(n)] for i in range(n)]
        for i in range(n):
            model.add_exactly_one(x[i][p] for p in range(n))
        for p in range(n):
            model.add_exactly_one(x[i][p] for i in range(n))

        terms = []
        for p in range(n - 1):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue  # an object cannot occupy two positions
                    y = model.new_bool_var(f"y_{p}_{i}_{j}")
                    # Costs are non-negative and this is a minimisation, so the
                    # lower bound is enough: y is never 1 unless it has to be.
                    model.add(y >= x[i][p] + x[j][p + 1] - 1)
                    terms.append(int(legs[p][i, j]) * y)
        model.minimize(sum(terms))

        # Warm start from greedy. Standard practice for a MIP/CP reference, and
        # necessary for this one to deserve the name: without it CP-SAT spends
        # its whole budget at N=20 and returns a path worse than the greedy
        # heuristic it is supposed to be a bar above. Recorded in the metadata,
        # because a warm-started result is a statement about CP-SAT *plus*
        # greedy, and the reader has to be able to see that.
        # Every variable, not just the ones set to 1: CP-SAT silently discards a
        # partial hint. Measured on n20_static -- partial hint 2.7875 km/s,
        # complete hint 2.4485, greedy itself 2.4485. A dropped hint looks
        # exactly like a solver that lost to greedy.
        hint = GreedySolver().solve(instance)
        at_position = {i: p for p, i in enumerate(hint.sequence)}
        for i in range(n):
            for p in range(n):
                model.add_hint(x[i][p], int(at_position[i] == p))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_workers = self.num_workers
        if seed is not None:
            solver.parameters.random_seed = int(seed)
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return _infeasible(
                self.name,
                f"cp-sat returned {solver.status_name(status)} within {self.time_limit_s}s",
                t0,
                time_limit_s=self.time_limit_s,
            )

        position = {}
        for i in range(n):
            for p in range(n):
                if solver.value(x[i][p]):
                    position[p] = i
        sequence = tuple(position[p] for p in sorted(position))
        if sorted(sequence) != list(range(n)):
            return _infeasible(self.name, "cp-sat returned a non-permutation", t0)

        return Solution(
            sequence=sequence,
            total_dv_kms=instance.path_cost(sequence),
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                # Structural constraints, so nothing to repair and nothing to
                # penalise -- the 1.0 here is not comparable with a sampler's.
                "feasibility_rate": 1.0,
                "feasibility_is_structural": True,
                "status": solver.status_name(status),
                # The whole point of the column: a time-limited answer is a
                # bound, not an optimum, and must never be quoted as one.
                "proven_optimal": status == cp_model.OPTIMAL,
                "best_objective_bound_kms": solver.best_objective_bound / COST_SCALE,
                "time_limit_s": self.time_limit_s,
                "num_workers": self.num_workers,
                "handles_time_dependent_costs": True,
                "warm_start": "greedy",
                "warm_start_dv_kms": hint.total_dv_kms,
                "cost_scale": COST_SCALE,
                "booleans": n * n + (n - 1) * n * (n - 1),
                "seed": seed,
                "versions": library_versions(*_PACKAGES),
            },
        )
