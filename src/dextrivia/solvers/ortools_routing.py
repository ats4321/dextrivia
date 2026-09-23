"""Google OR-Tools routing: the strong classical heuristic above Held-Karp range.

Not a quantum solver and not a QUBO one -- it is the bar. A quantum or
quantum-inspired result at N=4 means nothing next to an exact DP, and at N=50
it means nothing next to this, so the benchmark needs both oracles present.
Owned by the QUBO workspace all the same; see the QUBO section of CLAUDE.md.

Two things it cannot do, both reported honestly rather than papered over:

* **Time-slotted costs.** A routing arc-cost callback is a function of
  ``(from_node, to_node)`` only. It has no idea how many legs have already been
  flown, so it cannot express ``C[t, i, j]``; the model would silently optimise
  the wrong objective. Such instances get ``feasible=False``.
* **Real-valued costs.** Routing arc costs are int64. Delta-v in km/s is scaled
  to mm/s (1e6) for the search, but the delta-v that gets reported is
  recomputed from the unrounded instance, so the rounding only ever costs
  search quality, never accuracy of the number.

The open path is expressed with a dummy depot joined to every object by a
zero-cost arc: a closed tour through that depot, with the depot deleted, is
exactly an open path with a free start and a free end and N-1 real legs.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.qubo import library_versions

__all__ = ["ORToolsRoutingSolver", "COST_SCALE"]

#: km/s -> mm/s. Delta-v differences below 1 mm/s do not survive the cost model
#: anyway, and int64 has room to spare at this scale.
COST_SCALE = 1_000_000

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


class ORToolsRoutingSolver:
    """Guided local search over the open path. ``seed`` is accepted and ignored.

    OR-Tools routing is deterministic given the same model and time limit, so
    the reproducibility knob that matters here is ``time_limit_s``, not a seed.
    Both go in the metadata; a delta-v from a time-limited local search is not
    a property of the instance alone.
    """

    name = "ortools"

    def __init__(
        self, time_limit_s: float = 5.0, first_solution: str = "PATH_CHEAPEST_ARC"
    ) -> None:
        self.time_limit_s = float(time_limit_s)
        self.first_solution = first_solution

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        if instance.time_dependent:
            return _infeasible(
                self.name,
                "instance has time-slotted costs C[t,i,j]; a routing arc-cost "
                "callback sees only (from, to) and cannot express position-dependent legs",
                t0,
            )
        try:
            return self._run(instance, seed, t0)
        except ImportError as exc:
            return _infeasible(self.name, f"backend unavailable: {exc}", t0)

    def _run(self, instance: ProblemInstance, seed: int | None, t0: float) -> Solution:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        n = instance.n
        depot = n  # the dummy; arcs to and from it are free
        costs = np.rint(instance.leg_costs(0) * COST_SCALE).astype(np.int64)

        manager = pywrapcp.RoutingIndexManager(n + 1, 1, depot)
        routing = pywrapcp.RoutingModel(manager)

        def arc_cost(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            if i == depot or j == depot:
                return 0
            return int(costs[i, j])

        routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(arc_cost))

        parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters.first_solution_strategy = getattr(
            routing_enums_pb2.FirstSolutionStrategy, self.first_solution
        )
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        # Guided local search never converges on its own; without a limit it runs
        # until the heat death of the benchmark.
        parameters.time_limit.FromMilliseconds(int(self.time_limit_s * 1000))

        assignment = routing.SolveWithParameters(parameters)
        if assignment is None:
            return _infeasible(self.name, "routing search returned no assignment", t0)

        sequence = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot:
                sequence.append(node)
            index = assignment.Value(routing.NextVar(index))

        if sorted(sequence) != list(range(n)):
            return _infeasible(self.name, f"routing dropped nodes: visited {len(sequence)}/{n}", t0)

        return Solution(
            sequence=tuple(sequence),
            total_dv_kms=instance.path_cost(sequence),
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                "feasibility_rate": 1.0,  # routing constraints are structural, not penalised
                "first_solution_strategy": self.first_solution,
                "metaheuristic": "GUIDED_LOCAL_SEARCH",
                "time_limit_s": self.time_limit_s,
                "cost_scale": COST_SCALE,
                "scaled_objective": int(assignment.ObjectiveValue()),
                "seed": seed,
                "versions": library_versions(*_PACKAGES),
            },
        )
