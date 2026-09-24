"""Multi-start local search over the open path. Works on time-slotted costs.

This exists to fill the hole ``ortools_routing`` leaves. A routing arc-cost
callback cannot express ``C[t,i,j]`` (see CLAUDE.md limitation 8), so the two
instances in the committed family with any headroom -- ``n8_td30d`` and
``n15_td30d`` -- had no strong classical baseline at all. A solver that does not
care whether the cost is static or time-slotted does, because it never asks for
a cost matrix; it asks the instance what a whole path costs.

Neighbourhood: 2-opt (reverse a segment) plus or-opt (move one object
elsewhere). Both are re-evaluated with ``instance.path_cost``, i.e. in full.

    ponytail: O(N) re-evaluation per candidate move, not O(1) delta. Under
    time-slotted costs a 2-opt reversal changes the slot of every leg inside
    the segment, so the textbook four-leg delta is simply wrong here, and N<=20
    makes the difference unmeasurable. If N ever reaches the hundreds, or-opt
    admits an O(1) delta and 2-opt does not -- split them then.
"""

from __future__ import annotations

import time

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.solvers.greedy import greedy_from

__all__ = ["LocalSearchSolver", "local_search"]

#: Enough restarts that the result is about the neighbourhood rather than about
#: the luck of one starting order, cheap enough to stay in the noise of a run.
DEFAULT_RESTARTS = 20


def _descend(instance: ProblemInstance, sequence: list[int]) -> tuple[list[int], float]:
    """Best-improvement descent to a local optimum of 2-opt + or-opt."""
    n = instance.n
    best = list(sequence)
    best_cost = instance.path_cost(best)

    improved = True
    while improved:
        improved = False
        champion, champion_cost = best, best_cost

        # 2-opt: reverse [i, j]. On an open path the segment ends are free, so
        # unlike a tour this includes reversals touching the first/last object.
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                cost = instance.path_cost(candidate)
                if cost < champion_cost:
                    champion, champion_cost = candidate, cost

        # or-opt: lift one object out and reinsert it at every other position.
        for i in range(n):
            without = best[:i] + best[i + 1 :]
            for j in range(n):
                if j == i:
                    continue
                candidate = without[:j] + [best[i]] + without[j:]
                cost = instance.path_cost(candidate)
                if cost < champion_cost:
                    champion, champion_cost = candidate, cost

        if champion_cost < best_cost - 1e-15:
            best, best_cost = champion, champion_cost
            improved = True

    return best, best_cost


def local_search(
    instance: ProblemInstance,
    seed: int | None = None,
    restarts: int = DEFAULT_RESTARTS,
) -> tuple[tuple[int, ...], float, dict[str, object]]:
    """Descend from the greedy path and from ``restarts`` random orders."""
    rng = np.random.default_rng(seed)
    starts = [list(greedy_from(instance, start)[0]) for start in range(instance.n)]
    starts += [list(rng.permutation(instance.n)) for _ in range(max(0, restarts))]

    best_sequence: list[int] | None = None
    best_cost = float("inf")
    costs = []
    for start in starts:
        sequence, cost = _descend(instance, start)
        costs.append(cost)
        if cost < best_cost:
            best_sequence, best_cost = sequence, cost

    assert best_sequence is not None
    stats = {
        "descents": len(starts),
        "greedy_starts": instance.n,
        "random_starts": max(0, restarts),
        # The spread across descents says how rugged the landscape is: a solver
        # whose every restart lands on the same number is not searching.
        "descent_mean_dv_kms": float(np.mean(costs)),
        "descent_std_dv_kms": float(np.std(costs)),
        "descent_worst_dv_kms": float(np.max(costs)),
    }
    return tuple(best_sequence), float(best_cost), stats


class LocalSearchSolver:
    """2-opt + or-opt from every greedy start and ``restarts`` random ones.

    Stochastic through the random restarts, so ``seed`` matters and is recorded.
    """

    name = "local-search"

    def __init__(self, restarts: int = DEFAULT_RESTARTS) -> None:
        self.restarts = int(restarts)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        sequence, total, stats = local_search(instance, seed=seed, restarts=self.restarts)
        return Solution(
            sequence=sequence,
            total_dv_kms=total,
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                "neighbourhood": "2-opt + or-opt",
                "handles_time_dependent_costs": True,
                "seed": seed,
                **stats,
            },
        )
