"""Classical search directly over sequences: local search, and annealing.

These are CONTROLS, not competitors. ``sa-qubo`` anneals a penalty-encoded QUBO
over N**2 binary variables; ``sa-perm`` here anneals over the permutations
themselves with the same temperature schedule shape. Comparing them isolates
the one question the QUBO benchmark cannot otherwise answer: does the *encoding*
buy anything, or is any observed win just annealing being annealing?

``localsearch`` is the cheap strong heuristic -- greedy, then 2-opt and or-opt
to a local optimum. It is also ``sa-perm`` at zero temperature, which is why
both live here and share one set of move operators.

Time dependence changes the move algebra
----------------------------------------
Under static costs a 2-opt delta is O(1): reversing a segment leaves every leg
outside it untouched. Under ``C[t, i, j]`` it is **not**. Reversing a segment
does not just reverse those legs, it changes which leg *index* every subsequent
pair occupies, so the entire suffix re-prices. Or-opt shifts positions and has
the same problem.

So every move here re-evaluates the whole path. That is O(N) rather than O(1),
and at N <= 20 it is microseconds -- a price worth paying, because the textbook
O(1) 2-opt delta is silently *wrong* on a time-slotted instance and would
produce plausible-looking sequences that do not optimise the stated objective.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.solvers.greedy import GreedySolver

__all__ = [
    "LocalSearchSolver",
    "PermutationAnnealingSolver",
    "path_cost_array",
    "two_opt",
    "or_opt",
    "swap",
]


def path_cost_array(costs: np.ndarray, time_dependent: bool, sequence: np.ndarray) -> float:
    """Total delta-v of ``sequence``, vectorised.

    Takes the raw array rather than the instance because it is the inner loop of
    the annealer and runs millions of times. The caller is responsible for
    having obtained ``costs`` from the instance in the first place; the two
    branches here are exactly ``ProblemInstance.leg_costs`` inlined.
    """
    tail, head = sequence[:-1], sequence[1:]
    if time_dependent:
        return float(costs[np.arange(len(tail)), tail, head].sum())
    return float(costs[tail, head].sum())


def two_opt(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reverse a random internal segment."""
    n = len(sequence)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    if j - i < 1:
        return sequence
    out = sequence.copy()
    out[i : j + 1] = out[i : j + 1][::-1]
    return out


def or_opt(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lift a run of 1-3 objects out and reinsert it elsewhere."""
    n = len(sequence)
    length = int(rng.integers(1, min(3, n - 1) + 1))
    start = int(rng.integers(0, n - length + 1))
    segment = sequence[start : start + length]
    rest = np.concatenate([sequence[:start], sequence[start + length :]])
    insert = int(rng.integers(0, len(rest) + 1))
    return np.concatenate([rest[:insert], segment, rest[insert:]])


def swap(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exchange two positions."""
    i, j = rng.choice(len(sequence), size=2, replace=False)
    out = sequence.copy()
    out[i], out[j] = out[j], out[i]
    return out


#: Equal weight. Deliberately not tuned per instance -- a move mix fitted to the
#: instances it is then evaluated on is not a control, it is a second fit.
MOVES = (two_opt, or_opt, swap)


def _infeasible(name: str, reason: str, t0: float, **metadata: Any) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=time.perf_counter() - t0,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason, **metadata},
    )


def _descend(
    costs: np.ndarray,
    time_dependent: bool,
    sequence: np.ndarray,
    max_passes: int = 100,
) -> tuple[np.ndarray, float, int, int]:
    """Deterministic 2-opt + or-opt descent to a local optimum.

    Full neighbourhood, best-improvement, repeated until a pass finds nothing.
    Returns (sequence, cost, passes, improvements).
    """
    n = len(sequence)
    best = sequence.copy()
    best_cost = path_cost_array(costs, time_dependent, best)
    passes = improvements = 0

    for _ in range(max_passes):
        passes += 1
        improved = False
        # 2-opt: every segment reversal.
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = best.copy()
                candidate[i : j + 1] = candidate[i : j + 1][::-1]
                cost = path_cost_array(costs, time_dependent, candidate)
                if cost < best_cost - 1e-12:
                    best, best_cost, improved = candidate, cost, True
                    improvements += 1
        # or-opt: every run of 1..3 reinserted at every other position.
        for length in (1, 2, 3):
            if length >= n:
                break
            for start in range(n - length + 1):
                segment = best[start : start + length]
                rest = np.concatenate([best[:start], best[start + length :]])
                for insert in range(len(rest) + 1):
                    if insert == start:
                        continue
                    candidate = np.concatenate([rest[:insert], segment, rest[insert:]])
                    cost = path_cost_array(costs, time_dependent, candidate)
                    if cost < best_cost - 1e-12:
                        best, best_cost, improved = candidate, cost, True
                        improvements += 1
        if not improved:
            break

    return best, best_cost, passes, improvements


class LocalSearchSolver:
    """Greedy start, then 2-opt/or-opt descent. Deterministic; ``seed`` ignored.

    The cheap strong heuristic. It cannot be worse than ``greedy`` by
    construction -- descent only ever accepts a strict improvement -- which the
    tests assert, because a "local search" that can lose to its own starting
    point is a bug rather than a heuristic.
    """

    name = "localsearch"

    def __init__(self, max_passes: int = 100) -> None:
        self.max_passes = int(max_passes)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        start = GreedySolver().solve(instance)
        costs = instance.costs
        sequence, cost, passes, improvements = _descend(
            costs, instance.time_dependent, np.asarray(start.sequence), self.max_passes
        )
        return Solution(
            sequence=tuple(int(i) for i in sequence),
            total_dv_kms=float(cost),
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                "start": "greedy",
                "start_dv_kms": start.total_dv_kms,
                "passes": passes,
                "improvements": improvements,
                "seed": seed,
                "deterministic": True,
            },
        )


class PermutationAnnealingSolver:
    """Simulated annealing over sequences. The control for ``sa-qubo``.

    Geometric cooling between temperatures derived from the instance's own cost
    scale, so the schedule transfers across instances without hand-tuning. The
    budget is wall-clock (``time_limit_s``) rather than an iteration count,
    because that is the only budget that can be matched fairly against a
    compiled C++ annealer -- see the note in ``metadata``.
    """

    name = "sa-perm"

    def __init__(
        self,
        time_limit_s: float = 1.0,
        restarts: int = 4,
        cooling: float = 0.995,
        start_from_greedy: bool = True,
        max_iterations: int | None = None,
    ) -> None:
        self.time_limit_s = float(time_limit_s)
        self.restarts = int(restarts)
        self.cooling = float(cooling)
        self.start_from_greedy = bool(start_from_greedy)
        self.max_iterations = max_iterations

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        n = instance.n
        costs = instance.costs
        time_dependent = instance.time_dependent
        rng = np.random.default_rng(seed)

        # Temperature scale from the costs themselves: start hot enough to accept
        # a typical leg's worth of damage, end cold enough to reject it.
        spread = float(np.std(costs)) or 1.0
        t_start, t_end = spread, spread * 1e-4

        greedy_start = GreedySolver().solve(instance) if self.start_from_greedy else None
        budget = self.time_limit_s / max(1, self.restarts)

        best_sequence: np.ndarray | None = None
        best_cost = np.inf
        iterations = 0
        accepted = 0
        restart_costs: list[float] = []

        for restart in range(max(1, self.restarts)):
            if greedy_start is not None and restart == 0:
                current = np.asarray(greedy_start.sequence)
            else:
                current = rng.permutation(n)
            current_cost = path_cost_array(costs, time_dependent, current)
            local_best, local_best_cost = current.copy(), current_cost

            temperature = t_start
            deadline = time.perf_counter() + budget
            while time.perf_counter() < deadline:
                if self.max_iterations is not None and iterations >= self.max_iterations:
                    break
                # Check the clock every 200 moves rather than every move:
                # perf_counter costs more than the move does at this size.
                for _ in range(200):
                    iterations += 1
                    move = MOVES[int(rng.integers(len(MOVES)))]
                    candidate = move(current, rng)
                    candidate_cost = path_cost_array(costs, time_dependent, candidate)
                    delta = candidate_cost - current_cost
                    if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                        current, current_cost = candidate, candidate_cost
                        accepted += 1
                        if current_cost < local_best_cost:
                            local_best, local_best_cost = current.copy(), current_cost
                    temperature = max(temperature * self.cooling, t_end)

            restart_costs.append(float(local_best_cost))
            if local_best_cost < best_cost:
                best_sequence, best_cost = local_best, local_best_cost

        assert best_sequence is not None
        return Solution(
            sequence=tuple(int(i) for i in best_sequence),
            total_dv_kms=float(best_cost),
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                # Budget, recorded in both currencies so the sa-qubo comparison
                # is auditable. Note the asymmetry: this is interpreted Python
                # against dwave-samplers' compiled C++, so a wall-clock match
                # handicaps this solver. A win here is therefore conservative.
                "time_limit_s": self.time_limit_s,
                "iterations": iterations,
                "accepted_moves": accepted,
                "acceptance_rate": accepted / iterations if iterations else 0.0,
                "restarts": self.restarts,
                "restart_costs": restart_costs,
                "cooling": self.cooling,
                "temperature_start": t_start,
                "temperature_end": t_end,
                "start_from_greedy": self.start_from_greedy,
                "greedy_start_dv_kms": greedy_start.total_dv_kms if greedy_start else None,
                "moves": [m.__name__ for m in MOVES],
                "search_space": "permutation",
                "seed": seed,
            },
        )
