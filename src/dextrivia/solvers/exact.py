"""Exact open-path solvers: Held-Karp DP, and brute force as its test oracle.

Both handle static C[i, j] and time-slotted C[t, i, j] costs. Held-Karp can do
so for free because the number of visited objects in a DP state already tells
you which leg is being flown: a state covering k objects is about to fly leg
k-1. That is exactly the slot index ``ProblemInstance`` defines, which is why
slots are indexed by position-in-sequence and not by wall-clock time.

Neither solver is a benchmark competitor -- they exist to say what the true
optimum is, so "the heuristic did well" is a measurement rather than a hope.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from dextrivia.core import ProblemInstance, Solution

__all__ = ["ExactSolver", "BruteForceSolver", "held_karp", "brute_force"]

#: 2^N * N^2 time and 2^N * N memory. N=18 is about 40 s and 40 MB; past that
#: the DP stops being a convenience and starts being the experiment.
HELD_KARP_MAX_N = 18

#: N! * N. N=8 is ~0.3 M operations.
BRUTE_FORCE_MAX_N = 8


def held_karp(instance: ProblemInstance) -> tuple[tuple[int, ...], float]:
    """Optimal open path by Held-Karp DP over (visited set, last object)."""
    n = instance.n
    full = (1 << n) - 1
    dp = np.full((1 << n, n), np.inf)
    parent = np.full((1 << n, n), -1, dtype=np.int8)
    for j in range(n):
        dp[1 << j, j] = 0.0

    for mask in range(1, full):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        step = int(mask.bit_count()) - 1  # legs already flown == next leg's slot
        candidates = row[:, None] + instance.leg_costs(step)  # (from, to)
        best_from = np.argmin(candidates, axis=0)
        for to in range(n):
            if mask & (1 << to):
                continue
            value = candidates[best_from[to], to]
            nxt = mask | (1 << to)
            if value < dp[nxt, to]:
                dp[nxt, to] = value
                parent[nxt, to] = best_from[to]

    end = int(np.argmin(dp[full]))
    total = float(dp[full, end])

    sequence = [end]
    mask, last = full, end
    while parent[mask, last] >= 0:
        previous = int(parent[mask, last])
        mask ^= 1 << last
        last = previous
        sequence.append(last)
    sequence.reverse()
    return tuple(sequence), total


def brute_force(instance: ProblemInstance) -> tuple[tuple[int, ...], float]:
    """Optimal open path by enumerating every permutation. Oracle for tests."""
    best_sequence, best_total = None, np.inf
    for permutation in itertools.permutations(range(instance.n)):
        total = instance.path_cost(permutation)
        if total < best_total:
            best_sequence, best_total = permutation, total
    return tuple(best_sequence), float(best_total)


def _infeasible(name: str, reason: str, runtime: float) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=runtime,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason},
    )


class ExactSolver:
    """Held-Karp dynamic program. Optimal, exponential, ``seed`` ignored."""

    name = "exact"
    max_n = HELD_KARP_MAX_N

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        if instance.n > self.max_n:
            return _infeasible(
                self.name,
                f"N={instance.n} exceeds held-karp limit {self.max_n}",
                time.perf_counter() - t0,
            )
        sequence, total = held_karp(instance)
        return Solution(
            sequence=sequence,
            total_dv_kms=total,
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={"optimal": True, "algorithm": "held-karp"},
        )


class BruteForceSolver:
    """Enumerates all N! orders. Only used to check ``ExactSolver`` is right."""

    name = "brute"
    max_n = BRUTE_FORCE_MAX_N

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        if instance.n > self.max_n:
            return _infeasible(
                self.name,
                f"N={instance.n} exceeds brute-force limit {self.max_n}",
                time.perf_counter() - t0,
            )
        sequence, total = brute_force(instance)
        return Solution(
            sequence=sequence,
            total_dv_kms=total,
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={"optimal": True, "algorithm": "brute-force"},
        )
