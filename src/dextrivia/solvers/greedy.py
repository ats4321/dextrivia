"""Greedy nearest-neighbour baseline."""

from __future__ import annotations

import time

import numpy as np

from dextrivia.core import ProblemInstance, Solution

__all__ = ["GreedySolver", "greedy_from"]


def greedy_from(instance: ProblemInstance, start: int) -> tuple[list[int], float]:
    """Nearest-neighbour open path from a fixed start. Returns (sequence, total km/s)."""
    n = instance.n
    unvisited = set(range(n)) - {start}
    sequence = [start]
    total = 0.0
    current = start
    for step in range(n - 1):
        row = instance.leg_costs(step)[current]
        nxt = min(unvisited, key=lambda j: row[j])
        total += float(row[nxt])
        sequence.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return sequence, total


class GreedySolver:
    """Runs nearest-neighbour from every start point and keeps the best path.

    O(N^3) overall. Deterministic: ``seed`` is accepted and ignored.
    """

    name = "greedy"

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        best_sequence, best_total, best_start = None, np.inf, -1
        for start in range(instance.n):
            sequence, total = greedy_from(instance, start)
            if total < best_total:
                best_sequence, best_total, best_start = sequence, total, start
        runtime = time.perf_counter() - t0
        return Solution(
            sequence=tuple(best_sequence),
            total_dv_kms=float(best_total),
            runtime_s=runtime,
            solver_name=self.name,
            feasible=True,
            metadata={"best_start": best_start, "starts_tried": instance.n},
        )
