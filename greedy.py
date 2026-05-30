"""
Dextriva — greedy nearest-neighbor baseline solver

Always visits the cheapest unvisited debris object next.
This is the classical baseline your QUBO solver will beat.
"""

import numpy as np


def greedy_solve(cost_matrix: np.ndarray, start: int = 0) -> dict:
    """
    Greedy nearest-neighbor solver for debris sequencing.

    Parameters
    ----------
    cost_matrix : np.ndarray (N, N)
        cost[i][j] = delta-v in km/s to go from i to j
    start : int
        Index of starting debris object (default 0)

    Returns
    -------
    dict:
        'sequence'     : list[int]   ordered list of debris indices
        'total_deltav' : float       total mission delta-v in km/s
        'step_costs'   : list[float] delta-v for each individual transfer
    """
    n = len(cost_matrix)
    visited  = [False] * n
    sequence = [start]
    step_costs = []
    total_dv = 0.0

    visited[start] = True
    current = start

    for _ in range(n - 1):
        # Find cheapest unvisited next object
        best_cost = float('inf')
        best_next = -1

        for j in range(n):
            if not visited[j] and cost_matrix[current][j] < best_cost:
                best_cost = cost_matrix[current][j]
                best_next = j

        visited[best_next] = True
        sequence.append(best_next)
        step_costs.append(best_cost)
        total_dv += best_cost
        current = best_next

    return {
        'sequence':     sequence,
        'total_deltav': total_dv,
        'step_costs':   step_costs,
    }