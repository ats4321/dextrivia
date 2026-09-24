"""Simulated annealing directly on the permutation. The control for ``sa-qubo``.

``sa-qubo`` anneals a QUBO. Two different things could be responsible for any
result it gets: the annealing, or the position encoding that turns N objects
into N**2 binaries plus penalty terms. Reporting ``sa-qubo`` without this
control cannot tell those apart -- and CLAUDE.md limitation 10 (one win at N=8,
one collapse at N=15) is exactly the shape of question that needs them apart.

This solver anneals the *same* objective with the *same* move budget over a
search space that cannot be infeasible: a permutation, swapped two entries at a
time. Constraints are structural, so there is no penalty weight, no repair, and
the raw feasibility rate is 1.0 by construction.

Matched budget
--------------
``dwave-samplers`` counts one sweep as one proposed update per variable, and the
QUBO has N**2 variables, so ``sa-qubo`` proposes

    num_reads * num_sweeps * N**2

single-spin flips. This solver runs ``num_reads`` independent chains for
``num_sweeps`` sweeps of ``N**2`` proposed swaps each -- the same product, and
both counts are recorded in the metadata of both solvers so the match can be
audited rather than believed.

Wall-clock is NOT matched and cannot be: ``dwave-samplers`` is compiled C++ and
this is numpy. Both runtimes are recorded; neither is a fair comparison of the
two algorithms, and the proposal count is the honest axis.

    ponytail: chains are vectorised across reads rather than looped, so a
    "sweep" is N**2 numpy steps over all chains at once. Same proposals, same
    Metropolis rule, ~100x faster than the obvious Python loop.
"""

from __future__ import annotations

import time

import numpy as np

from dextrivia.core import ProblemInstance, Solution

__all__ = ["PermutationAnnealingSolver", "leg_cost_stack", "anneal_permutations"]

#: Defaults mirror ``SimulatedAnnealingSolver`` so the budgets match out of the box.
DEFAULT_READS = 500
DEFAULT_SWEEPS = 1000

#: Acceptance probabilities the hot and cold ends are calibrated to, for a swap
#: of typical magnitude. Standard annealing practice: start near-random-walk,
#: finish near-greedy. Calibrating against a measured delta rather than against
#: a hardcoded temperature keeps the schedule valid across cost models.
HOT_ACCEPTANCE = 0.8
COLD_ACCEPTANCE = 0.01

#: Random (state, swap) pairs used to measure the typical |delta| of a move.
CALIBRATION_SAMPLES = 512


def leg_cost_stack(instance: ProblemInstance) -> np.ndarray:
    """``(N-1, N, N)`` view of the costs, via the accessor, never ``instance.costs``.

    A static instance yields the same matrix repeated; broadcasting keeps that
    free of a copy. This is the one place the solver touches costs.
    """
    return np.stack([instance.leg_costs(step) for step in range(instance.n - 1)])


def _swap_delta(
    legs: np.ndarray,
    state: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Change in path cost from swapping positions ``p`` and ``q``, per chain.

    Only four legs can change -- the ones entering and leaving each swapped
    position -- and crucially their *slot indices* do not change, because a slot
    is a position in the sequence and a swap moves objects, not positions. That
    is what makes an O(1) delta correct here even for time-slotted costs.

    The four legs are handled as one stacked (4, chains) gather rather than a
    Python loop over them: this runs once per proposed move, which is the
    hottest line in the package.
    """
    n = state.shape[1]
    chains = np.arange(state.shape[0])
    at_p, at_q = state[chains, p], state[chains, q]

    # Legs p-1, p, q-1, q. q-1 is dropped when it *is* leg p (adjacent swap),
    # or it would be counted twice on both sides of the delta.
    starts = np.stack([p - 1, p, q - 1, q])
    valid = np.stack([p >= 1, np.ones_like(p, dtype=bool), q - 1 != p, q <= n - 2])
    starts = np.clip(starts, 0, n - 2)
    finishes = starts + 1
    rows = np.broadcast_to(chains, starts.shape)

    def object_at(position: np.ndarray) -> np.ndarray:
        """Who sits at ``position`` after the swap."""
        return np.where(position == p, at_q, np.where(position == q, at_p, state[rows, position]))

    # One flat gather per side rather than a three-array fancy index: same
    # numbers, roughly twice the speed, and this runs 10^8 times per benchmark.
    flat = legs.reshape(-1)
    plane = starts * (n * n)
    old = flat[plane + state[rows, starts] * n + state[rows, finishes]]
    new = flat[plane + object_at(starts) * n + object_at(finishes)]
    return np.sum(np.where(valid, new - old, 0.0), axis=0)


def _propose(rng: np.random.Generator, reads: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A uniformly random pair of distinct positions per chain, ordered p < q."""
    first = rng.integers(0, n, size=reads)
    second = rng.integers(0, n - 1, size=reads)
    second += second >= first  # skip the collision, keeping the draw uniform
    return np.minimum(first, second), np.maximum(first, second)


def _typical_delta(legs: np.ndarray, rng: np.random.Generator, n: int) -> float:
    """Mean |delta| of a random swap on a random permutation. Sets the schedule."""
    state = np.array([rng.permutation(n) for _ in range(CALIBRATION_SAMPLES)])
    p, q = _propose(rng, CALIBRATION_SAMPLES, n)
    magnitude = float(np.mean(np.abs(_swap_delta(legs, state, p, q))))
    return magnitude if magnitude > 0.0 else 1.0


def anneal_permutations(
    instance: ProblemInstance,
    seed: int | None = None,
    num_reads: int = DEFAULT_READS,
    num_sweeps: int = DEFAULT_SWEEPS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Run ``num_reads`` Metropolis chains. Returns (states, costs, schedule)."""
    n = instance.n
    legs = leg_cost_stack(instance)
    rng = np.random.default_rng(seed)

    scale = _typical_delta(legs, rng, n)
    t_hot = scale / np.log(1.0 / HOT_ACCEPTANCE)
    t_cold = scale / np.log(1.0 / COLD_ACCEPTANCE)
    temperatures = np.geomspace(t_hot, t_cold, num=max(1, num_sweeps))

    state = np.array([rng.permutation(n) for _ in range(num_reads)])
    moves_per_sweep = n * n
    accepted = 0
    for temperature in temperatures:
        # Drawn a sweep at a time: three RNG calls per sweep instead of three
        # per move, which is worth about a third of the runtime at N=20.
        firsts = rng.integers(0, n, size=(moves_per_sweep, num_reads))
        seconds = rng.integers(0, n - 1, size=(moves_per_sweep, num_reads))
        seconds += seconds >= firsts  # skip the collision, keeping the draw uniform
        uniforms = rng.random((moves_per_sweep, num_reads))

        for first, second, uniform in zip(firsts, seconds, uniforms, strict=True):
            p, q = np.minimum(first, second), np.maximum(first, second)
            delta = _swap_delta(legs, state, p, q)
            take = (delta <= 0.0) | (uniform < np.exp(-np.clip(delta, 0.0, None) / temperature))
            accepted += int(np.count_nonzero(take))
            chains = np.nonzero(take)[0]
            if chains.size:
                pc, qc = p[chains], q[chains]
                state[chains, pc], state[chains, qc] = state[chains, qc], state[chains, pc]

    costs = np.array([instance.path_cost(chain) for chain in state])
    proposals = num_reads * max(1, num_sweeps) * moves_per_sweep
    schedule = {
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "moves_per_sweep": moves_per_sweep,
        "proposed_moves": proposals,
        "accepted_moves": accepted,
        "acceptance_rate": accepted / proposals if proposals else 0.0,
        "temperature_hot": float(t_hot),
        "temperature_cold": float(t_cold),
        "typical_swap_delta_kms": scale,
    }
    return state, costs, schedule


class PermutationAnnealingSolver:
    """Metropolis annealing over permutations. Always feasible, no penalty weight.

    Budget-matched to ``sa-qubo``: same ``num_reads`` and ``num_sweeps``, and a
    sweep is ``N**2`` proposed moves on both sides. See the module docstring.
    """

    name = "sa-perm"

    def __init__(self, num_reads: int = DEFAULT_READS, num_sweeps: int = DEFAULT_SWEEPS) -> None:
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        states, costs, schedule = anneal_permutations(
            instance, seed=seed, num_reads=self.num_reads, num_sweeps=self.num_sweeps
        )
        best = int(np.argmin(costs))
        return Solution(
            sequence=tuple(int(i) for i in states[best]),
            total_dv_kms=float(costs[best]),
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                # 1.0 by construction, not by luck: the search space is the set
                # of permutations, so no chain can end anywhere infeasible.
                "feasibility_rate": 1.0,
                "feasibility_is_structural": True,
                "chain_mean_dv_kms": float(np.mean(costs)),
                "chain_std_dv_kms": float(np.std(costs)),
                "chain_worst_dv_kms": float(np.max(costs)),
                "seed": seed,
                **schedule,
            },
        )
