"""Solver contract: exact is optimal, greedy is not worse than optimal, both
handle static and time-slotted costs."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from dextrivia.core import ProblemInstance
from dextrivia.solvers import BruteForceSolver, ExactSolver, GreedySolver

EPOCH = datetime(2026, 4, 2, tzinfo=UTC)


def random_instance(n: int, seed: int, *, time_dependent=False, symmetric=True):
    rng = np.random.default_rng(seed)
    shape = (n - 1, n, n) if time_dependent else (n, n)
    costs = rng.random(shape) + 0.05
    if symmetric:
        costs = (costs + np.swapaxes(costs, -1, -2)) / 2
    idx = np.arange(n)
    costs[..., idx, idx] = 0.0
    return ProblemInstance(
        norad_ids=tuple(range(1000, 1000 + n)),
        names=tuple(f"obj{i}" for i in range(n)),
        epoch=EPOCH,
        costs=costs,
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("time_dependent", [False, True])
def test_exact_equals_brute_force_for_small_n(seed, time_dependent):
    """Held-Karp against the only oracle that cannot be subtly wrong."""
    inst = random_instance(7, seed, time_dependent=time_dependent, symmetric=False)
    exact = ExactSolver().solve(inst)
    brute = BruteForceSolver().solve(inst)
    assert exact.total_dv_kms == pytest.approx(brute.total_dv_kms)
    assert inst.path_cost(exact.sequence) == pytest.approx(exact.total_dv_kms)
    assert inst.is_permutation(exact.sequence)


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("time_dependent", [False, True])
def test_exact_is_never_worse_than_greedy(seed, time_dependent):
    inst = random_instance(9, seed, time_dependent=time_dependent, symmetric=False)
    exact = ExactSolver().solve(inst)
    greedy = GreedySolver().solve(inst)
    assert exact.total_dv_kms <= greedy.total_dv_kms + 1e-12
    assert inst.is_permutation(greedy.sequence)
    assert inst.path_cost(greedy.sequence) == pytest.approx(greedy.total_dv_kms)


def test_greedy_is_actually_beatable_on_general_costs():
    """Guards the benchmark: if greedy always tied the optimum on random costs,
    the test above would be vacuous and so would the whole comparison."""
    losses = 0
    for seed in range(30):
        inst = random_instance(9, seed, symmetric=False)
        exact = ExactSolver().solve(inst)
        greedy = GreedySolver().solve(inst)
        if greedy.total_dv_kms > exact.total_dv_kms + 1e-9:
            losses += 1
    assert losses > 0


def test_solutions_report_runtime_and_solver_name():
    inst = random_instance(6, 0)
    for solver in (GreedySolver(), ExactSolver(), BruteForceSolver()):
        solution = solver.solve(inst, seed=1)
        assert solution.solver_name == solver.name
        assert solution.feasible
        assert solution.runtime_s >= 0.0


def test_oversized_instances_are_reported_infeasible_not_raised():
    """Benchmarks must record a miss, not crash on it."""
    inst = random_instance(9, 0)
    solution = BruteForceSolver().solve(inst)
    assert not solution.feasible
    assert solution.sequence == ()
    assert "exceeds" in solution.metadata["reason"]
