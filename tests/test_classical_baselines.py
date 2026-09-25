"""Contract for the classical baselines: CP-SAT, local search, permutation SA.

These exist to make the QUBO comparison honest, so the bar for them is the same
as for the solvers they are controlling: they must be right on instances where
the answer is independently known, and they must record a miss rather than
raise when they cannot cope.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime

import numpy as np
import pytest

from dextrivia.core import ProblemInstance
from dextrivia.solvers import (
    SOLVERS,
    CPSATSolver,
    ExactSolver,
    GreedySolver,
    LocalSearchSolver,
    PermutationAnnealingSolver,
)
from dextrivia.solvers.permutation import or_opt, path_cost_array, swap, two_opt

EPOCH = datetime(2026, 4, 2, tzinfo=UTC)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


needs_ortools = pytest.mark.skipif(
    not _installed("ortools.sat.python"), reason="quantum extra not installed (ortools)"
)

#: Keep every annealing budget in this file small; these are contract tests,
#: not a benchmark. The benchmark lives in scripts/benchmark_family.py.
FAST_SA = dict(time_limit_s=0.15, restarts=1)


def random_instance(n: int, seed: int, *, time_dependent: bool = False) -> ProblemInstance:
    rng = np.random.default_rng(seed)
    shape = (n - 1, n, n) if time_dependent else (n, n)
    costs = rng.random(shape) + 0.05
    index = np.arange(n)
    costs[..., index, index] = 0.0
    return ProblemInstance(
        norad_ids=tuple(range(1000, 1000 + n)),
        names=tuple(f"obj{i}" for i in range(n)),
        epoch=EPOCH,
        costs=costs,
    )


TIME_DEPENDENT = [False, True]


def test_registry_exposes_the_new_baselines():
    assert {"cpsat", "localsearch", "sa-perm"} <= set(SOLVERS)


# --------------------------------------------------------------------------
# CP-SAT is the reference above Held-Karp range, so it has to match it below.
# --------------------------------------------------------------------------


@needs_ortools
@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("n", [4, 6, 7])
@pytest.mark.parametrize("seed", range(3))
def test_cpsat_equals_held_karp(n, seed, time_dependent):
    """Both cost shapes. A MIP that is only right on static costs is useless
    here -- the time-slotted instances are the entire reason it exists."""
    instance = random_instance(n, seed, time_dependent=time_dependent)
    mip = CPSATSolver(time_limit_s=20.0).solve(instance)
    oracle = ExactSolver().solve(instance)
    assert mip.feasible
    assert instance.is_permutation(mip.sequence)
    assert mip.total_dv_kms == pytest.approx(oracle.total_dv_kms, abs=1e-6)
    assert mip.metadata["proven_optimal"]


@needs_ortools
def test_cpsat_reports_a_lower_bound_and_a_non_negative_gap():
    instance = random_instance(6, 1, time_dependent=True)
    solution = CPSATSolver(time_limit_s=20.0).solve(instance)
    assert solution.metadata["lower_bound_kms"] <= solution.total_dv_kms + 1e-6
    assert solution.metadata["gap"] >= 0.0
    assert solution.metadata["num_variables"] == 36 + 5 * 6 * 5


@needs_ortools
def test_cpsat_objective_matches_the_reported_delta_v():
    """The scaled integer objective and the unrounded delta-v must agree to the
    scale, or the rounding is costing accuracy rather than just search quality."""
    instance = random_instance(6, 2, time_dependent=True)
    solution = CPSATSolver(time_limit_s=20.0).solve(instance)
    scaled = solution.metadata["scaled_objective"] / solution.metadata["cost_scale"]
    assert scaled == pytest.approx(solution.total_dv_kms, abs=1e-5)


def test_cpsat_refuses_oversized_instances():
    solution = CPSATSolver(max_n=5).solve(random_instance(8, 0))
    assert not solution.feasible
    assert "exceeds" in solution.metadata["reason"]


def test_cpsat_missing_backend_is_recorded_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "ortools.sat.python", None)
    solution = CPSATSolver().solve(random_instance(4, 0))
    assert not solution.feasible
    assert "backend unavailable" in solution.metadata["reason"]


# --------------------------------------------------------------------------
# Permutation-space solvers never emit an invalid sequence.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("n", [3, 5, 9])
@pytest.mark.parametrize("seed", range(3))
def test_permutation_solvers_return_valid_sequences(n, seed, time_dependent):
    instance = random_instance(n, seed, time_dependent=time_dependent)
    for solver in (LocalSearchSolver(), PermutationAnnealingSolver(**FAST_SA)):
        solution = solver.solve(instance, seed=seed)
        assert solution.feasible
        assert instance.is_permutation(solution.sequence)
        # The reported delta-v must be the cost of the sequence it reports.
        assert instance.path_cost(solution.sequence) == pytest.approx(solution.total_dv_kms)


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("seed", range(5))
def test_local_search_never_loses_to_its_own_starting_point(seed, time_dependent):
    """Descent only accepts strict improvements. A local search that can come
    back worse than greedy is a bug, not a heuristic."""
    instance = random_instance(8, seed, time_dependent=time_dependent)
    greedy = GreedySolver().solve(instance)
    improved = LocalSearchSolver().solve(instance)
    assert improved.total_dv_kms <= greedy.total_dv_kms + 1e-12
    assert improved.metadata["start_dv_kms"] == pytest.approx(greedy.total_dv_kms)


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("seed", range(3))
def test_neither_permutation_solver_beats_the_exact_optimum(seed, time_dependent):
    """The oracle bounds them from below; beating it would mean the cost
    evaluation disagrees with ProblemInstance.path_cost."""
    instance = random_instance(7, seed, time_dependent=time_dependent)
    optimum = ExactSolver().solve(instance).total_dv_kms
    for solver in (LocalSearchSolver(), PermutationAnnealingSolver(**FAST_SA)):
        assert solver.solve(instance, seed=seed).total_dv_kms >= optimum - 1e-9


@pytest.mark.parametrize("move", [two_opt, or_opt, swap])
@pytest.mark.parametrize("n", [2, 3, 8])
def test_every_move_preserves_the_permutation(move, n):
    rng = np.random.default_rng(0)
    for _ in range(100):
        sequence = rng.permutation(n)
        assert sorted(move(sequence, rng).tolist()) == list(range(n))


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_vectorised_path_cost_agrees_with_the_instance(time_dependent):
    """path_cost_array inlines leg_costs for speed; if it ever drifts from
    ProblemInstance.path_cost, every permutation result is quietly wrong."""
    instance = random_instance(6, 4, time_dependent=time_dependent)
    rng = np.random.default_rng(1)
    for _ in range(50):
        sequence = rng.permutation(instance.n)
        assert path_cost_array(instance.costs, instance.time_dependent, sequence) == pytest.approx(
            instance.path_cost(sequence.tolist())
        )


def test_permutation_annealing_is_reproducible_given_a_seed():
    instance = random_instance(8, 0, time_dependent=True)
    solver = PermutationAnnealingSolver(time_limit_s=0.2, restarts=1, max_iterations=5000)
    first = solver.solve(instance, seed=42)
    second = solver.solve(instance, seed=42)
    assert first.sequence == second.sequence


def test_permutation_annealing_records_both_budget_currencies():
    """The sa-qubo comparison is only auditable if both budgets are on record."""
    instance = random_instance(6, 0)
    metadata = PermutationAnnealingSolver(**FAST_SA).solve(instance, seed=1).metadata
    assert metadata["time_limit_s"] == FAST_SA["time_limit_s"]
    assert metadata["iterations"] > 0
    assert 0.0 <= metadata["acceptance_rate"] <= 1.0
    assert metadata["search_space"] == "permutation"
    assert metadata["seed"] == 1
