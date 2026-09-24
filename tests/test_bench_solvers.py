"""The three solvers the benchmark workspace added: cpsat, local-search, sa-perm.

The load-bearing test here is ``cpsat`` against Held-Karp. A CP-SAT model is the
benchmark's only strong classical bar above Held-Karp range *and* on time-slotted
costs, so if its objective disagrees with the exact DP the whole N>=15 comparison
is measuring the wrong thing.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

import dextrivia.solvers
from dextrivia.solvers import SOLVERS, CPSATSolver, GreedySolver, LocalSearchSolver
from dextrivia.solvers.exact import held_karp
from dextrivia.solvers.perm_annealing import (
    PermutationAnnealingSolver,
    _propose,
    _swap_delta,
    anneal_permutations,
    leg_cost_stack,
)
from tests.test_qubo_formulation import random_instance


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


needs_ortools = pytest.mark.skipif(
    not _installed("ortools"), reason="quantum extra not installed (ortools)"
)

TIME_DEPENDENT = (False, True)
NEW_MODULES = ("cpsat", "local_search", "perm_annealing")
BACKENDS = ("dwave", "dimod", "qiskit", "scipy", "ortools")


# --------------------------------------------------------------------------
# Must hold with no optional backend installed at all.
# --------------------------------------------------------------------------


def test_registry_exposes_the_benchmark_workspace_solvers():
    assert {"cpsat", "local-search", "sa-perm"} <= set(SOLVERS)


@pytest.mark.parametrize("module", NEW_MODULES)
def test_no_backend_is_imported_at_module_scope(module):
    """Same guard the QUBO workspace put on its own modules, for these three.

    Read with ``ast`` rather than by importing: on CI with --all-extras an
    import-based check passes no matter where the import actually sits.
    """
    source = pathlib.Path(dextrivia.solvers.__file__).parent / f"{module}.py"
    tree = ast.parse(source.read_text())

    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(BACKENDS), (
        f"{module} imports {imported & set(BACKENDS)} at top level"
    )


def test_cpsat_records_a_missing_backend_instead_of_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "ortools.sat.python", None)
    solution = CPSATSolver(time_limit_s=0.2).solve(random_instance(4, 0), seed=1)
    assert not solution.feasible
    assert "backend unavailable" in solution.metadata["reason"]


# --------------------------------------------------------------------------
# sa-perm: the control for sa-qubo.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_swap_delta_equals_a_full_recompute(time_dependent):
    """The O(1) delta is the whole solver. If it drifts, the anneal is fiction.

    Time-slotted costs are the interesting case: a swap moves objects between
    positions but leaves every leg's *slot* where it was, which is why four legs
    is the complete list of what can change.
    """
    instance = random_instance(6, 3, time_dependent=time_dependent)
    legs = leg_cost_stack(instance)
    rng = np.random.default_rng(0)
    states = np.array([rng.permutation(instance.n) for _ in range(200)])
    p, q = _propose(rng, len(states), instance.n)

    delta = _swap_delta(legs, states, p, q)

    for chain, (before, left, right) in enumerate(zip(states, p, q, strict=True)):
        after = list(before)
        after[left], after[right] = after[right], after[left]
        expected = instance.path_cost(after) - instance.path_cost(before)
        assert delta[chain] == pytest.approx(expected, abs=1e-12)


def test_leg_cost_stack_agrees_with_the_accessor_on_a_static_instance():
    """Static costs are one matrix repeated, not N-1 different ones."""
    instance = random_instance(5, 1)
    legs = leg_cost_stack(instance)
    assert legs.shape == (instance.n - 1, instance.n, instance.n)
    for step in range(instance.n - 1):
        assert np.array_equal(legs[step], instance.leg_costs(step))


def test_proposed_move_count_matches_the_documented_matched_budget():
    """The control is only a control if the budget claim is checkable."""
    instance = random_instance(4, 2)
    reads, sweeps = 8, 5
    _, _, schedule = anneal_permutations(instance, seed=1, num_reads=reads, num_sweeps=sweeps)
    assert schedule["moves_per_sweep"] == instance.n**2
    assert schedule["proposed_moves"] == reads * sweeps * instance.n**2


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_perm_annealing_is_always_feasible_and_reproducible(time_dependent):
    instance = random_instance(5, 4, time_dependent=time_dependent)
    solver = PermutationAnnealingSolver(num_reads=16, num_sweeps=20)

    first = solver.solve(instance, seed=7)
    second = solver.solve(instance, seed=7)

    assert instance.is_permutation(first.sequence)
    assert first.metadata["feasibility_rate"] == 1.0
    assert first.metadata["feasibility_is_structural"] is True
    assert first.total_dv_kms == pytest.approx(second.total_dv_kms)
    assert first.total_dv_kms == pytest.approx(instance.path_cost(first.sequence))


# --------------------------------------------------------------------------
# local-search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_local_search_never_returns_worse_than_greedy(time_dependent):
    """It descends from every greedy start, so it cannot lose to greedy."""
    instance = random_instance(7, 5, time_dependent=time_dependent)
    greedy = GreedySolver().solve(instance)
    found = LocalSearchSolver(restarts=3).solve(instance, seed=1)

    assert found.feasible
    assert instance.is_permutation(found.sequence)
    assert found.total_dv_kms <= greedy.total_dv_kms + 1e-12
    assert found.total_dv_kms == pytest.approx(instance.path_cost(found.sequence))


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_local_search_reaches_the_optimum_on_small_instances(time_dependent):
    instance = random_instance(6, 8, time_dependent=time_dependent)
    _, optimum = held_karp(instance)
    found = LocalSearchSolver(restarts=10).solve(instance, seed=2)
    assert found.total_dv_kms == pytest.approx(optimum, rel=1e-9)


# --------------------------------------------------------------------------
# cpsat, validated against the oracle.
# --------------------------------------------------------------------------


@needs_ortools
@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_cpsat_matches_held_karp(n, time_dependent):
    """The reference model and the exact DP must agree on the optimal value.

    Costs are rounded to mm/s for CP-SAT's int64 objective, so the comparison is
    on the delta-v recomputed from the unrounded instance, with a tolerance that
    is the rounding and nothing else.
    """
    instance = random_instance(n, n + 40, time_dependent=time_dependent)
    _, optimum = held_karp(instance)

    solution = CPSATSolver(time_limit_s=20.0).solve(instance, seed=1)

    assert solution.feasible
    assert solution.metadata["proven_optimal"] is True
    assert instance.is_permutation(solution.sequence)
    assert solution.total_dv_kms == pytest.approx(optimum, abs=1e-5)


@needs_ortools
def test_cpsat_is_never_worse_than_its_greedy_warm_start():
    """A dropped hint is invisible except as a suspiciously bad reference row.

    CP-SAT silently discards a *partial* hint, which is how this was found: the
    model then returned 2.7875 km/s on n20_static where greedy alone gets
    2.4485.
    """
    instance = random_instance(9, 17, time_dependent=True)
    greedy = GreedySolver().solve(instance)
    solution = CPSATSolver(time_limit_s=5.0).solve(instance, seed=1)

    assert solution.feasible
    assert solution.metadata["warm_start"] == "greedy"
    assert solution.total_dv_kms <= greedy.total_dv_kms + 1e-9


@needs_ortools
def test_cpsat_reports_a_lower_bound_and_whether_it_proved_optimality():
    """A time-limited answer is a bound, not an optimum, and says so."""
    instance = random_instance(5, 3)
    solution = CPSATSolver(time_limit_s=20.0).solve(instance, seed=1)
    assert solution.metadata["proven_optimal"] is True
    assert solution.metadata["best_objective_bound_kms"] == pytest.approx(
        solution.total_dv_kms, abs=1e-5
    )


@needs_ortools
@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_cpsat_accepts_time_slotted_costs_where_routing_refuses(time_dependent):
    """The reason this solver exists: ortools routing returns a miss here."""
    instance = random_instance(5, 9, time_dependent=time_dependent)
    routing = SOLVERS["ortools"](time_limit_s=1.0).solve(instance, seed=1)
    cpsat = CPSATSolver(time_limit_s=10.0).solve(instance, seed=1)

    assert cpsat.feasible
    assert cpsat.metadata["handles_time_dependent_costs"] is True
    if time_dependent:
        assert not routing.feasible
