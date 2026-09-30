"""Lower bounds, the solvers that certify them, and the headroom probes.

The rule every test here enforces: a lower bound that ever exceeds a known
optimum is not a weak bound, it is a wrong one, and it would silently turn a
best-known reference into a false "proven". So bounds are checked against
Held-Karp on every committed instance Held-Karp can reach, both cost shapes.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from dextrivia.core import ProblemInstance
from dextrivia.snapshots import Snapshot, default_snapshot_dir
from dextrivia.solvers import (
    SOLVERS,
    ColdPermutationAnnealingSolver,
    CPSATSolver,
    ExactSolver,
    GreedySolver,
    HiGHSSolver,
    IteratedLocalSearchSolver,
    LocalSearchSolver,
)
from dextrivia.solvers.permutation import batch_costs, double_bridge, neighbourhood
from tests.test_classical_baselines import random_instance


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


needs_scipy = pytest.mark.skipif(not _installed("scipy"), reason="quantum extra (scipy)")
needs_ortools = pytest.mark.skipif(not _installed("ortools.sat.python"), reason="ortools")

INSTANCES = default_snapshot_dir().parent / "instances"
HELD_KARP_REACH = sorted(
    p
    for p in [*INSTANCES.glob("*planecluster-v1*.npz"), *INSTANCES.glob("planecluster-v2/*.npz")]
    if ProblemInstance.load(p).n <= 15
)


def brute_optimum(instance: ProblemInstance) -> float:
    return min(instance.path_cost(p) for p in itertools.permutations(range(instance.n)))


# --------------------------------------------------------------------------
# Bounds never exceed the optimum.
# --------------------------------------------------------------------------


def test_every_committed_instance_within_held_karp_reach_is_covered():
    assert len(HELD_KARP_REACH) >= 25  # 10 from v1, 19 from v2


@needs_scipy
@pytest.mark.parametrize("path", HELD_KARP_REACH, ids=lambda p: Path(p).stem[-24:])
def test_bounds_never_exceed_held_karp_on_committed_instances(path):
    from dextrivia.solvers.highs_mip import lp_lower_bound

    instance = ProblemInstance.load(path)
    optimum = ExactSolver().solve(instance).total_dv_kms
    bound, _, stats = lp_lower_bound(instance)
    assert stats["lp_plain_bound_kms"] <= bound + 1e-9  # cuts only tighten
    assert bound <= optimum

    solution = HiGHSSolver().solve(instance)
    assert solution.metadata["lower_bound_kms"] <= optimum
    # Within Held-Karp reach highs should also *reach* the optimum.
    assert solution.total_dv_kms == pytest.approx(optimum, abs=1e-9)
    assert solution.metadata["proven_optimal"]


@needs_scipy
@pytest.mark.parametrize("time_dependent", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_bounds_never_exceed_brute_force_on_random_instances(seed, time_dependent):
    """Random costs are the adversarial case: no metric structure for cuts to lean on."""
    from dextrivia.solvers.highs_mip import lp_lower_bound

    instance = random_instance(7, seed, time_dependent=time_dependent)
    optimum = brute_optimum(instance)
    bound, _, _ = lp_lower_bound(instance)
    assert bound <= optimum
    solution = HiGHSSolver().solve(instance)
    assert solution.metadata["lower_bound_kms"] <= optimum
    assert solution.total_dv_kms == pytest.approx(optimum, abs=1e-9)


@needs_scipy
def test_highs_reads_costs_by_slot_not_by_matrix():
    """A time-slotted instance whose slots disagree: pricing slot 0 everywhere is wrong."""
    instance = random_instance(6, 11, time_dependent=True)
    static = ProblemInstance(
        norad_ids=instance.norad_ids,
        names=instance.names,
        epoch=instance.epoch,
        costs=instance.leg_costs(0),
    )
    assert brute_optimum(instance) != pytest.approx(brute_optimum(static))
    assert HiGHSSolver().solve(instance).total_dv_kms == pytest.approx(brute_optimum(instance))


def test_highs_refuses_rather_than_raises_past_its_limit():
    solution = HiGHSSolver(max_n=5).solve(random_instance(6, 1))
    assert not solution.feasible
    assert "exceeds highs limit" in solution.metadata["reason"]


def test_highs_is_registered_as_deterministic():
    from dextrivia.solvers import DETERMINISTIC

    assert "highs" in SOLVERS and "highs" in DETERMINISTIC


# --------------------------------------------------------------------------
# CP-SAT: warm start and a bound that survives the integer rounding.
# --------------------------------------------------------------------------


@needs_ortools
@pytest.mark.parametrize("time_dependent", [False, True])
def test_cpsat_bound_is_below_the_optimum_after_the_rounding_margin(time_dependent):
    instance = random_instance(7, 4, time_dependent=time_dependent)
    optimum = brute_optimum(instance)
    solution = CPSATSolver(time_limit_s=10).solve(instance, seed=1)
    meta = solution.metadata
    assert meta["lower_bound_kms"] <= optimum
    assert meta["lower_bound_kms"] == pytest.approx(meta["raw_bound_kms"] - 6 * 0.5e-6)


@needs_ortools
def test_cpsat_warm_start_is_never_worse_than_localsearch():
    instance = random_instance(9, 2, time_dependent=True)
    local = LocalSearchSolver().solve(instance).total_dv_kms
    solution = CPSATSolver(time_limit_s=2).solve(instance, seed=1)
    assert solution.metadata["warm_start"] == "localsearch"
    assert solution.metadata["beats_or_ties_hint"]
    assert solution.total_dv_kms <= local + 1e-9


# --------------------------------------------------------------------------
# ILS: correct on slotted costs, never worse than where it started.
# --------------------------------------------------------------------------


def test_neighbourhood_rows_are_permutations_and_include_every_2opt_move():
    moves = neighbourhood(6)
    assert all(sorted(row) == list(range(6)) for row in moves)
    reversal = np.array([0, 4, 3, 2, 1, 5])
    assert any((row == reversal).all() for row in moves)
    assert not any((row == np.arange(6)).all() for row in moves)  # no null move


@pytest.mark.parametrize("time_dependent", [False, True])
def test_batch_costs_match_path_cost(time_dependent):
    instance = random_instance(7, 3, time_dependent=time_dependent)
    candidates = np.array(list(itertools.permutations(range(7)))[:200])
    expected = [instance.path_cost(row) for row in candidates]
    got = batch_costs(instance.costs, time_dependent, candidates)
    assert got == pytest.approx(expected)


def test_double_bridge_is_a_permutation_and_moves_something():
    rng = np.random.default_rng(0)
    seq = np.arange(10)
    kicked = double_bridge(seq, rng)
    assert sorted(kicked) == list(range(10))
    assert not (kicked == seq).all()


@pytest.mark.parametrize("time_dependent", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_ils_finds_the_brute_force_optimum_at_small_n(seed, time_dependent):
    instance = random_instance(8, seed, time_dependent=time_dependent)
    solution = IteratedLocalSearchSolver().solve(instance, seed=seed)  # default budget
    assert solution.total_dv_kms == pytest.approx(brute_optimum(instance))
    assert solution.total_dv_kms == pytest.approx(instance.path_cost(solution.sequence))


def test_ils_is_never_worse_than_localsearch_and_is_reproducible():
    instance = random_instance(20, 5, time_dependent=True)
    local = LocalSearchSolver().solve(instance).total_dv_kms
    a = IteratedLocalSearchSolver(kicks=50).solve(instance, seed=3)
    b = IteratedLocalSearchSolver(kicks=50).solve(instance, seed=3)
    assert a.total_dv_kms <= local + 1e-12
    assert a.sequence == b.sequence
    assert a.metadata["kicks_run"] == 50 and not a.metadata["time_cap_hit"]


# --------------------------------------------------------------------------
# sa-perm-cold: the control for sa-perm's head start.
# --------------------------------------------------------------------------


def test_cold_sa_perm_never_sees_the_greedy_sequence():
    instance = random_instance(8, 1)
    solution = ColdPermutationAnnealingSolver(time_limit_s=0.1, restarts=1).solve(instance, 1)
    assert solution.solver_name == "sa-perm-cold"
    assert solution.metadata["start_from_greedy"] is False
    assert solution.metadata["greedy_start_dv_kms"] is None
    assert SOLVERS["sa-perm-cold"]().start_from_greedy is False


# --------------------------------------------------------------------------
# The pinned N=30 case: the instance the "is greedy optimal?" question came from.
# --------------------------------------------------------------------------


@needs_scipy
def test_pinned_n30_april_40deg_window_greedy_is_provably_optimal():
    """Built with the parameters the question was asked with; highs settles it.

    iridium33_20260402, ImpulsiveCostModel() static, ClusterWindow(raan 40 deg,
    alt 400 km), N=30. Greedy = localsearch = 3.9970 km/s, and it is optimal.
    RAAN order is NOT optimal here (+1.3%): with a 40 deg window and a 400 km
    band there is some structure off the RAAN axis. It still does not help any
    solver -- greedy already sits on the optimum.
    """
    from dextrivia.costs.realistic import ImpulsiveCostModel, mean_elements
    from dextrivia.costs.selection import ClusterWindow, build_cluster_instance

    snapshot = Snapshot.load(default_snapshot_dir() / "iridium33_20260402.json")
    instance = build_cluster_instance(
        snapshot,
        30,
        ImpulsiveCostModel(),
        window=ClusterWindow(raan_window_deg=40, alt_band_km=400),
    )
    greedy = GreedySolver().solve(instance).total_dv_kms
    assert greedy == pytest.approx(3.99704, abs=1e-5)

    certified = HiGHSSolver().solve(instance)
    assert certified.metadata["proven_optimal"]
    assert certified.metadata["lower_bound_kms"] <= greedy
    assert greedy - certified.metadata["lower_bound_kms"] < 1e-5

    by_id = {o.norad_id: o for o in snapshot.objects}
    raan = np.unwrap(
        [
            mean_elements(by_id[i].line1, by_id[i].line2, instance.epoch).raan_rad
            for i in instance.norad_ids
        ]
    )
    order = np.argsort(raan)
    raan_cost = min(instance.path_cost(order), instance.path_cost(order[::-1]))
    assert raan_cost / greedy - 1 == pytest.approx(0.0132, abs=0.001)


@needs_scipy
def test_static_v2_is_one_dimensional_raan_order_meets_the_certified_bound():
    """Locks docs/physics.md section 11, as test_degeneracy.py locks the altitude case.

    Visiting v2 n20_static in RAAN order -- a sort -- is provably optimal. If a
    cost-model change ever breaks this, the finding needs rewriting, not the test
    deleting.
    """
    from dextrivia.costs.realistic import mean_elements
    from dextrivia.solvers.highs_mip import lp_lower_bound

    path = INSTANCES / "planecluster-v2" / "iridium33_20260928_planecluster-v2_n20_static.npz"
    instance = ProblemInstance.load(path)
    snapshot = Snapshot.load(default_snapshot_dir() / instance.metadata["snapshot"])
    by_id = {o.norad_id: o for o in snapshot.objects}
    raan = np.unwrap(
        [
            mean_elements(by_id[i].line1, by_id[i].line2, instance.epoch).raan_rad
            for i in instance.norad_ids
        ]
    )
    order = np.argsort(raan)
    raan_cost = min(instance.path_cost(order), instance.path_cost(order[::-1]))
    bound, _, _ = lp_lower_bound(instance)
    assert raan_cost - bound < 1e-5


def test_every_solver_records_a_miss_rather_than_raising_at_n60():
    """The v3 family reaches N=60; a size check that overflows is a crash, not a miss.

    Found by the pre-canonical smoke run: qaoa's statevector-size message
    computed 16 * 2**3600 and raised OverflowError out of the bench.
    """
    instance = random_instance(60, 1)
    for name in ("qaoa", "qaoa-swap", "sa-qubo", "exact", "brute"):
        solution = SOLVERS[name]().solve(instance, seed=1)
        assert not solution.feasible, name
        assert solution.metadata["reason"], name
