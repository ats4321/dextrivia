"""The QUBO formulation, checked exhaustively at sizes where that is possible.

None of this imports an optional backend: the formulation is the part of the
QUBO workspace that must stay verified on a core install, because every solver
built on it inherits whatever is wrong here.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import numpy as np
import pytest

from dextrivia.core import ProblemInstance
from dextrivia.qubo import (
    build_qubo,
    decode,
    default_penalty,
    is_feasible,
    path_upper_bound,
    repair,
    summarize_samples,
    sweep_penalty,
)
from dextrivia.solvers import ExactSolver

EPOCH = datetime(2026, 4, 2, tzinfo=UTC)

#: 2**(N**2) assignments: 512 at N=3, 65536 at N=4. N=5 would be 33 million.
EXHAUSTIVE_N = (3, 4)


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


def permutation_sample(sequence, n: int) -> np.ndarray:
    """The 0/1 vector encoding ``sequence`` as a permutation matrix x[i, p]."""
    matrix = np.zeros((n, n), dtype=int)
    for position, obj in enumerate(sequence):
        matrix[obj, position] = 1
    return matrix.reshape(-1)


def all_assignments(n: int):
    for bits in itertools.product((0, 1), repeat=n * n):
        yield np.array(bits, dtype=int)


TIME_DEPENDENT = [False, True]


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("n", [3, 4, 6])
def test_feasible_energy_is_the_path_cost_plus_the_known_offset(n, time_dependent):
    """The whole formulation rests on this: energy IS delta-v on feasible points."""
    instance = random_instance(n, 11, time_dependent=time_dependent)
    qubo = build_qubo(instance)
    for sequence in itertools.permutations(range(n)):
        x = permutation_sample(sequence, n)
        raw = float(x @ qubo.Q @ x)
        assert raw + qubo.offset == pytest.approx(instance.path_cost(sequence))
        assert qubo.energy(x) == pytest.approx(instance.path_cost(sequence))
    assert qubo.offset == pytest.approx(2.0 * qubo.penalty * n)


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_objective_counts_n_minus_1_legs_not_n(time_dependent):
    """An open path has no return leg. A tour formulation would charge for one."""
    n = 5
    instance = random_instance(n, 3, time_dependent=time_dependent)
    qubo = build_qubo(instance, penalty=1.0)
    sequence = tuple(range(n))
    legs = [instance.leg_cost(step, sequence[step], sequence[step + 1]) for step in range(n - 1)]
    assert len(legs) == n - 1
    assert qubo.energy(permutation_sample(sequence, n)) == pytest.approx(sum(legs))


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("n", EXHAUSTIVE_N)
def test_every_infeasible_assignment_scores_above_the_best_feasible_one(n, time_dependent):
    """The penalty bound, checked by enumerating all 2**(N**2) assignments."""
    instance = random_instance(n, 5, time_dependent=time_dependent)
    qubo = build_qubo(instance)
    best_feasible = min(instance.path_cost(p) for p in itertools.permutations(range(n)))
    for x in all_assignments(n):
        if not is_feasible(x, n):
            assert qubo.energy(x) > best_feasible


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("n", EXHAUSTIVE_N)
def test_qubo_minimum_decodes_to_the_held_karp_optimum(n, time_dependent):
    """Minimising the QUBO must be the same thing as solving the problem."""
    instance = random_instance(n, 5, time_dependent=time_dependent)
    qubo = build_qubo(instance)
    best_x = min(all_assignments(n), key=qubo.energy)
    sequence = decode(best_x, n)
    optimum = ExactSolver().solve(instance)
    assert sequence is not None
    assert instance.path_cost(sequence) == pytest.approx(optimum.total_dv_kms)


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
@pytest.mark.parametrize("n", EXHAUSTIVE_N)
def test_ising_form_reproduces_every_qubo_energy(n, time_dependent):
    """QAOA optimises the Ising form; a sign slip there is invisible downstream."""
    instance = random_instance(n, 2, time_dependent=time_dependent)
    qubo = build_qubo(instance)
    h, j, const = qubo.to_ising()
    for x in all_assignments(n):
        z = 1 - 2 * x  # x=0 -> z=+1, x=1 -> z=-1
        energy = const + float(h @ z) + sum(w * z[u] * z[v] for (u, v), w in j.items())
        assert energy == pytest.approx(qubo.energy(x))


def test_penalty_below_the_threshold_can_be_beaten_by_an_infeasible_assignment():
    """Guards the bound: if no penalty were ever too small, it would say nothing."""
    n = 4
    instance = random_instance(n, 5)
    best_feasible = min(instance.path_cost(p) for p in itertools.permutations(range(n)))
    # Far under the threshold: leaving positions empty now costs less than flying.
    qubo = build_qubo(instance, penalty=1e-6)
    assert min(qubo.energy(x) for x in all_assignments(n) if not is_feasible(x, n)) < best_feasible


def test_default_penalty_clears_the_provable_threshold():
    instance = random_instance(6, 1)
    assert default_penalty(instance) > path_upper_bound(instance)
    assert path_upper_bound(instance) >= ExactSolver().solve(instance).total_dv_kms


def test_default_penalty_rejects_a_safety_factor_that_breaks_the_bound():
    with pytest.raises(ValueError, match="safety"):
        default_penalty(random_instance(4, 0), safety=1.0)


def test_build_rejects_negative_costs_that_would_void_the_bound():
    instance = random_instance(4, 0)
    negative = instance.costs.copy()
    negative[0, 1] = -1.0
    broken = ProblemInstance(
        instance.norad_ids, instance.names, instance.epoch, negative, instance.metadata
    )
    with pytest.raises(ValueError, match="non-negative"):
        build_qubo(broken)


def test_to_qubo_dict_matches_the_matrix():
    qubo = build_qubo(random_instance(4, 8))
    as_dict = qubo.to_qubo_dict()
    for x in (np.zeros(16, int), np.ones(16, int), permutation_sample((2, 0, 3, 1), 4)):
        assert sum(w * x[u] * x[v] for (u, v), w in as_dict.items()) == pytest.approx(
            float(x @ qubo.Q @ x)
        )


@pytest.mark.parametrize("n", [3, 5])
def test_decode_and_repair_round_trip(n):
    sequence = tuple(reversed(range(n)))
    x = permutation_sample(sequence, n)
    assert is_feasible(x, n)
    assert decode(x, n) == sequence
    assert repair(x, n) == sequence  # repairing something valid must not change it


@pytest.mark.parametrize("n", [3, 5])
def test_repair_always_returns_a_permutation(n):
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 2, size=n * n)
        sequence = repair(x, n)
        assert sorted(sequence) == list(range(n))
        if not is_feasible(x, n):
            assert decode(x, n) is None


def test_summarize_reports_raw_feasibility_separately_from_repaired():
    n = 4
    instance = random_instance(n, 4)
    qubo = build_qubo(instance)
    good = permutation_sample((0, 1, 2, 3), n)
    junk = np.zeros(n * n, dtype=int)  # violates every constraint at once
    summary = summarize_samples(instance, qubo, np.array([good, junk, junk]))

    assert summary["num_samples"] == 3
    assert summary["raw_feasible_samples"] == 1
    assert summary["feasibility_rate"] == pytest.approx(1 / 3)
    assert summary["best_raw_dv_kms"] == pytest.approx(instance.path_cost((0, 1, 2, 3)))
    # The repaired best can only ever be as good as or better than the raw one,
    # which is exactly why the two are reported apart.
    assert summary["best_repaired_dv_kms"] <= summary["best_raw_dv_kms"] + 1e-12
    assert sorted(summary["best_repaired_sequence"]) == list(range(n))


def test_summarize_on_all_infeasible_samples_reports_no_raw_result():
    n = 3
    instance = random_instance(n, 4)
    summary = summarize_samples(instance, build_qubo(instance), np.zeros((5, n * n), dtype=int))
    assert summary["feasibility_rate"] == 0.0
    assert summary["best_raw_dv_kms"] is None
    assert summary["best_repaired_dv_kms"] is not None
    assert summary["best_was_raw_sample"] is False


@pytest.mark.parametrize("time_dependent", TIME_DEPENDENT)
def test_sweep_penalty_reports_which_weights_are_provably_sufficient(time_dependent):
    instance = random_instance(4, 6, time_dependent=time_dependent)
    rng = np.random.default_rng(0)

    def sample_fn(qubo):
        return rng.integers(0, 2, size=(20, qubo.num_variables))

    records = sweep_penalty(instance, sample_fn, factors=(0.5, 2.0))
    assert [r["penalty_factor"] for r in records] == [0.5, 2.0]
    assert [r["provably_sufficient"] for r in records] == [False, True]
    assert records[1]["penalty"] == pytest.approx(2.0 * path_upper_bound(instance))
    assert all(0.0 <= r["feasibility_rate"] <= 1.0 for r in records)


def test_variable_indexing_is_a_bijection_and_range_checked():
    qubo = build_qubo(random_instance(4, 0))
    assert sorted(qubo.var(i, p) for i in range(4) for p in range(4)) == list(range(16))
    with pytest.raises(IndexError):
        qubo.var(4, 0)
