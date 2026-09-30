"""``qaoa-swap``: the constraint-preserving mixer and its exact subspace simulator.

The first block is numpy-only and runs on a core install; it checks the
algebra the subspace simulator rests on (docs/qubo.md §10). The second needs
qiskit and is the final word: the synthesised Qiskit circuit and the subspace
simulator must agree amplitude by amplitude, and nothing may leak out of the
feasible subspace.
"""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np
import pytest

from dextrivia.qubo.permutation_qaoa import PermutationSpace, color_partition, mixer_schedule
from dextrivia.solvers import SOLVERS
from dextrivia.solvers.quantum_qaoa_swap import QAOASwapSolver
from tests.test_qubo_formulation import random_instance
from tests.test_qubo_solvers import _installed, needs_qiskit

SHAPES = [False, True]  # static, time-slotted


# --------------------------------------------------------------------------
# numpy only
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_color_partition_is_a_proper_edge_colouring_of_k_n(n):
    groups = color_partition(n)
    pairs = [pair for group in groups for pair in group]
    assert sorted(pairs) == list(itertools.combinations(range(n), 2))
    for group in groups:
        objects = [obj for pair in group for obj in pair]
        assert len(objects) == len(set(objects)), "a colour shares an object"
    assert len(groups) == (n - 1 if n % 2 == 0 else n)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_mixer_schedule_has_every_adjacent_partial_mixer_exactly_once(n):
    schedule = mixer_schedule(n)
    expected = {(i, pair) for i in range(n - 1) for pair in itertools.combinations(range(n), 2)}
    assert len(schedule) == len(expected) == (n - 1) * math.comb(n, 2)
    assert set(schedule) == expected
    # Parity-major: every even position before any odd one.
    parities = [i % 2 for i, _ in schedule]
    assert parities == sorted(parities)


@pytest.mark.parametrize("time_dependent", SHAPES)
def test_subspace_costs_and_basis_match_the_instance(time_dependent):
    instance = random_instance(5, 3, time_dependent=time_dependent)
    space = PermutationSpace(instance)
    assert [tuple(p) for p in space.perms] == list(itertools.permutations(range(5)))
    for k in (0, 17, 64, 119):
        assert space.costs[k] == pytest.approx(instance.path_cost(tuple(space.perms[k])))


def test_partners_swap_exactly_positions_i_and_i_plus_1_and_are_involutions():
    space = PermutationSpace(random_instance(5, 0))
    for i, partner in enumerate(space.partners):
        assert np.array_equal(partner[partner], np.arange(space.dim))
        a, b = space.perms, space.perms[partner]
        assert np.array_equal(a[:, i], b[:, i + 1]) and np.array_equal(a[:, i + 1], b[:, i])
        others = [p for p in range(5) if p not in (i, i + 1)]
        assert np.array_equal(a[:, others], b[:, others])


@pytest.mark.parametrize("n", [3, 4, 5])
def test_partial_mixers_at_one_position_multiply_to_zero_on_the_feasible_subspace(n):
    """The claim that lets the circuit's per-object-pair terms be regrouped.

    H_PS,i,{u,v} and H_PS,i,{u,w} share qubit (u, i) and do not commute on the
    full Hilbert space. Restricted to permutations, each is non-zero only on
    states with exactly {u,v} (resp. {u,w}) at positions (i, i+1); exactly one
    pair sits there in any permutation, so every product of two distinct ones
    vanishes, they commute, and their sum is the value-independent pairing.
    """
    space = PermutationSpace(random_instance(n, 1))
    for i in range(n - 1):
        restricted = {}
        for u, v in itertools.combinations(range(n), 2):
            h = np.zeros((space.dim, space.dim))
            occupied = {tuple(sorted(pair)) for pair in space.perms[:, [i, i + 1]].tolist()}
            assert (u, v) in occupied
            for k, perm in enumerate(space.perms):
                if {int(perm[i]), int(perm[i + 1])} == {u, v}:
                    h[space.partners[i][k], k] = 1.0
            restricted[(u, v)] = h
        for (a, ha), (b, hb) in itertools.combinations(restricted.items(), 2):
            assert not np.any(ha @ hb), (i, a, b)
        total = sum(restricted.values())
        pairing = np.zeros((space.dim, space.dim))
        pairing[space.partners[i], np.arange(space.dim)] = 1.0
        assert np.array_equal(total, pairing)
        assert np.array_equal(total @ total, np.eye(space.dim))


def test_evolution_is_unitary_and_beta_zero_only_adds_phases():
    space = PermutationSpace(random_instance(5, 2, time_dependent=True))
    psi = space.evolve([0.7, 1.9], [0.3, 1.1], space.uniform_state())
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)
    phases_only = space.evolve([0.7, 1.9], [0.0, 0.0], space.uniform_state())
    assert np.allclose(np.abs(phases_only) ** 2, 1.0 / space.dim)


def test_registry_exposes_qaoa_swap():
    assert SOLVERS["qaoa-swap"] is QAOASwapSolver


@pytest.mark.parametrize(
    ("solver", "n", "needle"),
    [
        (QAOASwapSolver(backend="subspace"), 10, "subspace limit of N=9"),
        (QAOASwapSolver(backend="statevector"), 5, "25 qubits"),
    ],
)
def test_refuses_past_its_limit_without_raising(solver, n, needle):
    solution = solver.solve(random_instance(n, 0))
    assert not solution.feasible
    assert needle in solution.metadata["reason"]
    assert solution.sequence == ()


@pytest.mark.parametrize("missing", ["scipy.optimize", "qiskit.quantum_info"])
def test_missing_backend_is_recorded_not_raised(monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    backend = "statevector" if missing.startswith("qiskit") else "subspace"
    solution = QAOASwapSolver(reps=1, maxiter=5, restarts=1, backend=backend).solve(
        random_instance(3, 0), seed=1
    )
    assert not solution.feasible
    assert "backend unavailable" in solution.metadata["reason"]


def test_bad_configuration_is_a_programming_error():
    with pytest.raises(ValueError):
        QAOASwapSolver(backend="aer")
    with pytest.raises(ValueError):
        QAOASwapSolver(initial_state="dicke")


# --------------------------------------------------------------------------
# Needs scipy (the optimiser) -- the subspace backend itself is numpy
# --------------------------------------------------------------------------

needs_scipy = pytest.mark.skipif(
    not _installed("scipy"), reason="quantum extra not installed (scipy)"
)


@needs_scipy
@pytest.mark.parametrize("time_dependent", SHAPES)
def test_subspace_solver_contract(time_dependent):
    instance = random_instance(5, 4, time_dependent=time_dependent)
    solution = QAOASwapSolver(reps=2, maxiter=60, restarts=2, shots=1000).solve(instance, seed=3)
    assert solution.feasible
    assert sorted(solution.sequence) == list(range(5))
    assert solution.total_dv_kms == pytest.approx(instance.path_cost(solution.sequence))
    meta = solution.metadata
    assert meta["feasibility_rate"] == 1.0 and meta["feasibility_measured"] is False
    assert meta["num_samples"] == 1000
    assert meta["runtime_kind"] == "classical subspace simulation time"
    assert 0.0 <= meta["exact_optimal_probability"] <= 1.0
    assert meta["final_expectation"] == pytest.approx(meta["exact_mean_dv_kms"])


@needs_scipy
def test_same_seed_same_answer():
    instance = random_instance(4, 5, time_dependent=True)
    a = QAOASwapSolver(reps=1, maxiter=40, restarts=2).solve(instance, seed=9)
    b = QAOASwapSolver(reps=1, maxiter=40, restarts=2).solve(instance, seed=9)
    assert a.metadata["gammas"] == b.metadata["gammas"]
    assert a.metadata["best_raw_sample_count"] == b.metadata["best_raw_sample_count"]


# --------------------------------------------------------------------------
# Needs qiskit: the Qiskit circuit is the ground truth
# --------------------------------------------------------------------------


@needs_qiskit
@pytest.mark.parametrize("n", [3, 4])
def test_every_pauli_string_in_a_swap_term_commutes(n):
    """So exponentiating one term is exact, not a Trotter approximation."""
    from dextrivia.solvers.quantum_qaoa_swap import swap_mixer_term

    for i, (u, v) in mixer_schedule(n):
        term = swap_mixer_term(n, i, u, v)
        paulis = term.paulis
        assert len(paulis) == 8
        assert np.allclose(np.abs(term.coeffs), 1 / 8) and np.allclose(term.coeffs.imag, 0)
        for a, b in itertools.combinations(range(8), 2):
            assert paulis[a].commutes(paulis[b])


@needs_qiskit
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("time_dependent", SHAPES)
@pytest.mark.parametrize("start", ["uniform", "basis"])
def test_qiskit_circuit_matches_subspace_simulation_amplitude_by_amplitude(
    n, time_dependent, start
):
    """The final word on the subspace simulator, and on feasibility preservation.

    The circuit is synthesised to cx/rz/sx/x first, so this checks the gate
    sequence (per-object-pair partial mixers, Qiskit's Pauli-evolution
    synthesis, the Ising cost with its identity term) and not a matrix
    exponential of the ideal operator.
    """
    from qiskit.quantum_info import Statevector

    from dextrivia.solvers.quantum_qaoa_swap import swap_qaoa_circuit, synthesize

    instance = random_instance(n, 7 + n, time_dependent=time_dependent)
    space = PermutationSpace(instance)
    rng = np.random.default_rng(n)
    reps = 2
    gammas, betas = rng.uniform(0, 3, reps), rng.uniform(0, 1.5, reps)
    if start == "uniform":
        psi0 = space.uniform_state()
        circuit, g, b = swap_qaoa_circuit(instance, reps)
        initial = Statevector(space.embed(psi0))
    else:
        sequence = (n - 1, *range(n - 1))
        psi0 = space.basis_state(sequence)
        circuit, g, b = swap_qaoa_circuit(instance, reps, initial_sequence=sequence)
        initial = Statevector.from_label("0" * n * n)
    bound = synthesize(circuit).assign_parameters(
        {**dict(zip(g, gammas, strict=True)), **dict(zip(b, betas, strict=True))}
    )

    full = initial.evolve(bound).data
    expected = space.embed(space.evolve(gammas, betas, psi0))

    assert np.max(np.abs(full - expected)) < 1e-8
    leaked = 1.0 - np.sum(np.abs(full[space.bitstring_indices()]) ** 2)
    assert abs(leaked) < 1e-10


@needs_qiskit
@pytest.mark.parametrize("time_dependent", SHAPES)
def test_statevector_backend_measures_feasibility_of_one(time_dependent):
    """Feasibility measured from sampled bitstrings of the real circuit, N=3."""
    instance = random_instance(3, 2, time_dependent=time_dependent)
    solution = QAOASwapSolver(
        reps=1, maxiter=25, restarts=1, shots=2048, backend="statevector"
    ).solve(instance, seed=4)
    assert solution.feasible
    meta = solution.metadata
    assert meta["feasibility_measured"] is True
    assert meta["feasibility_rate"] == 1.0
    assert meta["leaked_probability"] < 1e-10
    assert meta["runtime_kind"] == "classical statevector simulation time"
