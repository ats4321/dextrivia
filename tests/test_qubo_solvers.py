"""Solver contract for the QUBO workspace's three solvers.

Split in two on purpose. The "no backend" tests run everywhere, including on a
core install, because the thing most likely to break the benchmark is a solver
that raises instead of recording a miss. The rest skip when the ``quantum``
extra is not installed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

import dextrivia.solvers
from dextrivia.solvers import SOLVERS, ExactSolver, ORToolsRoutingSolver, QAOASolver
from dextrivia.solvers.quantum_annealing import SimulatedAnnealingSolver
from dextrivia.solvers.quantum_qaoa import statevector_bytes
from tests.test_qubo_formulation import random_instance


def _installed(module: str) -> bool:
    """True if ``module`` is importable.

    ``find_spec`` raises ModuleNotFoundError rather than returning None when a
    *parent* package is missing, which is exactly the case on a core install --
    so catching it is the whole point of this helper.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


needs_dwave = pytest.mark.skipif(
    not _installed("dwave.samplers"), reason="quantum extra not installed (dwave-samplers)"
)
needs_qiskit = pytest.mark.skipif(
    not _installed("qiskit"), reason="quantum extra not installed (qiskit)"
)
needs_ortools = pytest.mark.skipif(
    not _installed("ortools"), reason="quantum extra not installed (ortools)"
)

#: QAOA is a statevector simulation inside an optimiser loop, so tests use the
#: smallest instance the problem admits (N=3, 9 qubits) and one shallow restart.
FAST_QAOA = dict(reps=1, shots=512, maxiter=30, restarts=1)


# --------------------------------------------------------------------------
# These must hold with no optional backend present at all.
# --------------------------------------------------------------------------


def test_registry_exposes_the_qubo_workspace_solvers():
    assert {"sa-qubo", "qaoa", "ortools"} <= set(SOLVERS)


BACKENDS = ("dwave", "dimod", "qiskit", "scipy", "ortools")


@pytest.mark.parametrize(
    "module",
    ["quantum_annealing", "quantum_qaoa", "ortools_routing"],
)
def test_no_backend_is_imported_at_module_scope(module):
    """The CLI has to stay importable on a core install.

    Checked by reading the source rather than by importing, because on a machine
    that happens to have the extra installed -- like CI with --all-extras -- an
    import-based check would pass no matter where the import sits.
    """
    import ast

    source = pathlib.Path(dextrivia.solvers.__file__).parent / f"{module}.py"
    tree = ast.parse(source.read_text())

    top_level = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    imported = set()
    for node in top_level:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(BACKENDS), (
        f"{module} imports {imported & set(BACKENDS)} at top level"
    )


@pytest.mark.parametrize(
    ("solver", "missing"),
    [
        (SimulatedAnnealingSolver(num_reads=4), "dwave.samplers"),
        (QAOASolver(**FAST_QAOA), "qiskit.circuit.library"),
        (ORToolsRoutingSolver(time_limit_s=0.2), "ortools.constraint_solver"),
    ],
)
def test_missing_backend_is_recorded_not_raised(monkeypatch, solver, missing):
    """A benchmark needs to log the miss and carry on, not die on an ImportError."""
    monkeypatch.setitem(sys.modules, missing, None)
    solution = solver.solve(random_instance(3, 0), seed=1)
    assert not solution.feasible
    assert solution.sequence == ()
    assert "backend unavailable" in solution.metadata["reason"]
    assert solution.runtime_s >= 0.0


def test_qaoa_refuses_instances_past_the_qubit_wall():
    solution = QAOASolver().solve(random_instance(6, 0))
    assert not solution.feasible
    assert "36 qubits" in solution.metadata["reason"]
    assert solution.metadata["qubits_required"] == 36


def test_simulated_annealing_refuses_instances_past_its_limit():
    solution = SimulatedAnnealingSolver(max_n=4).solve(random_instance(6, 0))
    assert not solution.feasible
    assert "exceeds" in solution.metadata["reason"]


def test_ortools_refuses_time_slotted_costs_rather_than_silently_ignoring_them():
    """Routing arc costs are (from, to) only -- C[t,i,j] is not expressible."""
    solution = ORToolsRoutingSolver(time_limit_s=0.2).solve(
        random_instance(5, 0, time_dependent=True)
    )
    assert not solution.feasible
    assert "time-slotted" in solution.metadata["reason"]


def test_statevector_cost_is_reported_honestly():
    assert statevector_bytes(4) == 16 * 2**16
    assert statevector_bytes(5) == 16 * 2**25


# --------------------------------------------------------------------------
# Backend-dependent behaviour.
# --------------------------------------------------------------------------


@needs_dwave
@pytest.mark.parametrize("time_dependent", [False, True])
def test_simulated_annealing_returns_a_valid_path(time_dependent):
    instance = random_instance(6, 3, time_dependent=time_dependent)
    solution = SimulatedAnnealingSolver(num_reads=100).solve(instance, seed=7)
    assert solution.feasible
    assert instance.is_permutation(solution.sequence)
    assert instance.path_cost(solution.sequence) == pytest.approx(solution.total_dv_kms)
    assert solution.total_dv_kms >= ExactSolver().solve(instance).total_dv_kms - 1e-9


@needs_dwave
def test_simulated_annealing_is_reproducible_given_a_seed():
    instance = random_instance(6, 3)
    solver = SimulatedAnnealingSolver(num_reads=100)
    first = solver.solve(instance, seed=42)
    second = solver.solve(instance, seed=42)
    assert first.sequence == second.sequence
    assert first.total_dv_kms == pytest.approx(second.total_dv_kms)


@needs_dwave
def test_simulated_annealing_records_a_full_run_record():
    instance = random_instance(5, 1)
    solution = SimulatedAnnealingSolver(num_reads=50, num_sweeps=200).solve(instance, seed=3)
    metadata = solution.metadata
    assert 0.0 <= metadata["feasibility_rate"] <= 1.0
    assert metadata["num_reads"] == 50
    assert metadata["num_sweeps"] == 200
    assert metadata["seed"] == 3
    assert metadata["num_variables"] == 25
    assert metadata["versions"]["dwave-samplers"] != "not installed"
    assert solution.runtime_s > 0.0


@needs_qiskit
@pytest.mark.parametrize("time_dependent", [False, True])
def test_qaoa_returns_a_valid_path(time_dependent):
    instance = random_instance(3, 2, time_dependent=time_dependent)
    solution = QAOASolver(**FAST_QAOA).solve(instance, seed=5)
    assert solution.feasible
    assert instance.is_permutation(solution.sequence)
    assert instance.path_cost(solution.sequence) == pytest.approx(solution.total_dv_kms)
    assert solution.total_dv_kms >= ExactSolver().solve(instance).total_dv_kms - 1e-9


@needs_qiskit
def test_qaoa_records_shots_layers_and_versions():
    instance = random_instance(3, 2)
    solution = QAOASolver(**FAST_QAOA).solve(instance, seed=5)
    metadata = solution.metadata
    assert metadata["qubits"] == 9
    assert metadata["reps"] == 1
    assert metadata["shots"] == 512
    assert metadata["num_samples"] == 512
    assert metadata["objective_evaluations"] > 0
    assert len(metadata["restart_expectations"]) == 1
    assert metadata["seed"] == 5
    assert metadata["versions"]["qiskit"] != "not installed"
    # If Qiskit ever renames the cost-layer parameters, initialisation quietly
    # degrades; this is the tripwire that says so.
    assert metadata["gamma_parameters_detected"] is True


@needs_qiskit
def test_qaoa_hamiltonian_reproduces_the_qubo_energies():
    """The Ising operator QAOA minimises must be the QUBO, not a cousin of it."""
    import itertools

    from qiskit.quantum_info import Pauli

    from dextrivia.qubo import build_qubo
    from dextrivia.solvers.quantum_qaoa import cost_hamiltonian

    instance = random_instance(3, 9)
    qubo = build_qubo(instance)
    hamiltonian = cost_hamiltonian(qubo)

    for bits in itertools.islice(itertools.product((0, 1), repeat=9), 0, 512, 37):
        x = np.array(bits)
        # Diagonal Hamiltonian: the eigenvalue of a Z-string on |x> is the
        # product of (-1)**x over the qubits the string acts on.
        total = 0.0
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs, strict=True):
            acted_on = np.flatnonzero(Pauli(pauli).z)
            sign = (-1.0) ** int(x[acted_on].sum())
            total += float(coeff.real) * sign
        assert total == pytest.approx(qubo.energy(x), abs=1e-8)


@needs_ortools
def test_ortools_matches_the_exact_optimum_on_small_static_instances():
    for seed in range(4):
        instance = random_instance(8, seed)
        solution = ORToolsRoutingSolver(time_limit_s=1.0).solve(instance)
        assert solution.feasible
        assert instance.is_permutation(solution.sequence)
        assert solution.total_dv_kms == pytest.approx(
            ExactSolver().solve(instance).total_dv_kms, rel=1e-6
        )


@needs_ortools
def test_ortools_open_path_has_no_return_leg():
    """The dummy depot must vanish on decode; a tour would cost one leg more."""
    instance = random_instance(7, 1)
    solution = ORToolsRoutingSolver(time_limit_s=1.0).solve(instance)
    scaled = solution.metadata["scaled_objective"] / solution.metadata["cost_scale"]
    assert scaled == pytest.approx(solution.total_dv_kms, abs=1e-5)
    tour_cost = solution.total_dv_kms + instance.leg_cost(
        0, solution.sequence[-1], solution.sequence[0]
    )
    assert scaled < tour_cost


@needs_ortools
def test_ortools_scales_past_held_karp_range():
    instance = random_instance(25, 0)
    solution = ORToolsRoutingSolver(time_limit_s=1.0).solve(instance)
    assert solution.feasible
    assert instance.is_permutation(solution.sequence)
    assert not ExactSolver().solve(instance).feasible  # the oracle has already given up
