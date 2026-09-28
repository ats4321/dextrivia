"""``qaoa-swap``: QAOA with a constraint-preserving permutation-swap mixer.

The existing ``qaoa`` uses the penalty QUBO and the standard X mixer. At N=4 it
concentrates amplitude on permutation matrices but, within them, barely beats a
random permutation. This solver tests the textbook remedy (Hadfield et al.
2019, §5.1): never leave the feasible subspace at all. The ansatz, the proof
that it preserves feasibility and the measured answer are in ``docs/qubo.md``
§10; the math of the subspace is in ``dextrivia.qubo.permutation_qaoa``.

Two backends run the *same* circuit:

* ``statevector`` -- the actual Qiskit circuit on a dense statevector, N**2
  qubits. The ground truth, N <= 4. Measures feasibility from bitstrings.
* ``subspace`` -- exact simulation in the N!-dimensional feasible subspace
  (``PermutationSpace``). Agrees with the statevector amplitude by amplitude
  (``tests/test_qaoa_swap.py``) and reaches N=9. Feasibility here is 1.0 BY
  CONSTRUCTION, not by measurement, and the run record says so.

Two initial states, answering different questions -- never mix them in a table:

* ``uniform`` -- equal superposition over all N! permutations. At reps=0 this
  IS the uniform-random-permutation baseline, so any improvement is due to the
  layers. Prepared as an initial statevector, not a gate: synthesising it on
  N**2 qubits is not something hardware can run. The quality study uses this.
* ``basis`` -- one permutation, prepared with X gates. Hardware-runnable. Its
  reps=0 distribution is a single sequence, so its numbers are not comparable
  with the uniform start's.

Every runtime here is classical simulation time. No quantum hardware is used by
this solver; ``scripts/qaoa_hardware.py`` is the separate, labelled hardware run.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.qubo import library_versions, objective_qubo, path_upper_bound, summarize_samples
from dextrivia.qubo.permutation_qaoa import (
    PermutationSpace,
    mixer_schedule,
    permutation_space_bytes,
)

__all__ = [
    "QAOASwapSolver",
    "STATEVECTOR_MAX_N",
    "SUBSPACE_MAX_N",
    "swap_mixer_term",
    "swap_qaoa_circuit",
    "synthesize",
]

#: Same qubit wall as ``qaoa``: 16 qubits, 1 MB.
STATEVECTOR_MAX_N = 4
#: 9! = 362,880 amplitudes, ~20 MB. N=10 is 3.6 million and ~200 MB of index
#: arrays; it runs, but an optimiser loop over it is minutes per restart.
SUBSPACE_MAX_N = 9

BACKENDS = ("subspace", "statevector")
INITIAL_STATES = ("uniform", "basis")
_PACKAGES = ("qiskit", "scipy", "numpy")


def swap_mixer_term(n: int, i: int, u: int, v: int):
    """``H_PS,i,{u,v} = S+(u,i+1) S+(v,i) S-(u,i) S-(v,i+1) + h.c.`` as a SparsePauliOp.

    Swaps objects u and v between positions i and i+1 iff they occupy them.
    Expands to 8 Pauli strings of weight 4, each X or Y on all four qubits with
    an even number of Ys -- which is why they mutually commute and
    ``PauliEvolutionGate`` exponentiates the term exactly (docs/qubo.md §10).
    """
    from qiskit.quantum_info import SparsePauliOp

    nq = n * n

    def ladder(qubit: int, raise_: bool) -> SparsePauliOp:
        # S+ = |1><0| = (X - iY)/2 ; S- = |0><1| = (X + iY)/2.
        sign = -0.5j if raise_ else 0.5j
        return SparsePauliOp.from_sparse_list([("X", [qubit], 0.5), ("Y", [qubit], sign)], nq)

    q = lambda obj, pos: obj * n + pos  # noqa: E731
    op = (
        ladder(q(u, i + 1), True)
        .compose(ladder(q(v, i), True))
        .compose(ladder(q(u, i), False))
        .compose(ladder(q(v, i + 1), False))
    )
    return (op + op.adjoint()).simplify()


def cost_operator(instance: ProblemInstance):
    """Objective-only Ising Hamiltonian. On a permutation matrix its value is the
    path cost in km/s, identity term included."""
    from dextrivia.solvers.quantum_qaoa import cost_hamiltonian

    return cost_hamiltonian(objective_qubo(instance))


def swap_qaoa_circuit(instance: ProblemInstance, reps: int, initial_sequence=None):
    """The parameterised circuit: ``[X prep] (phase separator, swap mixer) x reps``.

    Returns ``(circuit, gammas, betas)``. With ``initial_sequence`` the circuit
    prepares that permutation with X gates; without it there is no preparation
    and the caller supplies the initial statevector (the uniform start).
    """
    from qiskit.circuit import ParameterVector, QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate

    n = instance.n
    circuit = QuantumCircuit(n * n)
    if initial_sequence is not None:
        for p, obj in enumerate(initial_sequence):
            circuit.x(int(obj) * n + p)

    gammas = ParameterVector("γ", reps)
    betas = ParameterVector("β", reps)
    cost = cost_operator(instance)
    terms = [(i, swap_mixer_term(n, i, u, v)) for i, (u, v) in mixer_schedule(n)]
    for layer in range(reps):
        circuit.append(PauliEvolutionGate(cost, time=gammas[layer]), range(n * n))
        for _, term in terms:
            circuit.append(PauliEvolutionGate(term, time=betas[layer]), range(n * n))
    return circuit, gammas, betas


#: Gate set the statevector backend simulates. Transpiling first matters for
#: correctness evidence, not just speed: ``Statevector.evolve`` on an
#: unsynthesised ``PauliEvolutionGate`` exponentiates the whole operator as a
#: matrix, which would verify the maths but not the gate sequence hardware runs.
SIMULATION_BASIS = ("cx", "rz", "sx", "x")


def synthesize(circuit):
    """Decompose to ``SIMULATION_BASIS``, keeping parameters free (no layout)."""
    from qiskit import transpile

    return transpile(circuit, basis_gates=list(SIMULATION_BASIS), optimization_level=0)


def _infeasible(name: str, reason: str, t0: float, **metadata: Any) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=time.perf_counter() - t0,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason, "versions": library_versions(*_PACKAGES), **metadata},
    )


class QAOASwapSolver:
    """Constraint-preserving QAOA, COBYLA outer loop, random restarts.

    ``restarts`` random initialisations; the best final expectation wins and
    every restart's value is recorded, as in ``qaoa``. Angles are drawn as
    ``gamma ~ U(0, 2 pi / L)`` with L the greedy path cost (phase differences
    between sequences are gamma times a cost difference of order L) and
    ``beta ~ U(0, pi / 2)`` (the partial mixers square to the identity on the
    feasible subspace, so beta has period pi and pi/2 is the full swap).
    """

    name = "qaoa-swap"

    def __init__(
        self,
        reps: int = 2,
        shots: int = 4096,
        maxiter: int = 300,
        restarts: int = 3,
        backend: str = "subspace",
        initial_state: str = "uniform",
        max_n: int | None = None,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if initial_state not in INITIAL_STATES:
            raise ValueError(f"initial_state must be one of {INITIAL_STATES}")
        self.reps = int(reps)
        self.shots = int(shots)
        self.maxiter = int(maxiter)
        self.restarts = int(restarts)
        self.backend = backend
        self.initial_state = initial_state
        default_max = SUBSPACE_MAX_N if backend == "subspace" else STATEVECTOR_MAX_N
        self.max_n = default_max if max_n is None else int(max_n)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        n = instance.n
        if n > self.max_n:
            size = (
                f"{math.factorial(n):,} amplitudes, ~{permutation_space_bytes(n) / 2**20:.0f} MiB"
                if self.backend == "subspace"
                else f"{n * n} qubits, {16 * 2 ** (n * n) / 2**20:,.0f} MiB statevector"
            )
            return _infeasible(
                self.name,
                f"N={n} ({size}) is over the qaoa-swap {self.backend} limit of N={self.max_n}",
                t0,
                backend=self.backend,
            )
        if n < 2:
            return _infeasible(self.name, "need at least 2 objects to have a leg", t0)
        try:
            return self._run(instance, seed, t0)
        except ImportError as exc:
            return _infeasible(self.name, f"backend unavailable: {exc}", t0)
        except (ValueError, MemoryError) as exc:
            return _infeasible(self.name, f"{type(exc).__name__}: {exc}", t0)

    def _run(self, instance: ProblemInstance, seed: int | None, t0: float) -> Solution:
        from scipy.optimize import minimize

        n = instance.n
        rng = np.random.default_rng(seed)
        space = PermutationSpace(instance)
        initial_sequence = tuple(range(n)) if self.initial_state == "basis" else None
        psi0 = (
            space.uniform_state()
            if initial_sequence is None
            else space.basis_state(initial_sequence)
        )

        if self.backend == "subspace":
            probabilities = self._subspace_backend(space, psi0)
        else:
            probabilities = self._statevector_backend(instance, space, initial_sequence)

        evaluations = 0

        def objective(values: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            return float(probabilities(values)[0] @ space.costs)

        gamma_scale = 2.0 * math.pi / max(path_upper_bound(instance), 1e-12)
        best: tuple[float, np.ndarray] | None = None
        restart_values: list[float] = []
        for _ in range(max(1, self.restarts)):
            x0 = np.concatenate(
                [
                    rng.uniform(0.0, gamma_scale, self.reps),
                    rng.uniform(0.0, math.pi / 2, self.reps),
                ]
            )
            result = minimize(objective, x0, method="COBYLA", options={"maxiter": self.maxiter})
            restart_values.append(float(result.fun))
            if best is None or result.fun < best[0]:
                best = (float(result.fun), np.asarray(result.x, dtype=float))
        assert best is not None
        expectation, parameters = best

        perm_probs, leaked = probabilities(parameters)
        # Shots: feasible outcomes are drawn from the permutation distribution;
        # on the statevector backend the leaked mass (numerically ~1e-15) is
        # drawn too and lands as infeasible bitstrings, so feasibility is measured.
        counts = rng.multinomial(
            self.shots,
            np.append(perm_probs, max(leaked, 0.0)) / (perm_probs.sum() + max(leaked, 0.0)),
        )
        onehot = np.zeros((space.dim, n * n), dtype=int)
        onehot[np.arange(space.dim)[:, None], space.perms.astype(int) * n + np.arange(n)] = 1
        samples = np.repeat(onehot, counts[:-1], axis=0)
        if counts[-1]:
            samples = np.vstack([samples, np.zeros((counts[-1], n * n), dtype=int)])
        summary = summarize_samples(instance, objective_qubo(instance), samples)

        optimum = float(space.costs.min())
        return Solution(
            sequence=summary["best_repaired_sequence"],
            total_dv_kms=summary["best_repaired_dv_kms"],
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                **summary,
                "backend": self.backend,
                "initial_state": self.initial_state,
                "initial_sequence": initial_sequence,
                "feasibility_measured": self.backend == "statevector",
                "runtime_kind": f"classical {self.backend} simulation time",
                "qubits": n * n,
                "subspace_dim": space.dim,
                "reps": self.reps,
                "shots": self.shots,
                "restarts": self.restarts,
                "maxiter": self.maxiter,
                "objective_evaluations": evaluations,
                "final_expectation": expectation,
                "restart_expectations": restart_values,
                "gammas": parameters[: self.reps].tolist(),
                "betas": parameters[self.reps :].tolist(),
                # Exact distribution quantities, free from the simulation and
                # free of shot noise. The optimum here is computed from the
                # enumerated subspace the simulator already holds; it is not a
                # separate oracle call, and it never feeds back into the search.
                "exact_mean_dv_kms": float(perm_probs @ space.costs),
                "exact_optimal_probability": float(
                    perm_probs[space.costs <= optimum + 1e-12].sum()
                ),
                "leaked_probability": float(leaked),
                "seed": None if seed is None else int(seed),
                "versions": library_versions(*_PACKAGES),
            },
        )

    def _subspace_backend(self, space: PermutationSpace, psi0: np.ndarray):
        reps = self.reps

        def probabilities(values: np.ndarray) -> tuple[np.ndarray, float]:
            psi = space.evolve(values[:reps], values[reps:], psi0)
            return np.abs(psi) ** 2, 0.0

        return probabilities

    def _statevector_backend(self, instance, space: PermutationSpace, initial_sequence):
        from qiskit.quantum_info import Statevector

        circuit, gammas, betas = swap_qaoa_circuit(instance, self.reps, initial_sequence)
        circuit = synthesize(circuit)
        start = (
            Statevector(space.embed(space.uniform_state()))
            if initial_sequence is None
            else Statevector.from_label("0" * circuit.num_qubits)
        )
        indices = space.bitstring_indices()
        reps = self.reps

        def probabilities(values: np.ndarray) -> tuple[np.ndarray, float]:
            bound = circuit.assign_parameters(
                {
                    **dict(zip(gammas, values[:reps], strict=True)),
                    **dict(zip(betas, values[reps:], strict=True)),
                }
            )
            full = start.evolve(bound).probabilities()
            perm = full[indices]
            return perm, float(1.0 - perm.sum())

        return probabilities
