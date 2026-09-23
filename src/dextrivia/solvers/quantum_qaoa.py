"""QAOA on a statevector simulator, via Qiskit's V2 primitives.

THE QUBIT WALL IS THE HEADLINE. The position encoding needs N**2 qubits, and a
statevector is 2**(N**2) complex amplitudes:

    N = 3    9 qubits        512 amplitudes      8 KB
    N = 4   16 qubits         65536              1 MB
    N = 5   25 qubits      33.5 million        512 MB
    N = 6   36 qubits      6.9e10               1 TB

So this solver tops out at N=4 on a laptop and N=5 on a large machine, against a
Held-Karp oracle that is exact to N=18 and an OR-Tools heuristic that does not
care. That gap is the finding, not a footnote to it -- the interesting question
at these sizes is solution *quality* at fixed depth, which is why every run
records its reps, shots and final expectation value.

Deliberately does not use ``qiskit-optimization``: it has not seen a release
since 2025, it pulls in ``qiskit-algorithms`` behind it, and the translation
from our QUBO to an Ising Hamiltonian is a dozen lines we already own and test.
Uses the ``qaoa_ansatz`` function rather than the ``QAOAAnsatz`` class, which
Qiskit deprecated in 2.1 along with the rest of the NLocal hierarchy.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.qubo import PathQUBO, build_qubo, library_versions, summarize_samples

__all__ = ["QAOASolver", "QAOA_MAX_N", "statevector_bytes"]

#: 16 qubits, a 1 MB statevector. N=5 is 25 qubits and half a gigabyte before
#: the optimiser has evaluated anything, so it is opt-in, not the default.
QAOA_MAX_N = 4

_PACKAGES = ("qiskit", "scipy", "numpy")


def statevector_bytes(n: int) -> float:
    """Bytes for a dense statevector over the N**2 qubits this encoding needs."""
    return 16.0 * 2.0 ** (n * n)


def _infeasible(name: str, reason: str, t0: float, **metadata: Any) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=time.perf_counter() - t0,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason, "versions": library_versions(*_PACKAGES), **metadata},
    )


def cost_hamiltonian(qubo: PathQUBO):
    """QUBO -> Ising ``SparsePauliOp`` over N**2 qubits.

    The identity term carries the constant, so the expectation value the
    optimiser minimises is an energy in km/s-plus-penalty rather than an
    arbitrary shifted number, and can be compared with ``PathQUBO.energy``.
    """
    from qiskit.quantum_info import SparsePauliOp

    h, j, const = qubo.to_ising()
    terms: list[tuple[str, list[int], float]] = [("I", [0], const)]
    terms += [("Z", [u], float(w)) for u, w in enumerate(h) if w != 0.0]
    terms += [("ZZ", [u, v], w) for (u, v), w in j.items() if w != 0.0]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=qubo.num_variables).simplify()


class QAOASolver:
    """QAOA with a classical outer loop (scipy COBYLA) on a statevector sim.

    ``restarts`` random initialisations are tried because QAOA's landscape at
    p=1..3 is riddled with local optima; the best final expectation wins, and
    the spread across restarts is reported so a single lucky run cannot be
    mistaken for depth doing the work.
    """

    name = "qaoa"

    def __init__(
        self,
        reps: int = 2,
        shots: int = 4096,
        maxiter: int = 300,
        restarts: int = 3,
        penalty: float | None = None,
        max_n: int = QAOA_MAX_N,
    ) -> None:
        self.reps = int(reps)
        self.shots = int(shots)
        self.maxiter = int(maxiter)
        self.restarts = int(restarts)
        self.penalty = penalty
        self.max_n = int(max_n)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        qubits = instance.n * instance.n
        if instance.n > self.max_n:
            return _infeasible(
                self.name,
                f"N={instance.n} needs {qubits} qubits, over the qaoa limit of "
                f"{self.max_n} (={self.max_n**2} qubits, "
                f"{statevector_bytes(self.max_n) / 2**20:.0f} MiB statevector)",
                t0,
                qubits_required=qubits,
                statevector_bytes=statevector_bytes(instance.n),
            )

        try:
            return self._run(instance, seed, t0)
        except ImportError as exc:
            return _infeasible(self.name, f"backend unavailable: {exc}", t0)
        except (ValueError, MemoryError) as exc:
            return _infeasible(self.name, f"{type(exc).__name__}: {exc}", t0)

    def _run(self, instance: ProblemInstance, seed: int | None, t0: float) -> Solution:
        from qiskit.circuit.library import qaoa_ansatz
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler
        from scipy.optimize import minimize

        qubo = build_qubo(instance, penalty=self.penalty)
        hamiltonian = cost_hamiltonian(qubo)
        ansatz = qaoa_ansatz(hamiltonian, reps=self.reps)

        estimator = StatevectorEstimator(seed=seed)
        evaluations = 0

        def objective(values: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            job = estimator.run([(ansatz, hamiltonian, [values])])
            return float(job.result()[0].data.evs[0])

        # gamma multiplies the cost coefficients, so its useful range shrinks as
        # the Hamiltonian gets stiffer; beta multiplies the mixer and does not.
        scale = max(float(np.max(np.abs(hamiltonian.coeffs.real))), 1e-9)
        is_gamma = [p.name.startswith("γ") for p in ansatz.parameters]

        rng = np.random.default_rng(seed)
        best: tuple[float, np.ndarray] | None = None
        restart_values: list[float] = []
        for _ in range(max(1, self.restarts)):
            x0 = np.array(
                [
                    rng.uniform(0.0, math.pi / scale) if gamma else rng.uniform(0.0, math.pi)
                    for gamma in is_gamma
                ]
            )
            result = minimize(objective, x0, method="COBYLA", options={"maxiter": self.maxiter})
            restart_values.append(float(result.fun))
            if best is None or result.fun < best[0]:
                best = (float(result.fun), np.asarray(result.x, dtype=float))

        assert best is not None
        expectation, parameters = best

        measured = ansatz.copy()
        measured.measure_all()
        sampler = StatevectorSampler(seed=seed)
        counts = (
            sampler.run([(measured, [parameters])], shots=self.shots).result()[0].data.meas
        ).get_counts()

        # Qiskit bitstrings are little-endian: the rightmost character is qubit 0.
        samples = np.array(
            [
                [int(bit) for bit in key[::-1]]
                for key, count in counts.items()
                for _ in range(count)
            ],
            dtype=int,
        )
        summary = summarize_samples(instance, qubo, samples)

        return Solution(
            sequence=summary["best_repaired_sequence"],
            total_dv_kms=summary["best_repaired_dv_kms"],
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                **summary,
                "qubits": qubo.num_variables,
                "reps": self.reps,
                "shots": self.shots,
                "restarts": self.restarts,
                "maxiter": self.maxiter,
                "objective_evaluations": evaluations,
                "final_expectation": expectation,
                "restart_expectations": restart_values,
                "hamiltonian_terms": len(hamiltonian),
                "statevector_bytes": statevector_bytes(instance.n),
                "seed": seed,
                "versions": library_versions(*_PACKAGES),
            },
        )
