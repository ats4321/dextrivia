"""Simulated annealing on the position-encoded QUBO.

Quantum-*inspired*, not quantum: this is a classical Metropolis sampler on the
same Ising model a D-Wave annealer would be given, which makes it the control
that any real annealing result has to beat before the word "advantage" is worth
typing.

Backend is ``dwave-samplers``, not ``dwave-neal``. The two ship the same
``SimulatedAnnealingSampler``; neal was last released in 2022 and now only
re-exports from dwave-samplers, so depending on it would pin a dead name.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.qubo import PathQUBO, build_qubo, library_versions, summarize_samples

__all__ = ["SimulatedAnnealingSolver"]

#: Nothing breaks above this, but N**2 variables at N=40 is 1600 spins with an
#: all-to-all-ish penalty graph, and the run stops being a benchmark and starts
#: being a wait. Raise it deliberately, with a runtime you are prepared to pay.
SA_MAX_N = 40

_PACKAGES = ("dwave-samplers", "dimod", "numpy")


def _infeasible(name: str, reason: str, t0: float, **metadata: Any) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=time.perf_counter() - t0,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason, "versions": library_versions(*_PACKAGES), **metadata},
    )


class SimulatedAnnealingSolver:
    """Anneals the QUBO with dwave-samplers and decodes the best read.

    ``seed`` is passed straight to the sampler, so a run is reproducible given
    the same package version -- which is why the version goes in the metadata.
    """

    name = "sa-qubo"

    def __init__(
        self,
        num_reads: int = 500,
        num_sweeps: int = 1000,
        penalty: float | None = None,
        max_n: int = SA_MAX_N,
    ) -> None:
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)
        self.penalty = penalty
        self.max_n = int(max_n)

    def sample(self, qubo: PathQUBO, seed: int | None = None) -> np.ndarray:
        """Raw (num_reads, N**2) 0/1 samples. Also the hook ``sweep_penalty`` wants."""
        from dwave.samplers import SimulatedAnnealingSampler

        sampleset = SimulatedAnnealingSampler().sample_qubo(
            qubo.to_qubo_dict(),
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            seed=seed,
        )
        # Column k of the record belongs to sampleset.variables[k], which is not
        # guaranteed to be 0..N**2-1 in order; any variable the QUBO never
        # mentions is absent and stays 0.
        order = list(sampleset.variables)
        samples = np.zeros((len(sampleset.record.sample), qubo.num_variables), dtype=int)
        samples[:, order] = sampleset.record.sample
        return samples

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        if instance.n > self.max_n:
            return _infeasible(self.name, f"N={instance.n} exceeds sa limit {self.max_n}", t0)

        try:
            qubo = build_qubo(instance, penalty=self.penalty)
            samples = self.sample(qubo, seed=seed)
        except ImportError as exc:
            return _infeasible(self.name, f"backend unavailable: {exc}", t0)
        except (ValueError, MemoryError) as exc:
            return _infeasible(self.name, f"{type(exc).__name__}: {exc}", t0)

        summary = summarize_samples(instance, qubo, samples)
        return Solution(
            sequence=summary["best_repaired_sequence"],
            total_dv_kms=summary["best_repaired_dv_kms"],
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                **summary,
                "num_variables": qubo.num_variables,
                "num_reads": self.num_reads,
                "num_sweeps": self.num_sweeps,
                "seed": seed,
                "versions": library_versions(*_PACKAGES),
            },
        )
