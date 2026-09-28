"""Constraint-preserving QAOA over permutations, simulated in the permutation space.

The ansatz is Hadfield et al. 2019 (*From the Quantum Approximate Optimization
Algorithm to a Quantum Alternating Operator Ansatz*, Algorithms 12(2):34, §5.1):
the position one-hot encoding, a phase separator built from the path cost, and a
mixer made of *adjacent ordering-swap partial mixers*. The partial mixer for
positions (i, i+1) and objects {u, v} swaps u and v between those two positions
if and only if they occupy them, and does nothing otherwise:

    H_PS,i,{u,v} = S+(u,i+1) S+(v,i) S-(u,i) S-(v,i+1) + h.c.

(``S+ = |1><0|``, qubit (u, p) is "object u is visited p-th"). Every one of
these maps a permutation matrix to a permutation matrix, so a circuit that
starts in the feasible subspace never leaves it, and no penalty term is needed.

That is also what makes this module possible. The feasible subspace has
dimension N!, not 2**(N**2), and on it the whole circuit is simple:

* the phase separator is diagonal: ``exp(-i gamma cost(sigma))``;
* summing the partial mixers over all object pairs at a fixed position pair
  gives Hadfield's value-independent H_PS,i (their eq. 49), which on the
  feasible subspace *pairs* every permutation with the one that has positions
  i and i+1 exchanged. Its square is the identity there, so
  ``exp(-i beta H_PS,i) = cos(beta) I - i sin(beta) H_PS,i`` exactly.

This is an exact simulation of the same quantum circuit -- not an
approximation, not "quantum-inspired" -- and it is checked amplitude by
amplitude against the Qiskit circuit in ``tests/test_qaoa_swap.py``. Why the
per-object-pair partial mixers in the circuit may be regrouped this way (they
do NOT all act on disjoint qubits) is proved in ``docs/qubo.md`` §10.

Mixer ordering. Positions are applied in parity order -- all even i, then all
odd i -- because H_PS,i and H_PS,i' commute when |i - i'| > 1 (disjoint
qubits). Hadfield's color-parity mixer orders the same partial mixers
color-major rather than parity-major; on the feasible subspace ours is exactly
``exp(-i beta H_odd) exp(-i beta H_even)``. Open path: positions run 0..N-2,
there is no wrap-around "last" part as in their TSP tour.

numpy only. The Qiskit circuit that this mirrors lives in
``dextrivia.solvers.quantum_qaoa_swap``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

from dextrivia.core import ProblemInstance

__all__ = [
    "PermutationSpace",
    "color_partition",
    "mixer_schedule",
    "permutation_space_bytes",
]


def permutation_space_bytes(n: int) -> float:
    """Rough peak memory of ``PermutationSpace(n)``: the N! x N permutation table,
    N-1 partner index arrays, a cost vector and two complex state vectors."""
    return float(math.factorial(n)) * (n * 1 + (n - 1) * 4 + 8 + 2 * 16)


def color_partition(n: int) -> list[list[tuple[int, int]]]:
    """Proper edge colouring of K_n by the circle method: pairs grouped so that no
    two pairs in a group share an object.

    Partial mixers at one position with disjoint object pairs act on disjoint
    qubits, so a group can run as one parallel layer on hardware. Hadfield §5.1
    uses the same construction ("color partition"); n-1 colours for even n, n
    for odd n.
    """
    nodes = list(range(n)) + ([None] if n % 2 else [])
    m = len(nodes)
    groups = []
    for _ in range(m - 1):
        pairs = [(nodes[k], nodes[m - 1 - k]) for k in range(m // 2)]
        groups.append(sorted(tuple(sorted(p)) for p in pairs if None not in p))
        nodes = [nodes[0], nodes[-1], *nodes[1:-1]]
    return groups


def mixer_schedule(n: int) -> list[tuple[int, tuple[int, int]]]:
    """Order in which the circuit applies partial mixers ``(position i, (u, v))``.

    Parity-major (even i, then odd i), then colour, then i. Each object pair at
    each adjacent position pair appears exactly once: (N-1) * C(N, 2) terms.
    """
    schedule = []
    for parity in (0, 1):
        for group in color_partition(n):
            for i in range(parity, n - 1, 2):
                schedule.extend((i, pair) for pair in group)
    return schedule


def _rank(perms: np.ndarray) -> np.ndarray:
    """Lexicographic rank of each row, matching ``itertools.permutations`` order."""
    n = perms.shape[1]
    rank = np.zeros(len(perms), dtype=np.int64)
    for k in range(n):
        smaller_later = (perms[:, k + 1 :] < perms[:, k : k + 1]).sum(axis=1)
        rank += smaller_later * math.factorial(n - 1 - k)
    return rank


class PermutationSpace:
    """The N!-dimensional feasible subspace for one instance.

    Basis state ``k`` is the k-th permutation in ``itertools.permutations``
    order; ``perms[k][p]`` is the object visited p-th, so ``perms[k]`` is
    directly a visiting sequence.
    """

    def __init__(self, instance: ProblemInstance) -> None:
        n = instance.n
        self.n = n
        self.perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
        idx = self.perms.astype(np.intp)
        self.costs = np.zeros(len(self.perms))
        for p in range(n - 1):
            self.costs += instance.leg_costs(p)[idx[:, p], idx[:, p + 1]]
        self.partners = []
        for i in range(n - 1):
            swapped = self.perms.copy()
            swapped[:, [i, i + 1]] = swapped[:, [i + 1, i]]
            self.partners.append(_rank(swapped))
        self._parity_order = [*range(0, n - 1, 2), *range(1, n - 1, 2)]

    @property
    def dim(self) -> int:
        return len(self.perms)

    def uniform_state(self) -> np.ndarray:
        return np.full(self.dim, 1.0 / math.sqrt(self.dim), dtype=complex)

    def basis_state(self, sequence: Sequence[int]) -> np.ndarray:
        state = np.zeros(self.dim, dtype=complex)
        state[int(_rank(np.array([sequence]))[0])] = 1.0
        return state

    def evolve(
        self, gammas: Sequence[float], betas: Sequence[float], state: np.ndarray
    ) -> np.ndarray:
        """Apply ``reps = len(gammas)`` layers: phase separator, then parity mixer."""
        psi = np.array(state, dtype=complex)
        for gamma, beta in zip(gammas, betas, strict=True):
            psi *= np.exp(-1j * gamma * self.costs)
            c, s = math.cos(beta), math.sin(beta)
            for i in self._parity_order:
                psi = c * psi - 1j * s * psi[self.partners[i]]
        return psi

    def embed(self, psi: np.ndarray) -> np.ndarray:
        """Subspace amplitudes -> full 2**(N**2) statevector, Qiskit bit order
        (qubit ``u*N + p`` is bit ``u*N + p`` of the index). Tests only: this is
        exactly the object the subspace exists to avoid."""
        n = self.n
        full = np.zeros(2 ** (n * n), dtype=complex)
        full[self.bitstring_indices()] = psi
        return full

    def bitstring_indices(self) -> np.ndarray:
        """Computational-basis index of each permutation's one-hot bitstring."""
        n = self.n
        positions = np.arange(n)
        return (2 ** (self.perms.astype(np.int64) * n + positions)).sum(axis=1)
