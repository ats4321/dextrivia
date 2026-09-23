"""QUBO formulation of the open-path debris sequencing problem.

Position encoding: one binary ``x[i, p]`` per (object, position) pair, so N
objects cost N**2 variables. ``x[i, p] == 1`` means "object i is visited p-th".

    H(x) = H_obj(x) + A * H_pen(x)

    H_obj(x) = sum_{p=0}^{N-2} sum_{i,j} C_p[i,j] x[i,p] x[j,p+1]
    H_pen(x) = sum_i (1 - sum_p x[i,p])**2  +  sum_p (1 - sum_i x[i,p])**2

There are N-1 terms in H_obj, not N: this is an OPEN PATH, there is no leg from
the last object back to the first. ``C_p`` is ``instance.leg_costs(p)``, which
is the same matrix for every p on a static instance and a different one per leg
on a time-slotted instance -- so time dependence needs no extra machinery here,
it is the only thing the leg index p was ever going to mean.

This module imports nothing outside numpy on purpose. The formulation is the
part of the QUBO workspace that has to stay testable on a machine with no
quantum extra installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance

__all__ = [
    "PathQUBO",
    "build_qubo",
    "default_penalty",
    "DEFAULT_PENALTY_SAFETY",
    "path_upper_bound",
    "decode",
    "is_feasible",
    "repair",
    "summarize_samples",
    "sweep_penalty",
]


@dataclass(frozen=True)
class PathQUBO:
    """A built QUBO. ``energy(x) = x @ Q @ x + offset`` for binary vectors x.

    ``Q`` is upper triangular of shape (N**2, N**2); the diagonal carries the
    linear terms, which is legitimate because ``x_u**2 == x_u`` for binaries.
    ``offset`` is the constant ``2*A*N`` dropped by that expansion, and it is
    kept explicitly rather than folded away so that the energy of a feasible
    assignment is exactly its path cost in km/s -- see ``test_qubo_formulation``.
    """

    n: int
    penalty: float
    Q: np.ndarray
    offset: float

    @property
    def num_variables(self) -> int:
        return self.n * self.n

    def var(self, i: int, p: int) -> int:
        """Variable index of "object ``i`` is visited ``p``-th"."""
        if not (0 <= i < self.n and 0 <= p < self.n):
            raise IndexError(f"({i}, {p}) outside 0..{self.n - 1}")
        return i * self.n + p

    def energy(self, x: np.ndarray | Sequence[int]) -> float:
        """Energy including the constant offset."""
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(x @ self.Q @ x) + self.offset

    def to_qubo_dict(self) -> dict[tuple[int, int], float]:
        """``{(u, v): weight}`` in the shape dimod's ``sample_qubo`` wants."""
        rows, cols = np.nonzero(self.Q)
        return {(int(u), int(v)): float(self.Q[u, v]) for u, v in zip(rows, cols, strict=True)}

    def to_ising(self) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
        """Convert to ``const + sum_u h_u z_u + sum_{u<v} J_uv z_u z_v``.

        Uses ``x_u = (1 - z_u) / 2``, i.e. ``z = +1`` is ``x = 0`` and ``z = -1``
        is ``x = 1``, matching the usual Z-eigenvalue convention so that
        ``z_u`` maps straight onto a Pauli Z on qubit u.
        """
        diagonal = np.diag(self.Q).copy()
        off = self.Q - np.diag(diagonal)  # upper triangular, pair weights W_uv

        h = -diagonal / 2.0 - (off.sum(axis=0) + off.sum(axis=1)) / 4.0
        rows, cols = np.nonzero(off)
        J = {(int(u), int(v)): float(off[u, v]) / 4.0 for u, v in zip(rows, cols, strict=True)}
        const = self.offset + diagonal.sum() / 2.0 + off.sum() / 4.0
        return h, J, float(const)


def path_upper_bound(instance: ProblemInstance) -> float:
    """An upper bound on the optimal open-path cost L*, from the greedy solver.

    Any feasible path is an upper bound on nothing useful, but the *best* path
    is bounded above by whatever greedy found, and that is the quantity the
    penalty has to dominate.
    """
    from dextrivia.solvers.greedy import GreedySolver

    return float(GreedySolver().solve(instance).total_dv_kms)


#: Multiplier on the provable threshold. Chosen by measurement, not taste: on
#: the 10-object Hohmann instance, simulated annealing is already 100% feasible
#: at 1.01x and its gap to the optimum grows monotonically with the penalty
#: (1.9% at 1.01x, 6% at 1.1x, 27% at 1.5x, 45% at 8x). Big penalties do not buy
#: feasibility here, they only flatten the objective -- see docs/qubo.md.
DEFAULT_PENALTY_SAFETY = 1.1


def default_penalty(instance: ProblemInstance, safety: float = DEFAULT_PENALTY_SAFETY) -> float:
    """Penalty weight A, chosen so infeasible assignments cannot win.

    The bound: ``H_pen`` is a sum of squared integers, so it is 0 on a
    permutation matrix and >= 1 on anything else. Leg costs are non-negative, so
    ``H_obj >= 0`` everywhere. An infeasible assignment therefore scores at
    least ``A``, while the best feasible one scores exactly ``L*``. Hence

        A > L*   =>   every infeasible assignment is worse than the optimum.

    ``L*`` is unknown up front but bounded by ``path_upper_bound``, so
    ``A = safety * path_upper_bound(instance)`` with ``safety > 1`` is provably
    sufficient. Lucas 2014 ("Ising formulations of many NP problems", sec. 7.2)
    quotes the looser TSP-style rule ``A > max_ij W_ij``, which is a statement
    about single-variable flips and does not by itself rule out an infeasible
    assignment that dodges several expensive legs at once; the bound above costs
    one greedy run and does.

    ``safety`` must exceed 1 for the bound to hold at all. It should not exceed
    it by much: the leg costs on a real instance span ~0.005-0.05 km/s while A
    is ~0.13, so raising A further compresses the objective into the numerical
    noise of the penalty terms and a sampler stops being able to tell a good
    path from a mediocre one. ``sweep_penalty`` measures that directly.
    """
    if safety <= 1.0:
        raise ValueError(f"safety must exceed 1 for the bound to hold, got {safety}")
    bound = path_upper_bound(instance)
    if bound <= 0.0:
        # Degenerate all-zero costs: any positive A separates feasible from not.
        return float(max(np.max(np.abs(instance.costs)), 1.0))
    return float(safety * bound)


def build_qubo(instance: ProblemInstance, penalty: float | None = None) -> PathQUBO:
    """Build the position-encoded QUBO. ``penalty=None`` uses ``default_penalty``."""
    costs_min = float(np.min(instance.costs))
    if costs_min < 0.0:
        raise ValueError(
            f"leg costs must be non-negative for the penalty bound to hold, got {costs_min}"
        )

    n = instance.n
    a = default_penalty(instance) if penalty is None else float(penalty)
    if a <= 0.0:
        raise ValueError(f"penalty must be positive, got {a}")

    size = n * n
    m = np.zeros((size, size))

    # Objective: N-1 legs, leg p connects position p to position p+1.
    for p in range(n - 1):
        departing = np.arange(n) * n + p
        arriving = np.arange(n) * n + p + 1
        m[np.ix_(departing, arriving)] += instance.leg_costs(p)

    # Penalty: each object visited exactly once, each position filled exactly
    # once. Expanding (1 - S)**2 gives 1 - 2S + S**2; the 1 goes to the offset,
    # the -2S to the diagonal, the S**2 to the full block.
    for i in range(n):  # object i occupies exactly one position
        block = np.arange(n) + i * n
        m[np.ix_(block, block)] += a
        m[block, block] -= 2.0 * a
    for p in range(n):  # position p holds exactly one object
        block = np.arange(n) * n + p
        m[np.ix_(block, block)] += a
        m[block, block] -= 2.0 * a

    # Fold to upper triangular; x_u x_v == x_v x_u so this is exact for binaries.
    q = np.triu(m) + np.tril(m, -1).T
    return PathQUBO(n=n, penalty=a, Q=q, offset=2.0 * a * n)


def _as_matrix(x: np.ndarray | Sequence[int], n: int) -> np.ndarray:
    """Sample vector -> (object, position) occupancy matrix."""
    return np.asarray(x, dtype=int).reshape(n, n)


def is_feasible(x: np.ndarray | Sequence[int], n: int) -> bool:
    """True if ``x`` is a permutation matrix, i.e. decodes to a real sequence."""
    matrix = _as_matrix(x, n)
    return bool(
        np.array_equal(matrix.sum(axis=0), np.ones(n, dtype=int))
        and np.array_equal(matrix.sum(axis=1), np.ones(n, dtype=int))
    )


def decode(x: np.ndarray | Sequence[int], n: int) -> tuple[int, ...] | None:
    """Decode to a visiting order, or ``None`` if the sample violates a constraint."""
    if not is_feasible(x, n):
        return None
    matrix = _as_matrix(x, n)
    return tuple(int(np.argmax(matrix[:, p])) for p in range(n))


def repair(x: np.ndarray | Sequence[int], n: int) -> tuple[int, ...]:
    """Force any sample into a valid sequence, most-confident position first.

    Positions are filled in the order the sampler was most sure about them, and
    ties fall to the lowest object index, so the repair is deterministic. A
    repaired result is a different measurement from a raw one and must be
    reported as such -- see ``summarize_samples``.
    """
    matrix = _as_matrix(x, n).astype(float)
    sequence: list[int] = [-1] * n
    unassigned = set(range(n))
    # Confidence of a position = how strongly any single object claims it.
    for p in sorted(range(n), key=lambda p: -matrix[:, p].max()):
        best = max(unassigned, key=lambda i: (matrix[i, p], -i))
        sequence[p] = best
        unassigned.remove(best)
    return tuple(sequence)


def summarize_samples(
    instance: ProblemInstance,
    qubo: PathQUBO,
    samples: np.ndarray,
) -> dict[str, Any]:
    """Score a batch of samples, keeping raw and repaired results apart.

    ``samples`` is (num_samples, N**2) of 0/1. The raw feasibility rate is the
    honest number -- it says how often the sampler respected the one-hot
    constraints on its own. ``best_repaired_dv`` always exists, which is exactly
    why it must never be quoted as if it were a raw result.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=int))
    n = instance.n

    raw_feasible = 0
    best_was_raw = False
    best_raw: tuple[float, tuple[int, ...]] | None = None
    best_repaired: tuple[float, tuple[int, ...]] | None = None

    for sample in samples:
        sequence = decode(sample, n)
        was_raw = sequence is not None
        if sequence is not None:
            raw_feasible += 1
            cost = instance.path_cost(sequence)
            if best_raw is None or cost < best_raw[0]:
                best_raw = (cost, sequence)
        else:
            sequence = repair(sample, n)
            cost = instance.path_cost(sequence)
        if best_repaired is None or cost < best_repaired[0]:
            best_repaired = (cost, sequence)
            best_was_raw = was_raw

    return {
        "num_samples": int(len(samples)),
        "raw_feasible_samples": raw_feasible,
        "feasibility_rate": raw_feasible / len(samples) if len(samples) else 0.0,
        "best_raw_dv_kms": best_raw[0] if best_raw else None,
        "best_raw_sequence": best_raw[1] if best_raw else None,
        "best_repaired_dv_kms": best_repaired[0] if best_repaired else None,
        "best_repaired_sequence": best_repaired[1] if best_repaired else None,
        # True when the overall best came out of the sampler already feasible,
        # i.e. the repair contributed nothing to the headline number.
        "best_was_raw_sample": best_was_raw,
        "penalty": qubo.penalty,
    }


def sweep_penalty(
    instance: ProblemInstance,
    sample_fn: Callable[[PathQUBO], np.ndarray],
    factors: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 4.0, 8.0),
) -> list[dict[str, Any]]:
    """Run ``sample_fn`` at several penalty weights and report what each bought.

    ``factors`` multiply ``path_upper_bound(instance)``, so a factor of 1.0 sits
    exactly on the threshold where the bound stops being provable -- the point
    of sweeping is to see the feasibility rate collapse below it and the
    solution quality degrade well above it.
    """
    bound = path_upper_bound(instance)
    scale = bound if bound > 0.0 else float(max(np.max(np.abs(instance.costs)), 1.0))

    records = []
    for factor in factors:
        qubo = build_qubo(instance, penalty=factor * scale)
        record = summarize_samples(instance, qubo, sample_fn(qubo))
        record["penalty_factor"] = float(factor)
        record["path_upper_bound_kms"] = bound
        record["provably_sufficient"] = bool(qubo.penalty > bound)
        records.append(record)
    return records
