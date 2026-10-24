"""Certified lower bounds, and an exact MIP, via HiGHS (``scipy.optimize``).

The benchmark ran out of oracles at N=18. Above that the reference was "the
best any solver found", and nobody could tell whether greedy was optimal or
just unchallenged. This module answers that with a number that is provably
<= the optimum, so every reference is either proven optimal or reported with
its gap.

Model: the time-indexed ("3-index") formulation, open path, N-1 legs.

    x[i,p]    object i is visited p-th                      N*N
    y[p,i,j]  leg p flies i -> j                            (N-1)*N*N

    sum_p x[i,p] = 1,  sum_i x[i,p] = 1                     assignment
    sum_j y[p,i,j] = x[i,p]                                 leg p leaves what sits at p
    sum_i y[p,i,j] = x[j,p+1]                               ...and arrives at what sits at p+1
    minimise sum_p sum_ij C_p[i,j] y[p,i,j]                 C_p = instance.leg_costs(p)

That is exact for integer x (y is then forced), and it prices time-slotted
costs for free, because y's position index *is* the slot index. Its LP
relaxation alone is weak here -- 17-44% below the optimum on the committed
instances -- because a fractional x can smear an object over positions.

**Subtour cuts** fix most of that. Sum the legs over positions,
``z[i,j] = sum_p y[p,i,j]``, and add a depot d with arcs ``d -> i`` of weight
``x[i,0]`` (where the path starts). Every integer solution sends one unit of
flow from d to every object, so for every object set T:

    sum_{k in T} x[k,0] + sum_{i not in T, j in T} z[i,j] >= 1

(z is carried as its own variable, so a cut has |T|*(N-|T|) nonzeros rather
than N-1 times that -- the difference between seconds and minutes at N=40.)

Violated cuts are found exactly by one max-flow per object (the min cut from d
to that object), added, and the LP re-solved until none is violated. On the
static v2 instances this closes the gap completely at N=30; on time-slotted
ones HiGHS branch-and-bound on top of the cut LP (``HiGHSSolver``) closes the
rest where it can in the time limit, and reports ``mip_dual_bound`` where it
cannot.

"Certified" means: to HiGHS's feasibility tolerances (1e-7), minus a stated
safety margin ``BOUND_MARGIN_KMS``. It is a floating-point LP bound, not an
exact-rational proof, and every metadata record says so.

SciPy is imported inside the functions: ``dextrivia.solvers`` has to stay
importable without the ``quantum`` extra.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution

__all__ = [
    "BOUND_MARGIN_KMS",
    "HIGHS_MAX_N",
    "CutLP",
    "lp_lower_bound",
    "HiGHSSolver",
]

#: Subtracted from every bound this module reports. HiGHS works to 1e-7 primal
#: and dual feasibility on costs of order 1 km/s; 1e-6 km/s (1 mm/s) is ten
#: times that, and five orders of magnitude below any gap worth reporting.
BOUND_MARGIN_KMS = 1e-6

#: (N-1)*N*N leg variables: 205k at N=60. Measured, see docs/bounds.md.
HIGHS_MAX_N = 60

#: Max-flow needs integer capacities; LP values are scaled by this first.
_FLOW_SCALE = 10**7

CERTIFICATION = (
    "HiGHS LP/MIP dual bound, floating point (feasibility tol 1e-7), "
    f"minus a {BOUND_MARGIN_KMS:g} km/s margin; not an exact-rational proof"
)


@dataclass
class CutLP:
    """The time-indexed model plus the subtour cuts found so far."""

    n: int
    c: np.ndarray
    a_eq: Any  # scipy.sparse.csr_matrix
    b_eq: np.ndarray
    upper: np.ndarray
    cuts: list[np.ndarray] = field(default_factory=list)
    #: The last LP solution the cut loop produced.
    solution: np.ndarray | None = None
    _seen: set[frozenset[int]] = field(default_factory=set)

    @property
    def nx(self) -> int:
        return self.n * self.n

    @property
    def z0(self) -> int:
        """First column of the aggregate z[i,j]; the cuts live on z, not on y."""
        return self.nx + (self.n - 1) * self.n * self.n

    def x_col(self, i: int | np.ndarray, p: int | np.ndarray) -> Any:
        return i * self.n + p

    def cut_matrix(self) -> Any:
        from scipy import sparse

        if not self.cuts:
            return None
        rows = np.concatenate([np.full(len(c), k) for k, c in enumerate(self.cuts)])
        cols = np.concatenate(self.cuts)
        return sparse.csr_matrix(
            (np.ones(len(cols)), (rows, cols)), shape=(len(self.cuts), len(self.c))
        )

    def add_cut(self, sink_side: np.ndarray) -> bool:
        """Inflow into ``sink_side`` (object indices) >= 1. False if already present."""
        key = frozenset(int(k) for k in sink_side)
        if key in self._seen:
            return False
        self._seen.add(key)
        n = self.n
        inside = np.zeros(n, dtype=bool)
        inside[sink_side] = True
        tails, heads = np.flatnonzero(~inside), np.flatnonzero(inside)
        z_cols = self.z0 + tails[:, None] * n + heads[None, :]
        self.cuts.append(np.concatenate([self.x_col(heads, 0), z_cols.ravel()]))
        return True


def build_model(instance: ProblemInstance) -> CutLP:
    """The time-indexed model, costs read through ``leg_costs`` only."""
    from scipy import sparse

    n, legs = instance.n, instance.n - 1
    nx, ny = n * n, legs * n * n
    # x, then y, then z[i,j] = sum_p y[p,i,j] (cost 0; it keeps each cut to
    # |T|*(N-|T|) nonzeros instead of (N-1) times that)
    c = np.zeros(nx + ny + n * n)
    for p in range(legs):
        c[nx + p * n * n : nx + (p + 1) * n * n] = instance.leg_costs(p).ravel()

    idx = np.arange(n)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    row = 0

    def add(row_ids: np.ndarray, col_ids: np.ndarray, value: float) -> None:
        rows.append(row_ids.ravel())
        cols.append(col_ids.ravel())
        vals.append(np.full(col_ids.size, value))

    # each object once: row i, columns x[i, :]
    add(np.repeat(idx, n) + row, (idx[:, None] * n + idx[None, :]), 1.0)
    row += n
    # each position once: row p, columns x[:, p]
    add(np.repeat(idx, n) + row, (idx[None, :] * n + idx[:, None]), 1.0)
    row += n
    for p in range(legs):
        base = nx + p * n * n
        # out of i at leg p: sum_j y[p,i,j] - x[i,p] = 0
        add(np.repeat(idx, n) + row, base + idx[:, None] * n + idx[None, :], 1.0)
        add(idx + row, idx * n + p, -1.0)
        row += n
        # into j at leg p: sum_i y[p,i,j] - x[j,p+1] = 0
        add(np.repeat(idx, n) + row, base + idx[None, :] * n + idx[:, None], 1.0)
        add(idx + row, idx * n + p + 1, -1.0)
        row += n
    # z[i,j] - sum_p y[p,i,j] = 0
    pair = np.arange(n * n)
    add(pair + row, nx + ny + pair, 1.0)
    add(np.tile(pair, legs) + row, nx + np.arange(ny), -1.0)
    row += n * n

    a_eq = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(row, len(c)),
    )
    b_eq = np.zeros(row)
    b_eq[: 2 * n] = 1.0

    upper = np.ones(len(c))
    for p in range(legs):
        upper[nx + p * n * n + idx * n + idx] = 0.0  # no i -> i leg
    return CutLP(n=n, c=c, a_eq=a_eq, b_eq=b_eq, upper=upper)


def separate(model: CutLP, solution: np.ndarray) -> int:
    """Add every violated subtour cut found by min-cut from the depot. Returns count."""
    from scipy import sparse
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    n = model.n
    z = solution[model.z0 :].reshape(n, n)
    start = solution[model.x_col(np.arange(n), 0)]
    capacity = np.zeros((n + 1, n + 1))
    capacity[:n, :n] = z
    capacity[n, :n] = start  # node n is the depot
    graph = sparse.csr_matrix(np.rint(np.clip(capacity, 0, None) * _FLOW_SCALE).astype(np.int32))

    added = 0
    covered = np.zeros(n, dtype=bool)
    for target in range(n):
        if covered[target]:
            continue  # already inside a cut found this round
        flow = maximum_flow(graph, n, target)
        if flow.flow_value >= _FLOW_SCALE * (1.0 - 1e-6):
            continue
        residual = graph - flow.flow
        residual.data = (residual.data > 0).astype(np.int8)
        residual.eliminate_zeros()
        reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
        sink_side = np.setdiff1d(np.arange(n), reachable)
        covered[sink_side] = True
        added += model.add_cut(sink_side)
    return added


def _solve_lp(model: CutLP, time_limit_s: float | None = None) -> Any:
    from scipy.optimize import linprog

    cuts = model.cut_matrix()
    return linprog(
        model.c,
        A_ub=None if cuts is None else -cuts,
        b_ub=None if cuts is None else -np.ones(cuts.shape[0]),
        A_eq=model.a_eq,
        b_eq=model.b_eq,
        bounds=np.column_stack([np.zeros_like(model.upper), model.upper]),
        # Interior point: 2-3x faster than dual simplex per cut round at N=40,
        # because scipy cannot warm-start the simplex between rounds anyway.
        method="highs-ipm",
        # Enforced inside the solve: at N=50 one LP can outlast the whole budget.
        options={} if time_limit_s is None else {"time_limit": max(1.0, time_limit_s)},
    )


def lp_lower_bound(
    instance: ProblemInstance, max_rounds: int = 50, time_limit_s: float | None = None
) -> tuple[float, CutLP, dict[str, Any]]:
    """Cut-loop LP bound. Returns (certified bound km/s, model with cuts, stats).

    Stops when no cut is violated, after ``max_rounds``, or past the time
    limit. Every round's LP value is a valid bound, so stopping early only
    costs tightness, never validity.
    """
    t0 = time.perf_counter()
    model = build_model(instance)
    plain = None
    rounds = 0
    result = None
    value = None  # last LP that solved to optimality; only those are bounds
    stopped = "converged"
    while rounds < max_rounds:
        remaining = None if time_limit_s is None else time_limit_s - (time.perf_counter() - t0)
        if remaining is not None and remaining <= 0:
            stopped = "time limit"
            break
        rounds += 1
        result = _solve_lp(model, remaining)
        if result.status != 0:
            # Time limit (or numerical trouble) inside an LP: an unfinished LP
            # proves nothing, so keep the previous round's value.
            stopped = f"LP status {result.status}: {result.message}"
            break
        value = float(result.fun)
        model.solution = result.x
        if plain is None:
            plain = value
        if separate(model, result.x) == 0:
            break
    else:
        stopped = "max rounds"
    # Costs are non-negative, so 0 is always a valid (useless) bound.
    bound = (value if value is not None else 0.0) - BOUND_MARGIN_KMS
    stats = {
        "lp_plain_bound_kms": (plain if plain is not None else 0.0) - BOUND_MARGIN_KMS,
        "lp_cut_bound_kms": bound,
        "cut_rounds": rounds,
        "cuts": len(model.cuts),
        "lp_seconds": time.perf_counter() - t0,
        "lp_stopped": stopped,
    }
    return bound, model, stats


def _is_integral(values: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(np.minimum(np.abs(values), np.abs(values - 1)) <= tol))


def _infeasible(name: str, reason: str, t0: float, **metadata: Any) -> Solution:
    return Solution(
        sequence=(),
        total_dv_kms=float("inf"),
        runtime_s=time.perf_counter() - t0,
        solver_name=name,
        feasible=False,
        metadata={"reason": reason, **metadata},
    )


class HiGHSSolver:
    """Cut-strengthened time-indexed MIP. Proves optimality or reports a bound.

    Deterministic: HiGHS's branch-and-bound does not depend on ``seed``. Its
    result can still depend on the machine when the time limit binds, which is
    why ``proven_optimal`` and the bound are recorded rather than assumed.
    """

    name = "highs"

    def __init__(self, time_limit_s: float = 60.0, max_n: int = HIGHS_MAX_N) -> None:
        self.time_limit_s = float(time_limit_s)
        self.max_n = int(max_n)

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution:
        t0 = time.perf_counter()
        if instance.n > self.max_n:
            return _infeasible(self.name, f"N={instance.n} exceeds highs limit {self.max_n}", t0)
        try:
            import scipy.optimize  # noqa: F401
        except ImportError as exc:
            return _infeasible(self.name, f"backend unavailable: {exc}", t0)
        return self._run(instance, seed, t0)

    def _run(self, instance: ProblemInstance, seed: int | None, t0: float) -> Solution:
        from scipy.optimize import Bounds, LinearConstraint, milp

        lp_bound, model, stats = lp_lower_bound(instance, time_limit_s=self.time_limit_s * 2 / 3)
        n = instance.n
        integrality = np.zeros(len(model.c))
        integrality[: model.nx] = 1  # integral x forces integral y
        constraints = [LinearConstraint(model.a_eq, model.b_eq, model.b_eq)]
        cuts = model.cut_matrix()
        if cuts is not None:
            constraints.append(LinearConstraint(cuts, 1.0, np.inf))
        remaining = max(1.0, self.time_limit_s - (time.perf_counter() - t0))
        result = milp(
            model.c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(np.zeros_like(model.upper), model.upper),
            options={"time_limit": remaining, "mip_rel_gap": 1e-9, "disp": False},
        )
        dual = getattr(result, "mip_dual_bound", None)
        mip_bound = (
            float(dual) - BOUND_MARGIN_KMS if dual is not None and np.isfinite(dual) else -np.inf
        )
        bound = max(lp_bound, mip_bound)
        common = {
            **stats,
            "lower_bound_kms": bound,
            "mip_dual_bound_kms": mip_bound if np.isfinite(mip_bound) else None,
            "bound_certification": CERTIFICATION,
            "mip_status": int(result.status),
            "mip_message": str(result.message),
            "time_limit_s": self.time_limit_s,
            "num_variables": len(model.c),
            "seed": seed,
            "deterministic": True,
        }
        values = result.x
        if (
            values is None
            and model.solution is not None
            and _is_integral(model.solution[: model.nx])
        ):
            # The cut LP landed on a permutation: it is optimal outright, and
            # branch-and-bound had nothing left to do but ran out of clock.
            values = model.solution
            common["solution_source"] = "integral cut LP"
        if values is None:
            return _infeasible(
                self.name, f"no integer solution within {self.time_limit_s}s", t0, **common
            )

        x = values[: model.nx].reshape(n, n)  # x[i, p]
        sequence = tuple(int(i) for i in np.argmax(x, axis=0))
        if sorted(sequence) != list(range(n)):
            return _infeasible(self.name, "MIP returned a non-permutation", t0, **common)
        total = instance.path_cost(sequence)
        gap = max(0.0, (total - bound) / total) if total > 0 else 0.0
        return Solution(
            sequence=sequence,
            total_dv_kms=total,
            runtime_s=time.perf_counter() - t0,
            solver_name=self.name,
            feasible=True,
            metadata={
                **common,
                "proven_optimal": total - bound <= 10 * BOUND_MARGIN_KMS,
                "gap": gap,
            },
        )
