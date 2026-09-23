"""Core interfaces for Dextrivia.

Every other module in this package is written against the types defined here.
Changing a signature in this file breaks the parallel workspaces, so read the
"Interface stability" section of CLAUDE.md before you do.

Conventions
-----------
Units       distances km, speeds and delta-v km/s, times in seconds.
Time        timezone-aware UTC ``datetime`` only. Naive datetimes are rejected.
Identity    objects are keyed by NORAD catalog ID (``int``). Names are NOT
            unique -- 106 of the 107 objects in the Iridium-33 snapshot are
            called "IRIDIUM 33 DEB" -- so names are display-only.

Problem statement
-----------------
The mission is an OPEN PATH, not a tour: the servicer starts at the first
object of the sequence and stops at the last one. There is no return leg to the
start and no depot/parking orbit. A sequence over N objects therefore has
exactly N-1 transfer legs, and the reverse of a sequence costs the same only if
the cost is symmetric and time-independent.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "DebrisObject",
    "ProblemInstance",
    "Solution",
    "CostModel",
    "Solver",
]


def _require_utc(epoch: datetime) -> datetime:
    if epoch.tzinfo is None:
        raise ValueError("epoch must be timezone-aware UTC, got a naive datetime")
    return epoch.astimezone(UTC)


@dataclass(frozen=True)
class DebrisObject:
    """One catalogued object, as it appears in a snapshot file."""

    norad_id: int
    name: str
    line1: str
    line2: str


@dataclass(frozen=True)
class ProblemInstance:
    """A concrete sequencing problem: which objects, at what epoch, at what cost.

    ``costs`` is EITHER of two shapes, and solvers must handle both:

    * static       ``C[i, j]``     shape (N, N)
      Cost in km/s of the leg from object i to object j, independent of when the
      leg is flown. This is all the current Hohmann model produces.

    * time-slotted ``C[t, i, j]``  shape (N-1, N, N)
      Same, but ``t`` is the POSITION OF THE LEG IN THE SEQUENCE (0-based), not
      an absolute time. Leg t departs the t-th object of the sequence. There are
      N-1 legs in an open path, hence N-1 slots. A later workspace will emit
      this shape from a cost model that accounts for RAAN drift between visits,
      where the cost of going i->j genuinely depends on how many transfers have
      already been flown.

    Slot semantics were chosen as "position in sequence" rather than "elapsed
    time" on purpose: it keeps the problem a finite DP (see
    ``dextrivia.solvers.exact``) instead of a continuous-time scheduling
    problem. If a future cost model needs wall-clock time, it must map wall-clock
    onto these slots itself and record the mapping in ``metadata``.

    ``metadata`` is free-form provenance: snapshot filename, selection rule,
    seed, cost model name. Benchmarks are only reproducible if it is populated.
    """

    norad_ids: tuple[int, ...]
    names: tuple[str, ...]
    epoch: datetime
    costs: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "norad_ids", tuple(int(i) for i in self.norad_ids))
        object.__setattr__(self, "names", tuple(str(s) for s in self.names))
        object.__setattr__(self, "epoch", _require_utc(self.epoch))
        object.__setattr__(self, "costs", np.asarray(self.costs, dtype=float))

        n = len(self.norad_ids)
        if n < 2:
            raise ValueError(f"need at least 2 objects, got {n}")
        if len(set(self.norad_ids)) != n:
            raise ValueError("norad_ids must be unique")
        if len(self.names) != n:
            raise ValueError("names and norad_ids must have the same length")

        if self.costs.shape == (n, n):
            pass
        elif self.costs.shape == (n - 1, n, n):
            pass
        else:
            raise ValueError(
                f"costs must have shape ({n}, {n}) or ({n - 1}, {n}, {n}), got {self.costs.shape}"
            )
        if not np.isfinite(self.costs).all():
            raise ValueError("costs must be finite")

    @property
    def n(self) -> int:
        """Number of objects."""
        return len(self.norad_ids)

    @property
    def time_dependent(self) -> bool:
        """True if costs are time-slotted ``C[t, i, j]``."""
        return self.costs.ndim == 3

    def leg_costs(self, step: int) -> np.ndarray:
        """The (N, N) cost matrix for the leg at position ``step`` of the sequence."""
        if not 0 <= step < self.n - 1:
            raise IndexError(f"step {step} outside 0..{self.n - 2}")
        return self.costs[step] if self.time_dependent else self.costs

    def leg_cost(self, step: int, i: int, j: int) -> float:
        """Cost in km/s of flying i->j as the ``step``-th leg of the sequence."""
        return float(self.leg_costs(step)[i, j])

    def path_cost(self, sequence: Sequence[int]) -> float:
        """Total delta-v (km/s) of an open path. Does not check for duplicates."""
        return sum(
            self.leg_cost(step, i, j)
            for step, (i, j) in enumerate(zip(sequence, sequence[1:], strict=False))
        )

    def is_permutation(self, sequence: Sequence[int]) -> bool:
        """True if ``sequence`` visits every object exactly once."""
        return sorted(sequence) == list(range(self.n))

    def save(self, path: str | Path) -> Path:
        """Write to a .npz and return the path actually written.

        ``np.savez`` appends ``.npz`` when the name lacks it, so the suffix is
        normalised here first -- otherwise the returned path would name a file
        that does not exist.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            costs=self.costs,
            norad_ids=np.asarray(self.norad_ids, dtype=np.int64),
            names=np.asarray(self.names),
            header=json.dumps({"epoch": self.epoch.isoformat(), "metadata": self.metadata}),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ProblemInstance:
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            return cls(
                norad_ids=tuple(int(i) for i in z["norad_ids"]),
                names=tuple(str(s) for s in z["names"]),
                epoch=datetime.fromisoformat(header["epoch"]),
                costs=z["costs"],
                metadata=header["metadata"],
            )


@dataclass(frozen=True)
class Solution:
    """What every solver returns. ``sequence`` holds indices into the instance."""

    sequence: tuple[int, ...]
    total_dv_kms: float
    runtime_s: float
    solver_name: str
    feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def norad_order(self, instance: ProblemInstance) -> tuple[int, ...]:
        """The sequence expressed as NORAD catalog IDs."""
        return tuple(instance.norad_ids[i] for i in self.sequence)


@runtime_checkable
class CostModel(Protocol):
    """Turns catalogue objects into a cost array for a ProblemInstance.

    Implementations live in ``dextrivia/costs/`` (owned by the physics
    workspace). ``build`` returns either (N, N) or (N-1, N, N); see
    ``ProblemInstance``. Row/column order must match the order of ``objects``.
    """

    name: str

    def build(self, objects: Sequence[DebrisObject], epoch: datetime) -> np.ndarray: ...


@runtime_checkable
class Solver(Protocol):
    """Produces a visiting order for an instance.

    ``seed`` must make stochastic solvers reproducible; deterministic solvers
    accept and ignore it. A solver that cannot handle an instance (too large,
    backend missing) returns ``feasible=False`` with the reason in ``metadata``
    rather than raising -- benchmarks need to record misses, not crash on them.
    """

    name: str

    def solve(self, instance: ProblemInstance, seed: int | None = None) -> Solution: ...
