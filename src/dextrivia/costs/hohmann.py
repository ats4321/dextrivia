"""Coplanar Hohmann transfer cost model.

KNOWN TO BE WRONG, DELIBERATELY KEPT AS THE BASELINE.

Each object is modelled as a circular orbit at its mean semi-major axis, and the
cost of i->j is the two-burn Hohmann delta-v between those radii. That makes the
cost a function of ONE scalar per object, which collapses the sequencing problem
onto a line: the optimal open path is simply "visit in order of altitude", and
the greedy nearest-neighbour heuristic finds it. See
``tests/test_degeneracy.py`` -- the benchmark is degenerate until a cost model
that uses more than altitude exists.

What is missing, roughly in order of how much delta-v it is worth for the
Iridium-33 cloud: inclination and RAAN differences (plane changes dominate, and
RAAN drift makes the cost depend on WHEN the leg is flown, i.e. a time-slotted
C[t, i, j]), phasing, finite burns, and J2 secular drift of the argument of
perigee. Fixing that is the physics workspace's job.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from dextrivia.core import DebrisObject
from dextrivia.propagation import MU_WGS72_KM3_S2, mean_semi_major_axis_km

__all__ = ["hohmann_dv", "HohmannCostModel"]

# Radii closer than this are treated as the same orbit (no transfer needed).
SAME_ORBIT_TOL_KM = 1e-6


def hohmann_dv(r1: float, r2: float, mu: float = MU_WGS72_KM3_S2) -> float:
    """Two-burn Hohmann delta-v (km/s) between circular orbits of radius r1, r2 (km).

    Symmetric in its arguments and zero when r1 == r2.
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"orbital radii must be positive, got {r1}, {r2}")
    if abs(r1 - r2) < SAME_ORBIT_TOL_KM:
        return 0.0
    a_transfer = (r1 + r2) / 2.0
    dv1 = abs(np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer)) - np.sqrt(mu / r1))
    dv2 = abs(np.sqrt(mu / r2) - np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer)))
    return float(dv1 + dv2)


def hohmann_matrix(radii_km: Sequence[float]) -> np.ndarray:
    """(N, N) Hohmann delta-v matrix from orbital radii (km from Earth's centre)."""
    n = len(radii_km)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dv = hohmann_dv(radii_km[i], radii_km[j])
            matrix[i, j] = matrix[j, i] = dv
    return matrix


class HohmannCostModel:
    """``CostModel`` using mean semi-major axis at ``epoch``. Static (N, N) costs."""

    name = "hohmann-coplanar"

    def build(self, objects: Sequence[DebrisObject], epoch: datetime) -> np.ndarray:
        radii = [mean_semi_major_axis_km(o.line1, o.line2, epoch) for o in objects]
        return hohmann_matrix(radii)
