"""
Dextriva — delta-v cost matrix

Uses Hohmann transfer approximation between circular LEO orbits.
Input:  list of altitudes (km)
Output: N×N numpy array of delta-v costs (km/s)

Hohmann transfer assumes:
  - Both orbits circular
  - Coplanar (same inclination) — approximation, fine for demo
  - Instantaneous burns
"""

import numpy as np
import json
from propagation import propagate
from datetime import datetime, timezone

MU = 398600.4418   # Earth gravitational parameter km³/s²
RE = 6371.0        # Earth radius km


def hohmann_deltav(r1: float, r2: float) -> float:
    """
    Delta-v for Hohmann transfer between two circular orbits.

    Parameters
    ----------
    r1 : float  orbital radius of origin (km from Earth center)
    r2 : float  orbital radius of target (km from Earth center)

    Returns
    -------
    float  total delta-v in km/s
    """
    if abs(r1 - r2) < 0.1:
        return 0.0  # same orbit, no transfer needed

    a_transfer = (r1 + r2) / 2.0   # semi-major axis of transfer ellipse

    v1      = np.sqrt(MU / r1)
    v_trans1 = np.sqrt(MU * (2/r1 - 1/a_transfer))
    dv1     = abs(v_trans1 - v1)

    v2      = np.sqrt(MU / r2)
    v_trans2 = np.sqrt(MU * (2/r2 - 1/a_transfer))
    dv2     = abs(v2 - v_trans2)

    return dv1 + dv2


def build_cost_matrix(altitudes_km: list[float]) -> np.ndarray:
    """
    Build N×N delta-v cost matrix from a list of orbital altitudes.

    Parameters
    ----------
    altitudes_km : list of float
        Altitude above Earth surface in km for each debris object.

    Returns
    -------
    np.ndarray shape (N, N)
        cost[i][j] = delta-v in km/s to travel from object i to object j.
        Diagonal is 0 (no transfer needed to stay at same object).
    """
    n = len(altitudes_km)
    radii = [alt + RE for alt in altitudes_km]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = hohmann_deltav(radii[i], radii[j])

    return matrix


def get_altitudes_from_tles(tles: list[dict], t: datetime) -> list[float]:
    """
    Propagate all TLEs to time t and extract altitudes.
    Skips objects with propagation errors.

    Returns list of (index, name, altitude_km) tuples.
    """
    results = []
    for i, obj in enumerate(tles):
        r = propagate(obj["line1"], obj["line2"], t)
        if r["error"] == 0 and not np.isnan(r["altitude_km"]):
            results.append((i, obj["name"], r["altitude_km"]))
    return results