"""Hohmann cost model invariants."""

from __future__ import annotations

import numpy as np
import pytest

from dextrivia.costs.hohmann import HohmannCostModel, hohmann_dv, hohmann_matrix
from dextrivia.propagation import R_EARTH_WGS72_KM as RE

RADII = [RE + alt for alt in (350.0, 600.0, 771.0, 780.0, 1200.0)]


def test_matrix_diagonal_is_zero():
    matrix = hohmann_matrix(RADII)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(len(RADII)))


def test_matrix_is_symmetric():
    """Coplanar Hohmann costs the same in both directions -- raising and lowering
    an orbit are mirror manoeuvres. A cost model that models plane changes or
    RAAN drift need NOT be symmetric; nothing downstream assumes it is."""
    matrix = hohmann_matrix(RADII)
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)


def test_dv_is_positive_off_diagonal():
    matrix = hohmann_matrix(RADII)
    assert (matrix[~np.eye(len(RADII), dtype=bool)] > 0).all()


def test_dv_matches_textbook_400_to_800_km():
    # Standard LEO example: ~0.217 km/s total for a 400 -> 800 km circular transfer.
    assert hohmann_dv(RE + 400.0, RE + 800.0) == pytest.approx(0.2167, rel=1e-3)


def test_dv_grows_with_separation():
    base = RE + 700.0
    costs = [hohmann_dv(base, base + gap) for gap in (10.0, 50.0, 200.0, 1000.0)]
    assert costs == sorted(costs)


def test_dv_obeys_the_triangle_inequality_along_a_radius():
    """Stopping off at an intermediate orbit never saves delta-v. This is what
    makes altitude-only costs degenerate; see tests/test_degeneracy.py."""
    a, b, c = RE + 400.0, RE + 700.0, RE + 1100.0
    assert hohmann_dv(a, c) <= hohmann_dv(a, b) + hohmann_dv(b, c) + 1e-12


def test_non_positive_radius_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        hohmann_dv(-1.0, RE + 700.0)


def test_cost_model_builds_a_square_matrix_for_real_objects(snapshot):
    objects = snapshot.select(6)
    matrix = HohmannCostModel().build(objects, snapshot.median_epoch())
    assert matrix.shape == (6, 6)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T)
