"""THE BENCHMARK IS CURRENTLY DEGENERATE. This file documents that, on purpose.

The Hohmann cost model derives every cost from ONE scalar per object (its mean
semi-major axis), so the objects lie on a line and the cheapest open path is
simply "visit them in altitude order". Nearest-neighbour greedy finds that path,
so greedy already equals the exact optimum and there is no headroom for a
smarter solver -- classical, quantum or otherwise -- to show any advantage.

These tests must KEEP PASSING until a cost model that uses more than altitude
lands (physics workspace). When one does, the expected outcome flips: greedy
should start losing to exact, and `test_greedy_ties_the_exact_optimum` is the
test that will fail and should then be rewritten, not deleted.
"""

from __future__ import annotations

import numpy as np
import pytest

from dextrivia.core import ProblemInstance
from dextrivia.costs.hohmann import hohmann_matrix
from dextrivia.propagation import R_EARTH_WGS72_KM, mean_semi_major_axis_km
from dextrivia.solvers import ExactSolver, GreedySolver


def altitude_only_instance(altitudes_km, epoch):
    radii = [R_EARTH_WGS72_KM + a for a in altitudes_km]
    n = len(radii)
    return ProblemInstance(
        norad_ids=tuple(range(1, n + 1)),
        names=tuple(f"obj{i}" for i in range(n)),
        epoch=epoch,
        costs=hohmann_matrix(radii),
    )


@pytest.mark.parametrize("seed", range(20))
def test_sorting_by_altitude_is_optimal_for_altitude_only_costs(seed, snapshot):
    """Sorting -- an O(N log N) non-solver -- matches the exact optimum."""
    rng = np.random.default_rng(seed)
    altitudes = rng.uniform(600.0, 1000.0, size=8)
    inst = altitude_only_instance(altitudes, snapshot.median_epoch())

    sorted_sequence = tuple(np.argsort(altitudes))
    exact = ExactSolver().solve(inst)

    assert exact.total_dv_kms == pytest.approx(inst.path_cost(sorted_sequence))
    assert exact.sequence in (sorted_sequence, sorted_sequence[::-1])


def test_greedy_ties_the_exact_optimum_on_the_real_10_object_instance(benchmark_instance):
    greedy = GreedySolver().solve(benchmark_instance)
    exact = ExactSolver().solve(benchmark_instance)
    assert greedy.total_dv_kms == pytest.approx(exact.total_dv_kms, rel=1e-12)


def test_the_real_10_object_optimum_is_just_altitude_order(snapshot, benchmark_instance):
    objects = snapshot.select(10)
    assert tuple(o.norad_id for o in objects) == benchmark_instance.norad_ids

    radii = [mean_semi_major_axis_km(o.line1, o.line2, benchmark_instance.epoch) for o in objects]
    sorted_sequence = tuple(np.argsort(radii))

    exact = ExactSolver().solve(benchmark_instance)
    assert exact.sequence in (sorted_sequence, sorted_sequence[::-1])
    # Whole mission: ~0.13 km/s, i.e. less than a single small station-keeping
    # burn. A realistic plane-changing model costs orders of magnitude more.
    assert exact.total_dv_kms < 0.2
