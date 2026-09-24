"""The committed plane-cluster instance family, and proof that it is not trivial.

``tests/test_degeneracy.py`` documents what was wrong with the altitude-only
benchmark: sorting by altitude was optimal, and greedy tied the exact optimum,
so no solver could ever demonstrate anything. These tests are the other half of
that statement -- with plane geometry in the cost model, altitude order is far
from optimal on every committed instance, and greedy is strictly beaten by
Held-Karp on at least one of them.

The instances themselves are committed (see ``.gitignore``): a published gap
number has to refer to a fixed array, not to whatever the generator produces
today. Regenerate with ``scripts/build_instance_family.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dextrivia.core import ProblemInstance
from dextrivia.costs.realistic import ImpulsiveCostModel, mean_elements
from dextrivia.costs.selection import build_cluster_instance
from dextrivia.snapshots import default_snapshot_dir
from dextrivia.solvers import ExactSolver, GreedySolver
from dextrivia.solvers.exact import HELD_KARP_MAX_N

FAMILY_VERSION = "v1"
FAMILY_SIZES = (4, 5, 8, 10, 15, 20)
FAMILY_DELTA_DAYS = 30.0
FAMILY_PREFIX = "iridium33_20260402_planecluster-v1"

#: The instance the "greedy is not optimal any more" claim is pinned to.
GREEDY_LOSES_ON = f"{FAMILY_PREFIX}_n8_td30d.npz"


def instance_path(name: str):
    return default_snapshot_dir().parent / "instances" / name


def family_names() -> list[str]:
    return [
        f"{FAMILY_PREFIX}_n{n}_{variant}.npz"
        for n in FAMILY_SIZES
        for variant in ("static", "td30d")
    ]


@pytest.fixture(scope="module")
def family() -> dict[str, ProblemInstance]:
    return {name: ProblemInstance.load(instance_path(name)) for name in family_names()}


def altitude_order(instance: ProblemInstance, snapshot) -> tuple[int, ...]:
    by_id = {o.norad_id: o for o in snapshot.objects}
    radii = [
        mean_elements(by_id[i].line1, by_id[i].line2, instance.epoch).a_km
        for i in instance.norad_ids
    ]
    return tuple(int(i) for i in np.argsort(radii))


@pytest.mark.parametrize("name", family_names())
def test_every_family_member_is_committed_and_loadable(name):
    assert instance_path(name).exists(), f"{name} missing; run scripts/build_instance_family.py"
    ProblemInstance.load(instance_path(name))


@pytest.mark.parametrize("name", family_names())
def test_shapes_follow_the_open_path_convention(name, family):
    instance = family[name]
    n = int(name.split("_n")[1].split("_")[0])
    assert instance.n == n
    if name.endswith("td30d.npz"):
        assert instance.costs.shape == (n - 1, n, n)  # N-1 legs, never N
    else:
        assert instance.costs.shape == (n, n)


@pytest.mark.parametrize("name", family_names())
def test_provenance_is_complete_enough_to_rebuild_the_array(name, family):
    metadata = family[name].metadata
    assert metadata["snapshot"] == "iridium33_20260402.json"
    assert metadata["family_version"] == FAMILY_VERSION
    assert metadata["selection_rule"] == "plane-cluster"
    assert metadata["cost_model_family"] == "impulsive-plane"
    assert metadata["epoch_source"] == "snapshot-median-tle-epoch"
    assert metadata["cluster_seed_norad"] in family[name].norad_ids
    for key in ("raan_window_deg", "inc_window_deg", "alt_band_km"):
        assert isinstance(metadata[key], (int, float))
    if name.endswith("td30d.npz"):
        assert metadata["delta_per_leg_days"] == FAMILY_DELTA_DAYS
        assert "epoch + k * delta_per_leg_days" in metadata["leg_departure_rule"]
    else:
        assert metadata["delta_per_leg_days"] is None


def test_the_committed_arrays_still_match_what_the_generator_produces(snapshot, family):
    """Guards against a silent physics change orphaning the committed numbers."""
    for delta, suffix in ((None, "static"), (FAMILY_DELTA_DAYS, "td30d")):
        rebuilt = build_cluster_instance(
            snapshot, 8, ImpulsiveCostModel(delta_per_leg_days=delta), family_version="v1"
        )
        committed = family[f"{FAMILY_PREFIX}_n8_{suffix}.npz"]
        assert rebuilt.norad_ids == committed.norad_ids
        np.testing.assert_allclose(rebuilt.costs, committed.costs, rtol=1e-12)


@pytest.mark.parametrize("name", family_names())
def test_selection_really_did_cluster_the_planes(name, family):
    """Every pair shares a plane to within the recorded window (x2: it is a
    radius around the seed, so the cluster's own span can be twice as wide)."""
    metadata = family[name].metadata
    assert metadata["cluster_max_plane_angle_deg"] <= 2 * metadata["raan_window_deg"]


@pytest.mark.parametrize("name", family_names())
def test_altitude_order_no_longer_solves_the_problem(name, family, snapshot):
    """The headline: the old model's optimum is now 50%+ worse than optimal.

    Under ``hohmann.py`` visiting in altitude order WAS the exact optimum
    (tests/test_degeneracy.py). Here it is beaten on every single instance.
    """
    instance = family[name]
    if instance.n > HELD_KARP_MAX_N:
        pytest.skip(f"N={instance.n} is past the Held-Karp limit; no optimum to compare against")
    exact = ExactSolver().solve(instance)
    by_altitude = instance.path_cost(altitude_order(instance, snapshot))
    assert exact.sequence != altitude_order(instance, snapshot)
    assert by_altitude > exact.total_dv_kms * 1.5


def test_greedy_is_strictly_suboptimal_on_a_committed_instance(family):
    """Nearest-neighbour no longer ties the optimum, so there is headroom to beat.

    Pinned to one instance on purpose: greedy still TIES exact on all six
    static members of this family (a static plane-aware matrix is still close
    enough to metric that best-of-all-starts finds the optimum), and the
    honest statement is "time dependence is what breaks greedy", not "greedy is
    bad". See docs/physics.md for the full table.
    """
    instance = family[GREEDY_LOSES_ON]
    greedy = GreedySolver().solve(instance)
    exact = ExactSolver().solve(instance)
    assert exact.feasible and greedy.feasible
    assert greedy.total_dv_kms > exact.total_dv_kms * 1.01
    assert instance.is_permutation(exact.sequence)


def test_the_solvable_instances_are_solvable_and_the_big_one_is_not(family):
    """N=20 exists precisely so the benchmark records a miss, not a crash."""
    exact = ExactSolver().solve(family[f"{FAMILY_PREFIX}_n20_static.npz"])
    assert not exact.feasible
    assert "held-karp limit" in exact.metadata["reason"]

    exact = ExactSolver().solve(family[f"{FAMILY_PREFIX}_n15_static.npz"])
    assert exact.feasible


def test_a_cluster_mission_is_affordable_unlike_a_catalogue_order_one(family):
    """~1-2 km/s for a 10-15 object mission: expensive but not absurd.

    Catalogue-order targets cost >1 km/s PER LEG (tests/test_realistic_costs.py).
    """
    for name in (f"{FAMILY_PREFIX}_n10_static.npz", f"{FAMILY_PREFIX}_n15_static.npz"):
        exact = ExactSolver().solve(family[name])
        assert 0.5 < exact.total_dv_kms < 3.0
