"""Plane-aware cost models, checked against values you can look up.

Every test here pins a number that exists outside this repository -- a textbook
transfer, a closed-form limiting case, or a published orbit's nodal drift --
rather than whatever the code happened to print the day it was written.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from dextrivia.costs.hohmann import hohmann_dv
from dextrivia.costs.realistic import (
    EdelbaumCostModel,
    ImpulsiveCostModel,
    edelbaum_dv,
    impulsive_dv,
    mean_elements,
    nodal_precession_deg_per_day,
    plane_angle,
    separate_burn_dv,
)
from dextrivia.propagation import MU_WGS72_KM3_S2 as MU
from dextrivia.propagation import R_EARTH_WGS72_KM as RE

GEO_RADIUS_KM = 42164.0
LEO_PARKING_KM = RE + 300.0


def circular_speed(r: float) -> float:
    return float(np.sqrt(MU / r))


# --------------------------------------------------------------------------
# plane geometry
# --------------------------------------------------------------------------


def test_identical_planes_have_zero_angle():
    i, raan = np.radians(86.4), np.radians(120.0)
    assert plane_angle(i, raan, i, raan) == pytest.approx(0.0, abs=1e-12)


def test_opposite_raan_doubles_the_inclination():
    """i1 = i2 = i, RAAN 180 deg apart -> the planes differ by 2i (mod 180).

    This is why RAAN cannot be ignored for the Iridium-33 cloud: two objects
    that differ by 0.0 deg of inclination can still be 172 deg of plane apart.
    """
    i = np.radians(86.0)
    theta = plane_angle(i, 0.0, i, np.pi)
    assert np.degrees(theta) == pytest.approx(172.0, abs=1e-9)


def test_polar_planes_90_deg_of_raan_apart_are_perpendicular():
    theta = plane_angle(np.pi / 2, 0.0, np.pi / 2, np.pi / 2)
    assert np.degrees(theta) == pytest.approx(90.0, abs=1e-9)


# --------------------------------------------------------------------------
# impulsive model
# --------------------------------------------------------------------------


def test_leo_to_geo_hohmann_is_the_textbook_3_9_kms():
    """300 km circular -> GEO, coplanar: the canonical ~3.9 km/s two-burn total."""
    assert impulsive_dv(LEO_PARKING_KM, GEO_RADIUS_KM, 0.0) == pytest.approx(3.9, abs=0.05)


def test_zero_plane_angle_reduces_exactly_to_hohmann():
    for r2 in (RE + 400.0, RE + 1200.0, GEO_RADIUS_KM):
        assert impulsive_dv(RE + 300.0, r2, 0.0) == pytest.approx(
            hohmann_dv(RE + 300.0, r2), rel=1e-12
        )


@pytest.mark.parametrize("degrees", [0.5, 5.0, 28.5, 60.0, 120.0, 180.0])
def test_same_orbit_plane_change_is_the_textbook_2v_sin_half(degrees):
    """r1 == r2 -> the model must collapse onto dv = 2 v sin(di/2)."""
    r = RE + 780.0
    theta = np.radians(degrees)
    expected = 2.0 * circular_speed(r) * np.sin(theta / 2.0)
    assert impulsive_dv(r, r, theta) == pytest.approx(expected, rel=1e-9)


def test_geo_insertion_with_a_28_5_deg_plane_change():
    """Cape-latitude GTO -> GEO. Combined ~4.2 km/s, separate burns ~5.4 km/s.

    Both are standard back-of-envelope numbers; the point of the test is the
    ~1.2 km/s that folding the plane change into the burns saves.
    """
    theta = np.radians(28.5)
    combined = impulsive_dv(LEO_PARKING_KM, GEO_RADIUS_KM, theta)
    separate = separate_burn_dv(LEO_PARKING_KM, GEO_RADIUS_KM, theta)
    assert combined == pytest.approx(4.23, abs=0.05)
    assert separate == pytest.approx(5.41, abs=0.05)
    assert separate - combined > 1.0


@pytest.mark.parametrize("degrees", [0.0, 1.0, 10.0, 45.0, 90.0, 170.0])
@pytest.mark.parametrize("alt2", [520.0, 700.0, 900.0, 35786.0])
def test_combining_burns_never_costs_more_than_doing_them_separately(degrees, alt2):
    """``separate_burn_dv`` is an upper bound on ``impulsive_dv``, always."""
    r1, r2 = RE + 600.0, RE + alt2
    theta = np.radians(degrees)
    assert impulsive_dv(r1, r2, theta) <= separate_burn_dv(r1, r2, theta) + 1e-12


def test_the_optimal_split_beats_both_all_at_the_start_and_all_at_the_end():
    """The split really is interior: neither burn should do the whole rotation."""
    r1, r2 = RE + 600.0, RE + 1400.0
    theta = np.radians(40.0)
    a_t = (r1 + r2) / 2.0
    v1, v2 = circular_speed(r1), circular_speed(r2)
    vt1 = np.sqrt(MU * (2.0 / r1 - 1.0 / a_t))
    vt2 = np.sqrt(MU * (2.0 / r2 - 1.0 / a_t))

    def total(split: float) -> float:
        return float(
            np.sqrt(v1**2 + vt1**2 - 2 * v1 * vt1 * np.cos(split))
            + np.sqrt(v2**2 + vt2**2 - 2 * v2 * vt2 * np.cos(theta - split))
        )

    best = impulsive_dv(r1, r2, theta)
    assert best < total(0.0)
    assert best < total(theta)


def test_plane_change_dominates_altitude_change_in_this_cloud():
    """1 deg of plane costs more than 100 km of altitude at 700 km.

    This is the entire reason the coplanar model was degenerate.
    """
    r = RE + 700.0
    one_degree = impulsive_dv(r, r, np.radians(1.0))
    hundred_km = impulsive_dv(r, r + 100.0, 0.0)
    assert one_degree > hundred_km


def test_negative_inputs_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        impulsive_dv(-1.0, RE + 700.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        impulsive_dv(RE + 700.0, RE + 800.0, -0.1)


# --------------------------------------------------------------------------
# Edelbaum low-thrust model
# --------------------------------------------------------------------------


def test_edelbaum_with_no_plane_change_is_the_spiral_speed_difference():
    r1, r2 = RE + 500.0, RE + 900.0
    assert edelbaum_dv(r1, r2, 0.0) == pytest.approx(
        abs(circular_speed(r2) - circular_speed(r1)), rel=1e-12
    )


@pytest.mark.parametrize("degrees", [1.0, 30.0, 90.0, 114.0])
def test_edelbaum_at_constant_radius_is_2v_sin_quarter_pi_di(degrees):
    """Below the 114.6 deg clamp, where the closed form is still physical."""
    r = RE + 780.0
    theta = np.radians(degrees)
    expected = 2.0 * circular_speed(r) * np.sin(np.pi * theta / 4.0)
    assert edelbaum_dv(r, r, theta) == pytest.approx(expected, rel=1e-12)


def test_low_thrust_loses_to_impulsive_for_small_plane_changes_by_pi_over_two():
    """Small angles: low thrust pays pi/2 times the impulsive cost."""
    r = RE + 780.0
    theta = np.radians(0.5)
    assert edelbaum_dv(r, r, theta) / impulsive_dv(r, r, theta) == pytest.approx(
        np.pi / 2, rel=1e-3
    )


@pytest.mark.parametrize("degrees", [0.5, 10.0, 45.0, 90.0, 114.0])
def test_low_thrust_never_beats_impulsive_on_delta_v(degrees):
    """Low thrust wins on propellant mass, never on delta-v. See ``edelbaum_dv``."""
    r1, r2 = RE + 600.0, RE + 900.0
    theta = np.radians(degrees)
    assert edelbaum_dv(r1, r2, theta) >= impulsive_dv(r1, r2, theta) - 1e-12


def test_edelbaum_is_clamped_where_the_formula_stops_being_physical():
    """Above ~114.6 deg the raw expression decreases with plane angle; clamping
    it at ``v1 + v2`` keeps the model monotonic instead of offering a discount
    for a bigger plane change."""
    r = RE + 780.0
    v = circular_speed(r)
    angles = np.radians([100.0, 114.6, 130.0, 180.0])
    values = [edelbaum_dv(r, r, a) for a in angles]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(2.0 * v, rel=1e-12)


# --------------------------------------------------------------------------
# J2 nodal precession
# --------------------------------------------------------------------------


def test_iss_like_orbit_regresses_about_5_degrees_per_day():
    """ISS: ~420 km circular, 51.64 deg -> the widely quoted -5 deg/day."""
    rate = nodal_precession_deg_per_day(RE + 400.0, 0.0005, np.radians(51.64))
    assert rate == pytest.approx(-5.0, abs=0.1)


def test_sun_synchronous_orbit_precesses_with_the_earth_around_the_sun():
    """~700 km at 98.2 deg gives +0.9856 deg/day, one revolution per year."""
    rate = nodal_precession_deg_per_day(RE + 700.0, 0.001, np.radians(98.2))
    assert rate == pytest.approx(0.9856, abs=0.01)


def test_polar_orbits_do_not_precess_and_the_sign_follows_cos_i():
    a = RE + 700.0
    assert nodal_precession_deg_per_day(a, 0.0, np.pi / 2) == pytest.approx(0.0, abs=1e-12)
    assert nodal_precession_deg_per_day(a, 0.0, np.radians(45.0)) < 0.0  # prograde: regresses
    assert nodal_precession_deg_per_day(a, 0.0, np.radians(135.0)) > 0.0  # retrograde: advances


def test_the_iridium_cloud_regresses_about_0_44_degrees_per_day(snapshot):
    """Measured on the committed snapshot; the spread is what makes legs differ."""
    epoch = snapshot.median_epoch()
    rates = [
        nodal_precession_deg_per_day(*mean_elements(o.line1, o.line2, epoch)[:3])
        for o in snapshot.objects
    ]
    assert np.median(rates) == pytest.approx(-0.44, abs=0.02)
    assert max(rates) - min(rates) == pytest.approx(0.10, abs=0.02)


def test_the_analytic_rate_matches_sgp4s_own_node_regression(snapshot):
    """Within 2% -- first-order J2 secular theory against the full propagator."""
    epoch = snapshot.median_epoch()
    for obj in snapshot.objects[:8]:
        start = mean_elements(obj.line1, obj.line2, epoch)
        end = mean_elements(obj.line1, obj.line2, epoch + timedelta(days=100))
        observed = np.degrees((end.raan_rad - start.raan_rad + np.pi) % (2 * np.pi) - np.pi) / 100.0
        analytic = nodal_precession_deg_per_day(start.a_km, start.ecc, start.inc_rad)
        assert analytic == pytest.approx(observed, rel=0.02)


# --------------------------------------------------------------------------
# CostModel plumbing
# --------------------------------------------------------------------------


def test_static_model_builds_a_symmetric_matrix_with_a_zero_diagonal(snapshot):
    objects = snapshot.select(6)
    costs = ImpulsiveCostModel().build(objects, snapshot.median_epoch())
    assert costs.shape == (6, 6)
    assert np.isfinite(costs).all()
    np.testing.assert_allclose(costs, costs.T, rtol=1e-12)
    np.testing.assert_allclose(np.diag(costs), 0.0, atol=1e-12)


def test_time_dependent_model_has_one_slot_per_leg(snapshot):
    objects = snapshot.select(5)
    costs = ImpulsiveCostModel(delta_per_leg_days=30.0).build(objects, snapshot.median_epoch())
    assert costs.shape == (4, 5, 5)
    # Slots must actually differ: if J2 drift did nothing the extra axis would
    # be a lie and the exact solver would be wasting its time on it.
    assert not np.allclose(costs[0], costs[-1])


def test_raan_drift_moves_costs_by_a_meaningful_amount_over_a_long_mission(snapshot):
    """Differential drift is slow -- but 19 legs x 30 days is not short."""
    objects = snapshot.select(10)
    costs = ImpulsiveCostModel(delta_per_leg_days=30.0).build(objects, snapshot.median_epoch())
    drift = np.abs(costs[-1] - costs[0])
    assert drift.max() > 0.01  # km/s, i.e. not numerical noise
    assert drift.max() < 2.0  # and not a run-away either


def test_catalogue_order_targets_are_absurdly_expensive(snapshot):
    """The motivation for plane-cluster selection, as a number.

    Taking objects in catalogue order mixes planes, and the median leg costs
    more than a launch to GEO. This is what the old coplanar model priced at
    ~0.01 km/s per leg.
    """
    objects = snapshot.select(12)
    costs = ImpulsiveCostModel().build(objects, snapshot.median_epoch())
    off_diagonal = costs[~np.eye(len(objects), dtype=bool)]
    assert np.median(off_diagonal) > 1.0


def test_model_names_record_the_time_mapping():
    assert ImpulsiveCostModel().name == "impulsive-plane-static"
    assert ImpulsiveCostModel(delta_per_leg_days=30.0).name == "impulsive-plane-j2-30d"
    assert EdelbaumCostModel().name == "edelbaum-static"


def test_a_loiter_window_can_only_lower_costs(snapshot):
    objects = snapshot.select(6)
    epoch = snapshot.median_epoch()
    nominal = ImpulsiveCostModel(delta_per_leg_days=30.0).build(objects, epoch)
    windowed = ImpulsiveCostModel(
        delta_per_leg_days=30.0, window_days=20.0, window_samples=5
    ).build(objects, epoch)
    assert (windowed <= nominal + 1e-12).all()


def test_nonsense_model_configurations_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        ImpulsiveCostModel(delta_per_leg_days=0.0)
    with pytest.raises(ValueError, match="static"):
        ImpulsiveCostModel(window_days=5.0, window_samples=3)
    with pytest.raises(ValueError, match="2 samples"):
        ImpulsiveCostModel(delta_per_leg_days=30.0, window_days=5.0, window_samples=1)
