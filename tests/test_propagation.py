"""Propagation sanity, checked at the snapshot's own epoch rather than at a
hardcoded calendar date."""

from __future__ import annotations

from datetime import timedelta

import pytest

from dextrivia.propagation import (
    R_EARTH_WGS72_KM,
    mean_altitude_km,
    propagate,
    satrec,
    tle_epoch,
)


def test_every_object_propagates_cleanly_at_the_snapshot_median_epoch(snapshot):
    epoch = snapshot.median_epoch()
    for obj in snapshot.objects:
        result = propagate(obj.line1, obj.line2, epoch)
        assert result["error"] == 0, f"{obj.norad_id} failed SGP4"
        assert 200.0 < result["altitude_km"] < 2000.0, obj.norad_id


def test_median_epoch_sits_inside_the_range_of_tle_epochs(snapshot):
    """The default epoch tracks the data. Nothing in this package propagates to
    a hardcoded date -- the pre-package scripts used 2025-01-01, roughly 15
    months BEFORE these TLEs were issued."""
    epochs = snapshot.epochs()
    assert min(epochs) <= snapshot.median_epoch() <= max(epochs)
    assert max(epochs) - min(epochs) < timedelta(days=60)


def test_mean_altitude_lies_between_perigee_and_apogee(snapshot):
    epoch = snapshot.median_epoch()
    for obj in snapshot.objects[:20]:
        sat = satrec(obj.line1, obj.line2)
        mean_alt = mean_altitude_km(obj.line1, obj.line2, epoch)
        perigee = sat.altp * R_EARTH_WGS72_KM
        apogee = sat.alta * R_EARTH_WGS72_KM
        assert perigee - 1.0 <= mean_alt <= apogee + 1.0, obj.norad_id


def test_instantaneous_altitude_swings_but_mean_semi_major_axis_does_not(snapshot):
    """Why the cost models use mean semi-major axis (issue: altitude from |r|).

    The most eccentric object in this snapshot swings hundreds of km per orbit,
    so an instantaneous-altitude cost matrix would mostly encode where each
    object happened to be at the chosen epoch.
    """
    obj = max(snapshot.objects, key=lambda o: satrec(o.line1, o.line2).ecco)
    epoch = tle_epoch(obj.line1, obj.line2)
    period_min = 2 * 3.141592653589793 / satrec(obj.line1, obj.line2).no_kozai

    times = [epoch + timedelta(minutes=period_min * k / 40.0) for k in range(41)]
    instantaneous = propagate(obj.line1, obj.line2, times)["altitude_km"]
    mean = [mean_altitude_km(obj.line1, obj.line2, t) for t in times]

    assert max(instantaneous) - min(instantaneous) > 200.0
    assert max(mean) - min(mean) < 1.0


def test_naive_datetimes_are_rejected(snapshot):
    obj = snapshot.objects[0]
    naive = tle_epoch(obj.line1, obj.line2).replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        propagate(obj.line1, obj.line2, naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        mean_altitude_km(obj.line1, obj.line2, naive)
