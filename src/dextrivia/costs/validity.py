"""How far SGP4 can be trusted here, and what to do with objects it cannot propagate.

SGP4 is a fitted model. A TLE is a least-squares fit to a few days of tracking;
propagated months away from its epoch, drag and fit errors accumulate. Every
time-dependent instance prices leg ``k`` at ``epoch + k * delta``, so an
instance's reach is ``(N-2) * delta`` plus the age of its oldest TLE -- 561 days
for the v1 ``n20_td30d`` instance. This module makes that reach explicit:

``VALIDITY_HORIZON_DAYS``
    The longest propagation span this project's evidence supports. MEASURED,
    not assumed: ``scripts/validate_propagation.py`` propagated every object of
    ``iridium33_20260402.json`` to the epochs of ``iridium33_20260928.json``
    and compared against the real later TLEs (``docs/physics.md`` section 9).
    That is ONE horizon (~177 days), not an error-vs-time curve, and the TLE
    fit-noise floor is not separable from propagation error with two snapshots.

    The rule, fixed before the numbers were looked at: the horizon is the
    longest span at which the 90th-percentile error in an in-cluster leg cost
    (pairs within 20 deg of plane) stays at or below 5% of the median
    in-cluster leg cost, scaled linearly down from the measured point and
    CAPPED at the measured span. At the measured span the error was inside the
    tolerance (0.037 vs 0.065 km/s), so the cap binds: the horizon is the
    measured span, floored to whole days. Nothing beyond it has been checked.

``screen_decay``
    Objects SGP4 cannot propagate to every time a cost model will price are
    excluded and recorded, never allowed to crash ``mean_elements``. An SGP4
    error 6 is SGP4's own verdict that the object has decayed.

    This screen catches only what SGP4 predicts. Object 35080 re-entered on
    2026-07-10 (Celestrak SATCAT), 99 days after the April epoch; SGP4 kept
    propagating it until day 219 (2026-11-07), 120 days late. SGP4 is late
    about decay, which is one more reason to stay inside the horizon.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

import numpy as np

from dextrivia.core import DebrisObject, ProblemInstance
from dextrivia.propagation import satrec, tle_epoch

__all__ = [
    "VALIDITY_HORIZON_DAYS",
    "HORIZON_SOURCE",
    "PropagationHorizonWarning",
    "mission_times",
    "screen_decay",
    "propagation_span_days",
    "horizon_metadata",
    "check_instance",
    "compare_snapshots",
    "horizon_from_comparison",
]

#: Measured single-horizon validity limit. See the module docstring.
VALIDITY_HORIZON_DAYS = 177.0
HORIZON_SOURCE = (
    "docs/physics.md section 9: iridium33_20260402 propagated to iridium33_20260928 "
    "(single ~177-day horizon, two snapshots; fit-noise floor unmeasured)"
)

_SECONDS_PER_DAY = 86400.0


class PropagationHorizonWarning(UserWarning):
    """An instance prices a leg further from its TLE epochs than the horizon."""


def mission_times(cost_model: object, epoch: datetime, n: int) -> list[datetime]:
    """Every wall-clock time ``cost_model`` will propagate to for an ``n``-object instance.

    Static models price everything at ``epoch``; time-slotted ones at each leg's
    departure time, including loiter-window samples.
    """
    if getattr(cost_model, "time_dependent", False):
        return [t for leg in cost_model.departure_times(epoch, n - 1) for t in leg]
    return [epoch]


def screen_decay(
    objects: Sequence[DebrisObject], times: Iterable[datetime]
) -> tuple[tuple[DebrisObject, ...], tuple[dict[str, object], ...]]:
    """Split ``objects`` into those SGP4 propagates at every time and those it does not.

    Returns ``(kept, excluded)``; each excluded record names the NORAD id, the
    SGP4 error code and the first time it failed, so metadata can say why.
    """
    from sgp4.conveniences import jday_datetime

    times = sorted(times)
    kept, excluded = [], []
    for obj in objects:
        sat = satrec(obj.line1, obj.line2)
        for t in times:
            error, _, _ = sat.sgp4(*jday_datetime(t))
            if error != 0:
                excluded.append(
                    {"norad_id": obj.norad_id, "sgp4_error": int(error), "at": t.isoformat()}
                )
                break
        else:
            kept.append(obj)
    return tuple(kept), tuple(excluded)


def propagation_span_days(objects: Sequence[DebrisObject], times: Iterable[datetime]) -> float:
    """Largest ``|t - TLE epoch|`` over the objects and times, in days.

    Both directions count: SGP4 degrades away from its epoch either way.
    """
    times = list(times)
    epochs = [tle_epoch(o.line1, o.line2) for o in objects]
    forward = max(times) - min(epochs)
    backward = max(epochs) - min(times)
    return max(forward, backward, timedelta(0)).total_seconds() / _SECONDS_PER_DAY


def horizon_metadata(
    objects: Sequence[DebrisObject],
    times: Iterable[datetime],
    horizon_days: float = VALIDITY_HORIZON_DAYS,
    label: str = "instance",
) -> dict[str, object]:
    """Metadata keys recording the propagation span, and a warning if it exceeds the horizon."""
    span = propagation_span_days(objects, times)
    exceeds = span > horizon_days
    if exceeds:
        warnings.warn(
            f"{label} propagates {span:.1f} days from its TLE epochs, beyond the "
            f"{horizon_days:g}-day validity horizon ({HORIZON_SOURCE})",
            PropagationHorizonWarning,
            stacklevel=3,
        )
    return {
        "propagation_span_days": round(span, 3),
        "validity_horizon_days": horizon_days,
        "validity_horizon_source": HORIZON_SOURCE,
        "exceeds_validity_horizon": exceeds,
    }


def check_instance(
    instance: ProblemInstance, snapshot_dir=None, horizon_days: float = VALIDITY_HORIZON_DAYS
) -> dict[str, object]:
    """Horizon status of a SAVED instance, rebuilt from its metadata and snapshot.

    For instances built before the flag existed (the v1 family), which stay
    byte-identical on disk. Warns exactly like a fresh build would.
    """
    from dextrivia.snapshots import Snapshot, default_snapshot_dir

    meta = instance.metadata
    snapshot = Snapshot.load((snapshot_dir or default_snapshot_dir()) / str(meta["snapshot"]))
    by_id = {o.norad_id: o for o in snapshot.objects}
    objects = [by_id[i] for i in instance.norad_ids]

    delta = meta.get("delta_per_leg_days")
    window = float(meta.get("window_days") or 0.0)
    if delta is None:
        times = [instance.epoch]
    else:
        last = instance.epoch + timedelta(days=float(delta) * (instance.n - 2) + window)
        times = [instance.epoch, last]
    return horizon_metadata(objects, times, horizon_days, label=str(meta["snapshot"]))


def compare_snapshots(
    old: Sequence[DebrisObject],
    new: Sequence[DebrisObject],
    reference_time: datetime,
    cluster_angle_deg: float = 20.0,
) -> dict[str, object]:
    """Propagate ``old`` TLEs forward and compare against the ``new`` ones.

    Per object (matched by NORAD id), at the NEW TLE's own epoch: mean RAAN,
    inclination and semi-major-axis error, predicted minus actual. Per pair of
    objects within ``cluster_angle_deg`` of plane (actual), at ``reference_time``:
    predicted vs actual plane angle and impulsive leg cost -- the quantity the
    cost model actually uses, in which common-mode RAAN error cancels.

    Objects the old TLE cannot reach are reported in ``sgp4_failed``; objects in
    only one list are reported, not guessed about.
    """
    from dextrivia.costs.realistic import impulsive_dv, mean_elements, plane_angle

    old_by_id = {o.norad_id: o for o in old}
    new_by_id = {o.norad_id: o for o in new}
    common = sorted(set(old_by_id) & set(new_by_id))

    objects, failed = [], []
    predicted_ref, actual_ref = {}, {}
    for norad in common:
        o, n = old_by_id[norad], new_by_id[norad]
        t_new = tle_epoch(n.line1, n.line2)
        try:
            p = mean_elements(o.line1, o.line2, t_new)
            predicted_ref[norad] = mean_elements(o.line1, o.line2, reference_time)
        except RuntimeError as exc:
            failed.append({"norad_id": norad, "error": str(exc)})
            continue
        q = mean_elements(n.line1, n.line2, t_new)
        actual_ref[norad] = mean_elements(n.line1, n.line2, reference_time)
        span = (t_new - tle_epoch(o.line1, o.line2)).total_seconds() / _SECONDS_PER_DAY
        objects.append(
            {
                "norad_id": norad,
                "span_days": span,
                "raan_err_deg": (math.degrees(p.raan_rad - q.raan_rad) + 180.0) % 360.0 - 180.0,
                "inc_err_deg": math.degrees(p.inc_rad - q.inc_rad),
                "sma_err_km": p.a_km - q.a_km,
                "actual_altitude_km": q.a_km - 6378.135,
            }
        )

    pairs = []
    for i, j in itertools.combinations(sorted(actual_ref), 2):
        qi, qj, pi, pj = actual_ref[i], actual_ref[j], predicted_ref[i], predicted_ref[j]
        theta = plane_angle(qi.inc_rad, qi.raan_rad, qj.inc_rad, qj.raan_rad)
        if math.degrees(theta) > cluster_angle_deg:
            continue
        theta_p = plane_angle(pi.inc_rad, pi.raan_rad, pj.inc_rad, pj.raan_rad)
        actual = impulsive_dv(qi.a_km, qj.a_km, theta)
        pairs.append(
            {
                "norad_ids": (i, j),
                "actual_angle_deg": math.degrees(theta),
                "angle_err_deg": math.degrees(theta_p - theta),
                "actual_dv_kms": actual,
                "dv_err_kms": impulsive_dv(pi.a_km, pj.a_km, theta_p) - actual,
            }
        )

    return {
        "objects": objects,
        "pairs": pairs,
        "sgp4_failed": failed,
        "only_in_old": sorted(set(old_by_id) - set(new_by_id)),
        "only_in_new": sorted(set(new_by_id) - set(old_by_id)),
    }


def horizon_from_comparison(
    comparison: dict[str, object], tolerance_fraction: float = 0.05, percentile: float = 90.0
) -> dict[str, float]:
    """Apply the pre-registered horizon rule (module docstring) to a comparison."""
    pairs = comparison["pairs"]
    errors = np.abs([p["dv_err_kms"] for p in pairs])
    measured_span = float(np.median([o["span_days"] for o in comparison["objects"]]))
    tolerance = tolerance_fraction * float(np.median([p["actual_dv_kms"] for p in pairs]))
    error = float(np.percentile(errors, percentile))
    # Linear scaling is conservative if the true growth is super-linear (drag).
    scaled = measured_span * tolerance / error if error > 0 else math.inf
    return {
        "measured_span_days": measured_span,
        "tolerance_kms": tolerance,
        "error_at_span_kms": error,
        "horizon_days": float(math.floor(min(measured_span, scaled))),
    }
