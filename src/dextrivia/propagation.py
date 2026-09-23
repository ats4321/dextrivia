"""SGP4 propagation and mean-element extraction.

SGP4 output frame is ECI (Earth-centred inertial, axes fixed to the stars):
position in km, velocity in km/s.

Two different altitudes appear in this project and they are not interchangeable:

* instantaneous altitude = ``|r(t)| - R_EARTH``. Oscillates once per orbit by
  2*a*e; for the most eccentric object in the Iridium-33 snapshot (e = 0.021)
  that is roughly 285 km peak-to-peak. Useful for sanity checks, useless as a
  transfer-cost input, because it makes the cost depend on where in its orbit
  each object happened to be at the chosen epoch.

* mean semi-major axis = ``Satrec.am * R_EARTH_WGS72``, the Brouwer mean element
  SGP4 itself carries. Free of the once-per-orbit oscillation, and it still
  decays with time (~0.2 km per 100 days for these objects), so the epoch you
  propagate to still matters. This is what the cost models use.

All radii use the WGS72 Earth radius (6378.135 km), because that is the constant
SGP4 uses internally; mixing in 6371 km (mean radius) or 6378.137 (WGS84) would
put a constant offset into every altitude.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime, sat_epoch_datetime

R_EARTH_WGS72_KM = 6378.135
MU_WGS72_KM3_S2 = 398600.8

__all__ = [
    "R_EARTH_WGS72_KM",
    "MU_WGS72_KM3_S2",
    "satrec",
    "tle_epoch",
    "propagate",
    "mean_semi_major_axis_km",
    "mean_altitude_km",
]


def satrec(line1: str, line2: str) -> Satrec:
    """Parse a TLE pair into an SGP4 ``Satrec``."""
    return Satrec.twoline2rv(line1, line2)


def tle_epoch(line1: str, line2: str) -> datetime:
    """The TLE's own epoch as a timezone-aware UTC datetime."""
    return sat_epoch_datetime(satrec(line1, line2))


def propagate(line1: str, line2: str, t: datetime | list[datetime]) -> dict:
    """Propagate a TLE to one or more timezone-aware UTC datetimes.

    Returns a dict with ``position_km`` and ``velocity_kms`` (ECI), the
    instantaneous ``altitude_km`` above the WGS72 sphere, and SGP4's error code
    (0 means success). Scalar input gives scalar ``altitude_km``/``error``;
    list input gives arrays and an ``errors`` list.
    """
    sat = satrec(line1, line2)
    scalar = isinstance(t, datetime)
    times = [t] if scalar else list(t)

    positions, velocities, errors = [], [], []
    for dt in times:
        if dt.tzinfo is None:
            raise ValueError("propagation times must be timezone-aware UTC")
        jd, fr = jday_datetime(dt)
        e, pos, vel = sat.sgp4(jd, fr)
        errors.append(e)
        positions.append(pos if e == 0 else (float("nan"),) * 3)
        velocities.append(vel if e == 0 else (float("nan"),) * 3)

    pos_arr = np.array(positions)
    vel_arr = np.array(velocities)
    alts = np.linalg.norm(pos_arr, axis=1) - R_EARTH_WGS72_KM

    if scalar:
        return {
            "position_km": pos_arr[0],
            "velocity_kms": vel_arr[0],
            "altitude_km": float(alts[0]),
            "error": errors[0],
        }
    return {
        "position_km": pos_arr,
        "velocity_kms": vel_arr,
        "altitude_km": alts,
        "errors": errors,
    }


def mean_semi_major_axis_km(line1: str, line2: str, t: datetime) -> float:
    """Mean semi-major axis (km from Earth's centre) at time ``t``.

    SGP4 updates ``Satrec.am`` -- the Brouwer mean semi-major axis in Earth
    radii -- on every propagation step, so this is the drag-decayed mean
    element at ``t``, not the frozen TLE-epoch value.

    Raises ``RuntimeError`` if SGP4 fails, because an unusable object must not
    silently poison a cost matrix with a plausible-looking number.
    """
    if t.tzinfo is None:
        raise ValueError("propagation times must be timezone-aware UTC")
    sat = satrec(line1, line2)
    jd, fr = jday_datetime(t)
    error, _, _ = sat.sgp4(jd, fr)
    if error != 0:
        raise RuntimeError(f"SGP4 error {error} propagating {sat.satnum} to {t.isoformat()}")
    return float(sat.am) * R_EARTH_WGS72_KM


def mean_altitude_km(line1: str, line2: str, t: datetime) -> float:
    """Mean semi-major axis expressed as an altitude above the WGS72 sphere."""
    return mean_semi_major_axis_km(line1, line2, t) - R_EARTH_WGS72_KM
