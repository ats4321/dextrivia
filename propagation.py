"""
Dextriva — orbital propagation module

SGP4 output frame: ECI (Earth-Centered Inertial)
  Origin: Earth's center
  Axes: fixed to stars, not rotating with Earth
  Position units: km
  Velocity units: km/s
"""

from datetime import datetime, timezone
import numpy as np
from sgp4.api import Satrec, jday

EARTH_RADIUS_KM = 6371.0


def propagate(line1: str, line2: str, t) -> dict:
    """
    Propagate a TLE to one or more UTC datetimes.

    Parameters
    ----------
    line1 : str   — TLE line 1 (starts with '1 ')
    line2 : str   — TLE line 2 (starts with '2 ')
    t     : datetime or list[datetime]
            Must be timezone-aware UTC.
            e.g. datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    Returns
    -------
    dict:
        'position_km'  : np.ndarray (3,) or (N,3)  — ECI position in km
        'velocity_kms' : np.ndarray (3,) or (N,3)  — ECI velocity in km/s
        'altitude_km'  : float or np.ndarray        — altitude above spherical Earth
        'error'/'errors' : int or list[int]          — 0 means success
    """
    sat = Satrec.twoline2rv(line1, line2)

    scalar = isinstance(t, datetime)
    times = [t] if scalar else t

    positions, velocities, errors = [], [], []

    for dt in times:
        jd, fr = jday(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute,
            dt.second + dt.microsecond / 1e6
        )
        e, pos, vel = sat.sgp4(jd, fr)
        errors.append(e)
        if e == 0:
            positions.append(pos)
            velocities.append(vel)
        else:
            positions.append((float('nan'),) * 3)
            velocities.append((float('nan'),) * 3)

    pos_arr = np.array(positions)           # (N, 3) km
    vel_arr = np.array(velocities)          # (N, 3) km/s
    alts    = np.linalg.norm(pos_arr, axis=1) - EARTH_RADIUS_KM

    if scalar:
        return {
            'position_km':  pos_arr[0],
            'velocity_kms': vel_arr[0],
            'altitude_km':  float(alts[0]),
            'error':        errors[0],
        }
    return {
        'position_km':  pos_arr,
        'velocity_kms': vel_arr,
        'altitude_km':  alts,
        'errors':       errors,
    }