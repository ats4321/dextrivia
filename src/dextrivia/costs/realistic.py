"""Cost models that account for orbit-plane geometry, not just altitude.

Why this file exists
--------------------
``hohmann.py`` derives every cost from one scalar per object, which collapses
the sequencing problem onto a line (see ``tests/test_degeneracy.py``). For the
Iridium-33 cloud that is not a small error: inclinations sit in a 0.5 deg band
around 86 deg, but RAANs span the full circle, so the median pairwise plane
angle is 45.5 deg and a median impulsive plane change costs ~5.8 km/s --
roughly 46x the 0.126 km/s the coplanar model reports for an entire 10-object
mission.

Models in this file
-------------------
``ImpulsiveCostModel``   two finite burns, Hohmann transfer with the plane
                         change folded into the burns (chemical servicer).
``EdelbaumCostModel``    continuous low-thrust circle-to-circle transfer with
                         plane change (electric servicer); the standard
                         first-order estimate in active-debris-removal studies.

Both are static ``C[i, j]`` by default and time-slotted ``C[k, i, j]`` when
``delta_per_leg_days`` is set; see "Time slots" below.

Assumptions shared by both models
---------------------------------
1.  **Circular orbits at the mean semi-major axis.** Real eccentricities here
    are 8e-5 .. 2.1e-2 (median 1.8e-3). Ignoring e biases a transfer by roughly
    ``e * v`` ~ 13 m/s at the median and ~160 m/s for the worst object.
2.  **Impulsive burns** (ImpulsiveCostModel): no gravity losses, no finite-burn
    steering losses, no burn duration. Real losses are a few percent for a
    high-thrust stage.
3.  **No phasing.** The servicer is assumed to arrive at the right point in the
    target's orbit for free. In reality phasing is paid either in delta-v or in
    waiting time, and for a dense plane it is usually waiting time. This is the
    single largest omission left; see ``docs/physics.md``.
4.  **Plane change only, no argument-of-perigee or true-anomaly matching.** The
    two orbits are treated as two circles whose planes differ by ``theta``, and
    the burns happen at the line of nodes between them.
5.  **No J2 during the transfer itself** -- J2 enters only through the secular
    RAAN drift between legs (``nodal_precession_rate``).
6.  **No drag make-up, no collision-avoidance margin, no propellant mass.**
    Every cost is delta-v, never mass or time.

Earth constants are WGS72 throughout, matching SGP4 (see ``propagation.py``).

Time slots
----------
``ProblemInstance`` indexes time-dependent costs by POSITION IN THE SEQUENCE:
leg ``k`` is the ``k``-th transfer of the path, and there are exactly N-1 of
them for an open path. This module maps that onto wall clock with one number:

    leg k departs at  t_k = epoch + k * delta_per_leg

``delta_per_leg`` is the whole per-object budget -- transfer, rendezvous,
capture and release -- and is recorded in the instance metadata as
``delta_per_leg_days``. It is deliberately a single constant: making it depend
on the leg would make the departure time depend on which objects were visited,
which is exactly the continuous-time scheduling problem the slot convention
exists to avoid.

``C[k, i, j]`` is then the cost of flying i->j using each object's mean
elements propagated to ``t_k``. Optionally (``window_days > 0``) it is the
MINIMUM over several departure times inside ``[t_k, t_k + window]``; that is
off by default, and ``docs/physics.md`` explains why: near 86 deg inclination
the differential nodal drift is ~0.016 deg/day (median pair), so the discount
is ~0.002 km/s per day of waiting, and taking the minimum grants it without the
objective ever paying for the wait.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
from sgp4.conveniences import jday_datetime

from dextrivia.core import DebrisObject
from dextrivia.costs.hohmann import hohmann_dv
from dextrivia.propagation import MU_WGS72_KM3_S2, R_EARTH_WGS72_KM, satrec

__all__ = [
    "J2_WGS72",
    "MeanElements",
    "mean_elements",
    "plane_angle",
    "impulsive_dv",
    "separate_burn_dv",
    "edelbaum_dv",
    "nodal_precession_rate_rad_s",
    "nodal_precession_deg_per_day",
    "ImpulsiveCostModel",
    "EdelbaumCostModel",
]

#: WGS72 second zonal harmonic -- the value SGP4 itself uses.
J2_WGS72 = 1.082616e-3

#: Grid resolution for the burn-split search in ``impulsive_dv``. The objective
#: is smooth but NOT convex in the split angle, so a coarse global scan comes
#: first and golden-section only refines inside the winning bracket.
_SPLIT_GRID = 181
_SPLIT_REFINEMENTS = 60


class MeanElements(tuple):
    """``(a_km, ecc, inc_rad, raan_rad)`` -- Brouwer mean elements at a time.

    A plain tuple subclass so it stays cheap to build in bulk while still being
    self-documenting at call sites.
    """

    __slots__ = ()

    def __new__(cls, a_km: float, ecc: float, inc_rad: float, raan_rad: float) -> MeanElements:
        return super().__new__(cls, (a_km, ecc, inc_rad, raan_rad))

    @property
    def a_km(self) -> float:
        return self[0]

    @property
    def ecc(self) -> float:
        return self[1]

    @property
    def inc_rad(self) -> float:
        return self[2]

    @property
    def raan_rad(self) -> float:
        return self[3]


def mean_elements(line1: str, line2: str, t: datetime) -> MeanElements:
    """Brouwer mean elements of a TLE at ``t``.

    SGP4 updates ``Satrec.am/em/im/Om`` on every propagation step, so these are
    the mean elements at ``t``, already including SGP4's own secular J2/J4 node
    regression and drag decay -- not the frozen TLE-epoch values. Using them
    keeps the RAAN drift between legs consistent with the propagator rather than
    with a hand-rolled approximation; ``nodal_precession_deg_per_day`` exists to
    explain and check that drift, and ``tests/test_realistic_costs.py`` asserts
    the two agree on real objects.

    Raises ``RuntimeError`` on SGP4 failure: a decayed object must not quietly
    contribute a plausible-looking cost.
    """
    if t.tzinfo is None:
        raise ValueError("propagation times must be timezone-aware UTC")
    sat = satrec(line1, line2)
    jd, fr = jday_datetime(t)
    error, _, _ = sat.sgp4(jd, fr)
    if error != 0:
        raise RuntimeError(f"SGP4 error {error} propagating {sat.satnum} to {t.isoformat()}")
    return MeanElements(
        float(sat.am) * R_EARTH_WGS72_KM, float(sat.em), float(sat.im), float(sat.Om)
    )


def plane_angle(inc1: float, raan1: float, inc2: float, raan2: float) -> float:
    """Angle (rad) between two orbit planes, from inclination and RAAN (rad).

    The planes' normals make the same angle as the planes, so::

        cos(theta) = cos(i1)cos(i2) + sin(i1)sin(i2)cos(raan1 - raan2)

    This is the TRUE plane angle. Using ``|i1 - i2|`` instead (a common
    shortcut) would report ~0.5 deg for this cloud and miss everything: the
    Iridium-33 debris differs by up to 173 deg of plane once RAAN is included.
    """
    cos_theta = np.cos(inc1) * np.cos(inc2) + np.sin(inc1) * np.sin(inc2) * np.cos(raan1 - raan2)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def _law_of_cosines(a: float, b: float, angle: np.ndarray | float) -> np.ndarray | float:
    """``sqrt(a^2 + b^2 - 2ab cos(angle))``, evaluated without cancellation."""
    return np.sqrt((a - b) ** 2 + 4.0 * a * b * np.sin(angle / 2.0) ** 2)


def _two_burn_dv(
    split: np.ndarray | float,
    theta: float,
    v1: float,
    v2: float,
    vt1: float,
    vt2: float,
) -> np.ndarray | float:
    """Total delta-v when ``split`` rad of the plane change is done at burn 1.

    Each burn rotates AND resizes the velocity vector, so its cost is the law of
    cosines between the pre- and post-burn velocities, written in the
    half-angle form ``(a-b)^2 + 4ab sin^2(s/2)`` rather than
    ``a^2 + b^2 - 2ab cos s``. The two are algebraically identical, but the
    second loses most of its significant digits when ``s`` is small and ``a`` is
    close to ``b`` -- which is the common case here (sub-degree plane angles
    inside a cluster), and it would show up as ~1e-9 relative noise on exactly
    the legs the sequencing decision turns on.
    """
    return _law_of_cosines(v1, vt1, split) + _law_of_cosines(v2, vt2, theta - split)


def impulsive_dv(r1: float, r2: float, theta: float, mu: float = MU_WGS72_KM3_S2) -> float:
    """Two-burn Hohmann transfer with the plane change folded into the burns.

    ``r1``, ``r2`` are circular orbit radii (km from Earth's centre) and
    ``theta`` the plane angle (rad). Returns km/s.

    Method
    ------
    A Hohmann transfer ellipse connects the two radii. Each burn is allowed to
    rotate the velocity vector as well as change its magnitude, so burn 1 does
    ``split`` of the plane change and burn 2 the remaining ``theta - split``::

        dv = |v1 -> vt1 through angle split| + |vt2 -> v2 through angle theta-split|
           = sqrt(v1^2 + vt1^2 - 2 v1 vt1 cos(split))
           + sqrt(v2^2 + vt2^2 - 2 v2 vt2 cos(theta - split))

    ``split`` is chosen to minimise the sum. Doing the rotation where the
    velocity is smallest is cheaper, which is why the optimum pushes most of the
    plane change onto the burn at the higher radius, but "most" is not "all":
    the exact share depends on the radius ratio, so it is solved numerically
    rather than assumed.

    The objective is smooth but not convex in ``split`` (the second derivative
    of each term changes sign near 90 deg), so this does a global scan on a
    fixed grid followed by golden-section refinement inside the winning bracket.
    That is ~1e-9 km/s from the true minimum and needs no scipy.

    Limitations
    -----------
    * Two burns only. For plane angles above ~60 deg a three-burn bi-elliptic
      manoeuvre (raise apoapsis, rotate cheaply, re-circularise) beats this, so
      the numbers here are an UPPER bound on what a real mission would pay for
      the worst pairs. ``separate_burn_dv`` is a looser upper bound still.
    * Burns are assumed to happen at the relative line of nodes; no phasing or
      argument-of-perigee penalty is charged for getting there.
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"orbital radii must be positive, got {r1}, {r2}")
    if theta < 0:
        raise ValueError(f"plane angle must be non-negative, got {theta}")
    if theta == 0.0:
        return hohmann_dv(r1, r2, mu)

    a_transfer = (r1 + r2) / 2.0
    v1 = float(np.sqrt(mu / r1))
    v2 = float(np.sqrt(mu / r2))
    vt1 = float(np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer)))
    vt2 = float(np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer)))

    grid = np.linspace(0.0, theta, _SPLIT_GRID)
    values = _two_burn_dv(grid, theta, v1, v2, vt1, vt2)
    best = int(np.argmin(values))
    low = grid[max(best - 1, 0)]
    high = grid[min(best + 1, _SPLIT_GRID - 1)]

    # Golden-section on the bracket. Fixed iteration count: the bracket is at
    # most 2*theta/180 wide, so 60 steps drive it far below float noise.
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = high - phi * (high - low), low + phi * (high - low)
    f1 = _two_burn_dv(x1, theta, v1, v2, vt1, vt2)
    f2 = _two_burn_dv(x2, theta, v1, v2, vt1, vt2)
    for _ in range(_SPLIT_REFINEMENTS):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - phi * (high - low)
            f1 = _two_burn_dv(x1, theta, v1, v2, vt1, vt2)
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + phi * (high - low)
            f2 = _two_burn_dv(x2, theta, v1, v2, vt1, vt2)
    return float(min(values[best], f1, f2))


def separate_burn_dv(r1: float, r2: float, theta: float, mu: float = MU_WGS72_KM3_S2) -> float:
    """Hohmann transfer PLUS a separate pure plane change. Upper bound.

    The plane change is charged at the higher of the two orbits, where the
    circular velocity is lowest and a rotation is therefore cheapest::

        dv = hohmann(r1, r2) + 2 * v_high * sin(theta / 2)

    This is the naive "add the two manoeuvres" model. It is never cheaper than
    ``impulsive_dv`` (combining a rotation with a magnitude change beats doing
    them separately, by the triangle inequality), which
    ``tests/test_realistic_costs.py`` asserts. Kept because it is the number
    most mission-design back-of-envelopes quote, so it makes the saving from
    combining burns visible instead of implicit.
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"orbital radii must be positive, got {r1}, {r2}")
    v_high = float(np.sqrt(mu / max(r1, r2)))
    return hohmann_dv(r1, r2, mu) + 2.0 * v_high * float(np.sin(theta / 2.0))


def edelbaum_dv(r1: float, r2: float, theta: float, mu: float = MU_WGS72_KM3_S2) -> float:
    """Edelbaum (1961) low-thrust circle-to-circle transfer with plane change.

    ::

        dv = sqrt(v1^2 + v2^2 - 2 v1 v2 cos(pi/2 * theta))

    with ``v = sqrt(mu / r)``. This is the closed-form optimum for a
    CONTINUOUS, constant-acceleration transfer between circular orbits when the
    thrust yaw angle is steered optimally -- the standard first-order estimate
    for an electric-propulsion servicer, and the reason low-thrust ADR studies
    quote plane changes that a chemical stage could never afford.

    Limiting cases (all asserted in the tests):

    * ``theta = 0``      -> ``|v2 - v1|``, the pure spiral.
    * ``r1 = r2 = r``    -> ``2 v sin(pi * theta / 4)``, a pure low-thrust plane
      change. For small angles that is ``pi/2`` times the impulsive
      ``2 v sin(theta/2)``.

    Low thrust never costs LESS delta-v than the impulsive model for the same
    geometry -- rotating the velocity vector a little at a time is strictly
    worse than rotating it all at once at the cheapest point of the orbit. Its
    advantage in real ADR studies is propellant MASS: an ~3000 s specific
    impulse buys back far more than the delta-v penalty costs. This package only
    models delta-v, so the Edelbaum model will always look like the loser here.
    That is a property of the metric, not of the propulsion.

    Limitations
    -----------
    * **Valid only up to ``theta ~ 114.6 deg`` (2 rad).** Above that the
      effective angle ``pi/2 * theta`` passes ``pi`` and the expression turns
      around and starts reporting CHEAPER transfers for bigger plane changes,
      which is nonsense. The effective angle is therefore clamped at ``pi``,
      giving ``v1 + v2`` -- stop dead and re-accelerate. That is an honest
      ceiling rather than a fake discount, but it is a ceiling: treat any
      Edelbaum number above 114.6 deg of plane as "too expensive to matter"
      rather than as a quantity.
    * No transfer TIME is returned. For constant acceleration ``a`` the time is
      ``dv / a``, which for a 1 mN/kg thruster turns a 1 km/s transfer into
      ~12 days; nothing in this package models that, and the per-leg budget
      ``delta_per_leg_days`` is an input, not a consequence.
    * Assumes many revolutions and negligible eccentricity throughout, no
      shadowing, no oblateness perturbation of the spiral.
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"orbital radii must be positive, got {r1}, {r2}")
    if theta < 0:
        raise ValueError(f"plane angle must be non-negative, got {theta}")
    v1 = float(np.sqrt(mu / r1))
    v2 = float(np.sqrt(mu / r2))
    return float(_law_of_cosines(v1, v2, min(np.pi / 2.0 * theta, np.pi)))


def nodal_precession_rate_rad_s(
    a_km: float,
    ecc: float,
    inc_rad: float,
    mu: float = MU_WGS72_KM3_S2,
    r_earth_km: float = R_EARTH_WGS72_KM,
    j2: float = J2_WGS72,
) -> float:
    """Secular J2 nodal precession ``dRAAN/dt`` in rad/s.

    ::

        dRAAN/dt = -3/2 * n * J2 * (Re / p)^2 * cos(i),   p = a (1 - e^2)

    Negative for prograde orbits (the node regresses westward), positive for
    retrograde ones -- which is how sun-synchronous orbits work, and what the
    ~98 deg / +0.9856 deg-per-day test case checks.

    First-order secular theory only: no J4, no drag coupling, no lunisolar
    terms. Good to ~1% against SGP4's own node regression for these orbits,
    which ``tests/test_realistic_costs.py`` verifies on real snapshot objects.
    """
    if a_km <= 0:
        raise ValueError(f"semi-major axis must be positive, got {a_km}")
    if not 0.0 <= ecc < 1.0:
        raise ValueError(f"eccentricity must be in [0, 1), got {ecc}")
    n = np.sqrt(mu / a_km**3)
    p = a_km * (1.0 - ecc * ecc)
    return float(-1.5 * n * j2 * (r_earth_km / p) ** 2 * np.cos(inc_rad))


def nodal_precession_deg_per_day(a_km: float, ecc: float, inc_rad: float, **kwargs: float) -> float:
    """``nodal_precession_rate_rad_s`` in the units mission designers quote."""
    return float(np.degrees(nodal_precession_rate_rad_s(a_km, ecc, inc_rad, **kwargs)) * 86400.0)


def _pair_matrix(
    elements: Sequence[MeanElements],
    dv_fn,
) -> np.ndarray:
    """(N, N) delta-v matrix from mean elements, using ``dv_fn(r1, r2, theta)``.

    Symmetric with a zero diagonal: both models here are direction-agnostic at
    a fixed time. Nothing downstream relies on that -- time-slotted costs are
    still asymmetric ACROSS slots, which is the whole point.
    """
    n = len(elements)
    matrix = np.zeros((n, n))
    for i in range(n):
        a_i, _, inc_i, raan_i = elements[i]
        for j in range(i + 1, n):
            a_j, _, inc_j, raan_j = elements[j]
            theta = plane_angle(inc_i, raan_i, inc_j, raan_j)
            matrix[i, j] = matrix[j, i] = dv_fn(a_i, a_j, theta)
    return matrix


class _PlaneAwareCostModel:
    """Shared machinery: propagate to each slot's departure time, build C.

    Subclasses supply ``base_name`` and ``dv_fn``. Instantiating with
    ``delta_per_leg_days=None`` gives static (N, N) costs at ``epoch``; setting
    it gives (N-1, N, N) with leg k departing at ``epoch + k * delta``.
    """

    base_name = "abstract"

    @staticmethod
    def dv_fn(r1: float, r2: float, theta: float) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def __init__(
        self,
        delta_per_leg_days: float | None = None,
        window_days: float = 0.0,
        window_samples: int = 1,
    ) -> None:
        if delta_per_leg_days is not None and delta_per_leg_days <= 0:
            raise ValueError(f"delta_per_leg_days must be positive, got {delta_per_leg_days}")
        if window_days < 0:
            raise ValueError(f"window_days must be non-negative, got {window_days}")
        if window_days > 0 and window_samples < 2:
            raise ValueError("a non-zero window needs at least 2 samples to be worth taking")
        if window_days > 0 and delta_per_leg_days is None:
            raise ValueError("window_days is meaningless for a static (time-independent) model")
        self.delta_per_leg_days = delta_per_leg_days
        self.window_days = window_days
        self.window_samples = window_samples if window_days > 0 else 1

    @property
    def name(self) -> str:
        if self.delta_per_leg_days is None:
            return f"{self.base_name}-static"
        tag = f"{self.base_name}-j2-{self.delta_per_leg_days:g}d"
        return f"{tag}-win{self.window_days:g}d" if self.window_days else tag

    @property
    def time_dependent(self) -> bool:
        return self.delta_per_leg_days is not None

    def departure_times(self, epoch: datetime, n_legs: int) -> list[list[datetime]]:
        """Sample times per leg slot. ``[k][m]`` is sample m of leg k's window."""
        delta = timedelta(days=float(self.delta_per_leg_days or 0.0))
        window = timedelta(days=self.window_days)
        out = []
        for k in range(n_legs):
            start = epoch + k * delta
            if self.window_samples == 1:
                out.append([start])
            else:
                fractions = np.linspace(0.0, 1.0, self.window_samples)
                out.append([start + f * window for f in fractions])
        return out

    def build(self, objects: Sequence[DebrisObject], epoch: datetime) -> np.ndarray:
        if not self.time_dependent:
            elements = [mean_elements(o.line1, o.line2, epoch) for o in objects]
            return _pair_matrix(elements, self.dv_fn)

        n_legs = len(objects) - 1
        slots = []
        for times in self.departure_times(epoch, n_legs):
            samples = [
                _pair_matrix([mean_elements(o.line1, o.line2, t) for o in objects], self.dv_fn)
                for t in times
            ]
            slots.append(np.min(np.stack(samples), axis=0))
        return np.stack(slots)

    def provenance(self) -> dict[str, object]:
        """What ``metadata`` must carry for a number built here to be reproducible."""
        return {
            "cost_model": self.name,
            "cost_model_family": self.base_name,
            "time_dependent": self.time_dependent,
            "delta_per_leg_days": self.delta_per_leg_days,
            "leg_departure_rule": (
                "leg k departs at epoch + k * delta_per_leg_days"
                if self.time_dependent
                else "all legs priced at epoch"
            ),
            "window_days": self.window_days,
            "window_samples": self.window_samples,
            "window_rule": (
                "C[k,i,j] = min over sampled departure times in [t_k, t_k + window_days]"
                if self.window_days
                else "C[k,i,j] priced at the nominal departure time t_k (no loiter)"
            ),
        }


class ImpulsiveCostModel(_PlaneAwareCostModel):
    """Hohmann + optimally split plane change. Chemical servicer. See ``impulsive_dv``."""

    base_name = "impulsive-plane"
    dv_fn = staticmethod(impulsive_dv)


class EdelbaumCostModel(_PlaneAwareCostModel):
    """Low-thrust Edelbaum transfer. Electric servicer. See ``edelbaum_dv``."""

    base_name = "edelbaum"
    dv_fn = staticmethod(edelbaum_dv)
