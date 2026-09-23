"""Picking a physically meaningful set of targets.

Why this is not just ``Snapshot.select(n)``
-------------------------------------------
``first`` and ``random`` take objects in catalogue order or at random. With a
plane-aware cost model that produces nonsense missions: the Iridium-33 cloud
spans the full 360 deg of RAAN, the median pairwise plane angle is 45.5 deg,
and a single median leg costs ~5.8 km/s -- more than a launch to GEO, per leg.
No servicer flies that. Real active-debris-removal studies pick targets that
already share a plane and differ mostly in altitude.

So this module adds one more explicit, recorded selection rule:

``plane-cluster``
    1. Keep the objects within ``raan_window_deg`` of RAAN, ``inc_window_deg``
       of inclination and ``alt_band_km`` of mean altitude from a SEED object.
    2. Seed = the object with the most neighbours surviving that filter
       ("the densest plane"), ties broken by the lowest NORAD id. It can also be
       pinned explicitly with ``seed_norad``.
    3. Take the ``n`` survivors closest to the seed in TRUE plane angle
       (NORAD id breaks ties), and order the instance by NORAD id.

The rule is fixed before any solver runs and does not look at delta-v, let
alone at which solver wins: choosing instances by "where greedy loses" would
cook the benchmark this repository exists to produce.

Note on the inclination window: this cloud's inclinations span 0.51 deg in
total, so ``inc_window_deg`` is effectively inert here. It is kept because the
same rule applied to a mixed catalogue (Cosmos-2251 debris, say) needs it, and
because a window that silently does nothing should say so rather than be
absent.

Instances are built here rather than through ``dextrivia.instances.build_instance``
because that function's ``rule`` argument is ``first``/``random`` only --
foundation-owned. Nothing in ``core.py`` changes; this just populates the same
``ProblemInstance`` with a superset of the same metadata keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dextrivia.core import CostModel, DebrisObject, ProblemInstance
from dextrivia.costs.realistic import mean_elements, plane_angle
from dextrivia.propagation import R_EARTH_WGS72_KM
from dextrivia.snapshots import Snapshot

__all__ = [
    "PLANE_CLUSTER_RULE",
    "ClusterWindow",
    "ClusterSelection",
    "select_plane_cluster",
    "build_cluster_instance",
]

PLANE_CLUSTER_RULE = "plane-cluster"


@dataclass(frozen=True)
class ClusterWindow:
    """How close two objects must be to count as "the same plane".

    Defaults are tuned to the Iridium-33 cloud: a 10 deg plane-angle-scale RAAN
    window is about the widest that still leaves plane changes comparable to
    altitude changes (1 deg of plane ~ 0.13 km/s, 100 km of altitude ~ 0.05
    km/s), and a 250 km altitude band covers most of the cloud's 513-892 km
    spread without pulling in objects whose nodal drift rates differ much.
    """

    raan_window_deg: float = 10.0
    inc_window_deg: float = 2.0
    alt_band_km: float = 250.0

    def as_metadata(self) -> dict[str, float]:
        return {
            "raan_window_deg": self.raan_window_deg,
            "inc_window_deg": self.inc_window_deg,
            "alt_band_km": self.alt_band_km,
        }


@dataclass(frozen=True)
class ClusterSelection:
    """The objects a ``plane-cluster`` selection picked, plus how tight it is."""

    objects: tuple[DebrisObject, ...]
    seed_norad: int
    epoch: datetime
    window: ClusterWindow
    max_plane_angle_deg: float
    altitude_span_km: float

    def as_metadata(self) -> dict[str, object]:
        return {
            "selection_rule": PLANE_CLUSTER_RULE,
            "selection_n": len(self.objects),
            "selection_seed": None,  # no RNG: the rule is deterministic
            "cluster_seed_norad": self.seed_norad,
            "cluster_max_plane_angle_deg": round(self.max_plane_angle_deg, 4),
            "cluster_altitude_span_km": round(self.altitude_span_km, 3),
            **self.window.as_metadata(),
        }


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two angles in degrees (wraps at 360)."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def select_plane_cluster(
    snapshot: Snapshot,
    n: int,
    epoch: datetime | None = None,
    window: ClusterWindow | None = None,
    seed_norad: int | None = None,
) -> ClusterSelection:
    """Pick ``n`` objects sharing a plane. See the module docstring for the rule.

    Raises ``ValueError`` if no seed has ``n`` neighbours inside the window --
    silently widening the window would make the recorded rule a lie.
    """
    window = window or ClusterWindow()
    epoch = epoch or snapshot.median_epoch()
    objects = snapshot.objects
    if not 2 <= n <= len(objects):
        raise ValueError(f"n must be in 2..{len(objects)}, got {n}")

    elements = [mean_elements(o.line1, o.line2, epoch) for o in objects]
    norad = np.array([o.norad_id for o in objects])
    alt = np.array([e.a_km for e in elements]) - R_EARTH_WGS72_KM
    inc_deg = np.degrees([e.inc_rad for e in elements])
    raan_deg = np.degrees([e.raan_rad for e in elements]) % 360.0

    size = len(objects)
    theta_deg = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            theta = np.degrees(
                plane_angle(
                    elements[i].inc_rad,
                    elements[i].raan_rad,
                    elements[j].inc_rad,
                    elements[j].raan_rad,
                )
            )
            theta_deg[i, j] = theta_deg[j, i] = theta

    inside = (
        (_angle_diff_deg(raan_deg[:, None], raan_deg[None, :]) <= window.raan_window_deg)
        & (np.abs(inc_deg[:, None] - inc_deg[None, :]) <= window.inc_window_deg)
        & (np.abs(alt[:, None] - alt[None, :]) <= window.alt_band_km)
    )

    if seed_norad is not None:
        matches = np.flatnonzero(norad == seed_norad)
        if matches.size == 0:
            raise ValueError(f"seed NORAD id {seed_norad} is not in {snapshot.path.name}")
        seed = int(matches[0])
    else:
        counts = inside.sum(axis=1)
        # densest plane first, lowest NORAD id as the tie-break
        seed = int(np.lexsort((norad, -counts))[0])

    candidates = np.flatnonzero(inside[seed])
    if candidates.size < n:
        raise ValueError(
            f"seed NORAD {norad[seed]} has only {candidates.size} objects inside "
            f"{window} (need {n}); widen the window explicitly or ask for fewer objects"
        )
    ranked = candidates[np.lexsort((norad[candidates], theta_deg[seed][candidates]))][:n]
    chosen = np.sort(ranked)

    return ClusterSelection(
        objects=tuple(objects[i] for i in chosen),
        seed_norad=int(norad[seed]),
        epoch=epoch,
        window=window,
        max_plane_angle_deg=float(theta_deg[np.ix_(chosen, chosen)].max()),
        altitude_span_km=float(alt[chosen].max() - alt[chosen].min()),
    )


def build_cluster_instance(
    snapshot: Snapshot,
    n: int,
    cost_model: CostModel,
    epoch: datetime | None = None,
    window: ClusterWindow | None = None,
    seed_norad: int | None = None,
    family_version: str | None = None,
) -> ProblemInstance:
    """``plane-cluster`` selection + a cost model -> a fully provenanced instance.

    Metadata carries everything needed to rebuild the exact same array: snapshot
    filename, selection rule and window, epoch source, cost model name, and (for
    time-slotted costs) the per-leg duration and the leg-to-wall-clock mapping.
    """
    selection = select_plane_cluster(snapshot, n, epoch=epoch, window=window, seed_norad=seed_norad)
    costs = cost_model.build(selection.objects, selection.epoch)
    provenance = (
        cost_model.provenance()
        if hasattr(cost_model, "provenance")
        else {"cost_model": cost_model.name}
    )
    metadata: dict[str, object] = {
        "snapshot": snapshot.path.name,
        "snapshot_fetched_utc": (
            snapshot.fetched_utc.isoformat() if snapshot.fetched_utc else None
        ),
        "snapshot_size": len(snapshot),
        "epoch_source": "snapshot-median-tle-epoch" if epoch is None else "caller",
        **selection.as_metadata(),
        **provenance,
    }
    if family_version is not None:
        metadata["family_version"] = family_version
    return ProblemInstance(
        norad_ids=tuple(o.norad_id for o in selection.objects),
        names=tuple(o.name for o in selection.objects),
        epoch=selection.epoch,
        costs=costs,
        metadata=metadata,
    )
