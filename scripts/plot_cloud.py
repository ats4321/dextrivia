"""Plot the Iridium-33 cloud in the plane that actually matters: RAAN vs altitude.

    uv run python scripts/plot_cloud.py            # -> docs/figures/raan_altitude.png

Altitude alone (the x axis of the old cost model) makes the cloud look like a
tidy 380 km-wide band. Adding RAAN shows what a servicer really faces: the
objects are smeared across the full 360 deg of node, and moving sideways on this
plot is roughly 0.13 km/s per degree, against 0.05 km/s per 100 km vertically.
The highlighted clusters are what ``plane-cluster`` selection picks -- the only
regions where a multi-target mission is affordable at all.

Needs the ``physics`` extra (matplotlib): ``uv sync --all-extras``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dextrivia.costs.realistic import mean_elements, nodal_precession_deg_per_day
from dextrivia.costs.selection import select_plane_cluster
from dextrivia.propagation import R_EARTH_WGS72_KM
from dextrivia.snapshots import Snapshot, default_snapshot_dir

DEFAULT_SNAPSHOT = "iridium33_20260402.json"
HIGHLIGHTED_SIZES = (5, 10, 20)


def figure_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "figures"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")  # scripted, headless, deterministic
    import matplotlib.pyplot as plt

    path = Path(args.snapshot)
    if not path.exists():
        path = default_snapshot_dir() / args.snapshot
    snapshot = Snapshot.load(path)
    epoch = snapshot.median_epoch()

    elements = [mean_elements(o.line1, o.line2, epoch) for o in snapshot.objects]
    altitude = np.array([e.a_km for e in elements]) - R_EARTH_WGS72_KM
    raan = np.degrees([e.raan_rad for e in elements]) % 360.0
    drift = np.array([nodal_precession_deg_per_day(e.a_km, e.ecc, e.inc_rad) for e in elements])

    fig, (ax, ax_drift) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    ax.scatter(raan, altitude, s=26, c="0.72", edgecolors="0.45", linewidths=0.5, label="cloud")
    # Clusters are nested (same seed), so draw the widest first and let the
    # tighter ones sit inside it rather than under it.
    colours = ("#2ca02c", "#1f77b4", "#d62728")
    marker_sizes = (190, 100, 40)
    for size, colour, marker in zip(
        sorted(HIGHLIGHTED_SIZES, reverse=True), colours, marker_sizes, strict=True
    ):
        selection = select_plane_cluster(snapshot, size, epoch=epoch)
        chosen = {o.norad_id for o in selection.objects}
        mask = np.array([o.norad_id in chosen for o in snapshot.objects])
        ax.scatter(
            raan[mask],
            altitude[mask],
            s=marker,
            facecolors="none",
            edgecolors=colour,
            linewidths=1.8,
            label=(f"plane-cluster N={size} (<={selection.max_plane_angle_deg:.1f} deg plane)"),
        )
    ax.set_xlabel("RAAN at epoch (deg)")
    ax.set_ylabel("mean altitude (km)")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 60))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"{len(snapshot)} Iridium-33 debris objects at {epoch:%Y-%m-%d}\n"
        "inclination 85.96-86.47 deg: RAAN, not altitude, is the expensive axis",
        fontsize=11,
    )

    ax_drift.scatter(drift, altitude, s=22, c="0.55")
    ax_drift.set_xlabel("J2 nodal drift dRAAN/dt (deg/day)")
    ax_drift.set_ylabel("mean altitude (km)")
    ax_drift.grid(alpha=0.25)
    ax_drift.set_title(
        "Drift is common-mode\n"
        f"spread {drift.max() - drift.min():.3f} deg/day "
        f"= {10 / (drift.max() - drift.min()):.0f} days per 10 deg",
        fontsize=10,
    )

    fig.tight_layout()
    out = args.out or (figure_dir() / "raan_altitude.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"  altitude {altitude.min():.0f}-{altitude.max():.0f} km")
    print(f"  RAAN     {raan.min():.1f}-{raan.max():.1f} deg")
    print(f"  drift    {drift.min():.3f} to {drift.max():.3f} deg/day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
