"""Check SGP4 against reality: propagate an old snapshot, compare with a newer one.

    uv run python scripts/validate_propagation.py                 # report + JSON + figure
    uv run python scripts/validate_propagation.py --no-figure     # without matplotlib

For every object present in both snapshots (matched by NORAD id) the OLD TLE is
propagated to the NEW TLE's epoch and its mean RAAN, inclination and semi-major
axis are compared with the new TLE's own mean elements. For every pair within
20 deg of plane the predicted and actual impulsive leg cost are compared too --
that is the number the cost model uses, and common-mode RAAN error cancels in it.

THIS IS A SINGLE-HORIZON MEASUREMENT (~177 days), NOT AN ERROR-VS-TIME CURVE.
Two snapshots give one propagation span per object (164-197 days here). Per-day
rates quoted below assume linear growth, which is reasonable for the secular
RAAN/inclination rate errors and optimistic for drag-driven semi-major-axis
error. The TLE fit-noise floor is unmeasured: part of each error may be the
new TLE's own fit noise rather than propagation error.

Objects in only one snapshot are cross-checked against the committed Celestrak
SATCAT records in ``data/validation/`` to separate re-entries from catalogue
churn, and SGP4's own predicted decay date is found by daily stepping.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
from sgp4.conveniences import jday_datetime

from dextrivia.costs.validity import compare_snapshots, horizon_from_comparison
from dextrivia.propagation import satrec
from dextrivia.snapshots import Snapshot, default_snapshot_dir

OLD = "iridium33_20260402.json"
NEW = "iridium33_20260928.json"
SATCAT = "satcat_20260928.json"
FIGURE = Path(__file__).resolve().parents[1] / "docs" / "figures" / "propagation_validation.png"

BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"


def sgp4_decay_day(obj, start, max_days: int = 2000) -> int | None:
    """First whole day after ``start`` at which SGP4 refuses the object, or None."""
    sat = satrec(obj.line1, obj.line2)
    for day in range(max_days):
        error, _, _ = sat.sgp4(*jday_datetime(start + timedelta(days=day)))
        if error:
            return day
    return None


def summarise(comparison: dict, old: Snapshot, satcat: dict) -> dict:
    objects = comparison["objects"]
    span = np.array([o["span_days"] for o in objects])
    median_span = float(np.median(span))
    out: dict = {
        "old_snapshot": old.path.name,
        "measurement": "single horizon, two snapshots; not an error-vs-time curve",
        "objects_compared": len(objects),
        "span_days": {"min": span.min(), "median": median_span, "max": span.max()},
        "elements": {},
    }
    for key, unit in (("raan_err_deg", "deg"), ("inc_err_deg", "deg"), ("sma_err_km", "km")):
        err = np.array([o[key] for o in objects])
        mag = np.abs(err)
        out["elements"][key] = {
            "unit": unit,
            "mean_signed": float(err.mean()),
            "median_abs": float(np.median(mag)),
            "p90_abs": float(np.percentile(mag, 90)),
            "max_abs": float(mag.max()),
            "worst_norad": int(objects[int(mag.argmax())]["norad_id"]),
            "median_abs_per_day_if_linear": float(np.median(mag) / median_span),
            "p90_abs_per_30d_if_linear": float(np.percentile(mag, 90) / median_span * 30),
        }
    pairs = comparison["pairs"]
    angle = np.abs([p["angle_err_deg"] for p in pairs])
    dv = np.abs([p["dv_err_kms"] for p in pairs])
    out["pairs"] = {
        "count": len(pairs),
        "cluster_angle_deg": 20.0,
        "median_leg_kms": float(np.median([p["actual_dv_kms"] for p in pairs])),
        "angle_err_deg": {
            "median": float(np.median(angle)),
            "p90": float(np.percentile(angle, 90)),
        },
        "dv_err_kms": {
            "median": float(np.median(dv)),
            "p90": float(np.percentile(dv, 90)),
            "max": float(dv.max()),
        },
    }
    out["horizon"] = horizon_from_comparison(comparison)

    records = {r["NORAD_CAT_ID"]: r for r in satcat["records"]}
    by_id = {o.norad_id: o for o in old.objects}
    out["only_in_old"] = []
    for norad in comparison["only_in_old"]:
        record = records.get(norad, {})
        predicted = sgp4_decay_day(by_id[norad], old.median_epoch())
        out["only_in_old"].append(
            {
                "norad_id": norad,
                "satcat_decay_date": record.get("DECAY_DATE") or None,
                "sgp4_decay_day_after_old_epoch": predicted,
                "sgp4_decay_date": (
                    (old.median_epoch() + timedelta(days=predicted)).date().isoformat()
                    if predicted is not None
                    else None
                ),
            }
        )
    out["only_in_new"] = [
        {"norad_id": n, "satcat_decay_date": records.get(n, {}).get("DECAY_DATE") or None}
        for n in comparison["only_in_new"]
    ]
    out["sgp4_failed"] = comparison["sgp4_failed"]
    return out


def plot(comparison: dict, summary: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objects = comparison["objects"]
    alt = [o["actual_altitude_km"] for o in objects]
    span = summary["span_days"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    panels = (
        ("raan_err_deg", "RAAN error (deg)"),
        ("inc_err_deg", "Inclination error (deg)"),
        ("sma_err_km", "Mean semi-major-axis error (km)"),
    )
    for ax, (key, label) in zip(axes.flat, panels, strict=False):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.scatter(
            alt,
            [o[key] for o in objects],
            s=22,
            color=BLUE,
            edgecolors=SURFACE,
            linewidths=0.8,
        )
        stats = summary["elements"][key]
        ax.set_title(
            f"{label}\nmedian |err| {stats['median_abs']:.3g}, p90 {stats['p90_abs']:.3g}",
            fontsize=10,
            color=INK,
            loc="left",
        )
        ax.set_xlabel("Actual mean altitude (km)", color=MUTED)
        ax.set_ylabel("Predicted - actual", color=MUTED)

    ax = axes.flat[3]
    ax.set_facecolor(SURFACE)
    dv = np.abs([p["dv_err_kms"] for p in comparison["pairs"]]) * 1e3
    ax.hist(dv, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    horizon = summary["horizon"]
    for value, text, style, y in (
        (horizon["error_at_span_kms"] * 1e3, "p90", ":", 0.92),
        (horizon["tolerance_kms"] * 1e3, "tolerance (5% of median leg)", "--", 0.80),
    ):
        ax.axvline(value, color=INK, ls=style, lw=1.2)
        ax.annotate(
            f" {text}: {value:.0f} m/s",
            (value, y),
            xycoords=("data", "axes fraction"),
            fontsize=9,
            color=INK,
        )
    pairs = summary["pairs"]
    ax.set_title(
        f"|Leg-cost error|, {pairs['count']} pairs within 20 deg of plane\n"
        f"median leg {pairs['median_leg_kms']:.2f} km/s",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.set_xlabel("|Predicted - actual impulsive delta-v| (m/s)", color=MUTED)
    ax.set_ylabel("Pairs", color=MUTED)

    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(True, color="0.9", lw=0.6)
        a.set_axisbelow(True)
    fig.suptitle(
        f"SGP4 from {summary['old_snapshot']} vs the real later TLEs "
        f"({summary['objects_compared']} objects)\n"
        f"ONE horizon: {span['min']:.0f}-{span['max']:.0f} days, median {span['median']:.0f}. "
        "Not an error-vs-time curve; TLE fit noise is not separated from propagation error.",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", default=OLD)
    parser.add_argument("--new", default=NEW)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    snapshots = default_snapshot_dir()
    old, new = Snapshot.load(snapshots / args.old), Snapshot.load(snapshots / args.new)
    validation_dir = snapshots.parent / "validation"
    satcat = json.loads((validation_dir / SATCAT).read_text(encoding="utf-8"))

    comparison = compare_snapshots(old.objects, new.objects, new.median_epoch())
    summary = summarise(comparison, old, satcat)
    summary["new_snapshot"] = new.path.name

    out = validation_dir / f"propagation_{Path(args.old).stem}_vs_{Path(args.new).stem}.json"
    out.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print(f"wrote {out}")
    if not args.no_figure:
        plot(comparison, summary, FIGURE)
        print(f"wrote {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
