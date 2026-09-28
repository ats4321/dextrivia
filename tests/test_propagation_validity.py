"""Propagation horizon, decay handling, the v2 family, and CLI reproducibility.

Every time-dependent instance prices legs months from its TLE epochs. These
tests pin the measured horizon to the evidence that produced it, show that the
v1 over-horizon instances are flagged without being modified, and that the
decayed-object case that used to crash with "SGP4 error 6" now builds and
records its exclusions.
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from dextrivia.cli import main
from dextrivia.core import ProblemInstance
from dextrivia.costs import ClusterWindow, ImpulsiveCostModel
from dextrivia.costs.realistic import mean_elements
from dextrivia.costs.selection import build_cluster_instance
from dextrivia.costs.validity import (
    VALIDITY_HORIZON_DAYS,
    PropagationHorizonWarning,
    check_instance,
    compare_snapshots,
    horizon_from_comparison,
    screen_decay,
)
from dextrivia.instances import build_instance
from dextrivia.snapshots import Snapshot, default_snapshot_dir

FRESH_SNAPSHOT = "iridium33_20260928.json"
INSTANCES = default_snapshot_dir().parent / "instances"
V2_DIR = INSTANCES / "planecluster-v2"

#: Re-entered 2026-07-10 per Celestrak SATCAT; SGP4 only gives up on it in November.
DECAYED = 35080

#: v1 stays byte-identical: the README's canonical benchmark run cites these arrays.
V1_PREFIX = "iridium33_20260402_planecluster-v1_"
V1_SHA256 = {
    "n10_static": "6561ee0438a265ebadb834d43a38b1bcb609bd8d547a49fa5b120a84d082809f",
    "n10_td30d": "872a3d5d5c991c2390be596bb0dec58707e7876b00a0a55618988ea5486f4708",
    "n15_static": "b784f97ee11ec6ac65624416de3dce9a12866ae57e831601215bf84c65960808",
    "n15_td30d": "fa125de7bdd5cc8f29b5957366e95211165533181f3eb46dd8b3095f994bc991",
    "n20_static": "d5304a35e208a8d1a6c1550f81e11d327c290352967a2517f486c16f871e0790",
    "n20_td30d": "779884f29da7ad52d8be2c819b3106fbbaa66dedc35b2df55a2c4ded3c0e3d0d",
    "n4_static": "78b360f2b568e31a7a429da39538d542276832003b8d37a74b150b70122c89c8",
    "n4_td30d": "e542d234ff959af32871dd335240b4cd597083187d412fc9a4ce7e9cf206936b",
    "n5_static": "5e5bd5ed3be82f9f2ab487479e627968b08fb8b60ffdec265e53c4c73023f0df",
    "n5_td30d": "1bbbe52f030deeb13c9247f6161363ad947c728d3765e7000c9f9bd281e151aa",
    "n8_static": "faae8a50cf4c076dd42e2f528f7f506bdb684b89b24dd5b5e5f613cfadc5ac09",
    "n8_td30d": "3832a8659e4d86519ba2bcef8b49a10adce69cd6e62933a938f0de24ef23233f",
}


@pytest.fixture(scope="module")
def fresh() -> Snapshot:
    return Snapshot.load(default_snapshot_dir() / FRESH_SNAPSHOT)


# --- the horizon is the evidence, not a preference ----------------------------


def test_horizon_constant_is_what_the_two_snapshot_comparison_measures(snapshot, fresh):
    comparison = compare_snapshots(snapshot.objects, fresh.objects, fresh.median_epoch())
    measured = horizon_from_comparison(comparison)
    assert measured["horizon_days"] == VALIDITY_HORIZON_DAYS
    # The cap binds: the error at the one measured span is inside the tolerance.
    assert measured["error_at_span_kms"] < measured["tolerance_kms"]
    assert comparison["only_in_old"] == [DECAYED]


def test_compare_snapshots_reports_zero_error_against_itself(snapshot):
    objects = snapshot.objects[:6]
    comparison = compare_snapshots(objects, objects, snapshot.median_epoch())
    for row in comparison["objects"]:
        assert row["span_days"] == 0.0
        assert row["raan_err_deg"] == pytest.approx(0.0, abs=1e-9)
        assert row["sma_err_km"] == pytest.approx(0.0, abs=1e-9)
    assert all(p["dv_err_kms"] == pytest.approx(0.0, abs=1e-12) for p in comparison["pairs"])


# --- decay: excluded and recorded, never a crash ----------------------------------


def test_the_decayed_object_is_what_used_to_crash_the_cost_model(snapshot):
    obj = next(o for o in snapshot.objects if o.norad_id == DECAYED)
    late = snapshot.median_epoch() + timedelta(days=300)
    with pytest.raises(RuntimeError, match="SGP4 error 6"):
        mean_elements(obj.line1, obj.line2, late)

    kept, excluded = screen_decay([obj], [snapshot.median_epoch(), late])
    assert kept == ()
    assert excluded[0]["norad_id"] == DECAYED
    assert excluded[0]["sgp4_error"] == 6


def test_n30_td30d_builds_and_records_every_decayed_exclusion(snapshot):
    """The case that crashed: 28 legs x 30 days reaches 840 days past the epoch."""
    with pytest.warns(PropagationHorizonWarning, match="beyond the 177-day"):
        instance = build_cluster_instance(
            snapshot,
            30,
            ImpulsiveCostModel(delta_per_leg_days=30.0),
            window=ClusterWindow(raan_window_deg=20.0),
        )
    excluded = {r["norad_id"] for r in instance.metadata["excluded_decayed"]}
    assert DECAYED in excluded
    assert len(excluded) >= 5
    assert excluded.isdisjoint(instance.norad_ids)
    assert np.isfinite(instance.costs).all()
    assert instance.metadata["exceeds_validity_horizon"] is True


# --- the flag ------------------------------------------------------------------


def test_short_instances_are_within_the_horizon_and_silent(snapshot):
    with warnings.catch_warnings():
        warnings.simplefilter("error", PropagationHorizonWarning)
        instance = build_cluster_instance(snapshot, 5, ImpulsiveCostModel(delta_per_leg_days=7.0))
    assert instance.metadata["exceeds_validity_horizon"] is False
    assert instance.metadata["propagation_span_days"] < VALIDITY_HORIZON_DAYS
    assert instance.metadata["excluded_decayed"] == []


def test_a_far_caller_epoch_is_flagged_even_for_static_hohmann(snapshot):
    """The original bug: 455 days of propagation to a hardcoded 2025-01-01."""
    far = snapshot.median_epoch() - timedelta(days=455)
    with pytest.warns(PropagationHorizonWarning):
        instance = build_instance(snapshot, n=4, epoch=far)
    assert instance.metadata["exceeds_validity_horizon"] is True


def test_v1_is_untouched_and_its_over_horizon_instances_are_flagged():
    exceeds = set()
    for label, digest in V1_SHA256.items():
        path = INSTANCES / f"{V1_PREFIX}{label}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"{label} changed"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PropagationHorizonWarning)
            status = check_instance(ProblemInstance.load(path))
        if status["exceeds_validity_horizon"]:
            exceeds.add(label)
    assert exceeds == {"n8_td30d", "n10_td30d", "n15_td30d", "n20_td30d"}


# --- v2 ------------------------------------------------------------------------


def v2_family() -> dict[str, ProblemInstance]:
    return {p.name: ProblemInstance.load(p) for p in sorted(V2_DIR.glob("*.npz"))}


def test_v2_stays_inside_the_horizon_and_reaches_n40():
    family = v2_family()
    assert len(family) == 31
    sizes = {i.n for i in family.values()}
    assert {25, 30, 40} <= sizes
    for name, instance in family.items():
        meta = instance.metadata
        assert meta["snapshot"] == FRESH_SNAPSHOT, name
        assert meta["family_version"] == "v2", name
        assert meta["exceeds_validity_horizon"] is False, name
        assert meta["propagation_span_days"] <= VALIDITY_HORIZON_DAYS, name
        assert "excluded_decayed" in meta, name
    assert any(i.n == 40 and i.time_dependent for i in family.values())


def test_v2_is_not_picked_up_by_the_flat_bench_glob():
    assert not any("planecluster-v2" in p.name for p in INSTANCES.glob("*.npz"))


# --- every committed instance is reproducible from the CLI -----------------------


@pytest.mark.parametrize(
    ("committed", "argv"),
    [
        (
            INSTANCES / "iridium33_20260402_planecluster-v1_n8_td30d.npz",
            ["--snapshot", "iridium33_20260402.json", "--n", "8", "--delta-days", "30"],
        ),
        (
            V2_DIR / "iridium33_20260928_planecluster-v2_n25_td7d.npz",
            ["--snapshot", FRESH_SNAPSHOT, "--n", "25", "--delta-days", "7"]
            + ["--raan-window", "20", "--family-version", "v2"],
        ),
        (
            V2_DIR / "iridium33_20260928_planecluster-v2_n10_static.npz",
            ["--snapshot", FRESH_SNAPSHOT, "--n", "10", "--family-version", "v2"],
        ),
    ],
)
def test_cli_rebuilds_committed_instances(tmp_path, committed: Path, argv):
    argv = list(argv)
    argv[1] = str(default_snapshot_dir() / argv[1])
    out = tmp_path / "rebuilt.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PropagationHorizonWarning)
        code = main(
            ["build", "--select", "plane-cluster", "--cost-model", "impulsive-plane"]
            + argv
            + ["--out", str(out)]
        )
    assert code == 0
    rebuilt, original = ProblemInstance.load(out), ProblemInstance.load(committed)
    assert rebuilt.norad_ids == original.norad_ids
    # Same objects; costs to 1e-12 rather than bit-for-bit, because libm and the
    # sgp4 build differ in the last ulp between macOS (where the family was
    # built) and Linux CI -- same tolerance as tests/test_instance_family.py.
    np.testing.assert_allclose(rebuilt.costs, original.costs, rtol=1e-12)


def test_cli_builds_edelbaum_plane_cluster(tmp_path, capsys):
    out = tmp_path / "e.npz"
    snapshot = str(default_snapshot_dir() / FRESH_SNAPSHOT)
    argv = ["build", "--snapshot", snapshot, "--select", "plane-cluster", "--n", "6"]
    assert main([*argv, "--cost-model", "edelbaum", "--delta-days", "7", "--out", str(out)]) == 0
    assert "within the 177-day validity horizon" in capsys.readouterr().out
    instance = ProblemInstance.load(out)
    assert instance.costs.shape == (5, 6, 6)
    assert instance.metadata["cost_model"] == "edelbaum-j2-7d"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--delta-days", "7"], "hohmann is static"),
        (["--select", "first", "--cost-model", "impulsive-plane"], "hohmann costs only"),
    ],
)
def test_cli_rejects_combinations_that_would_mislabel_an_instance(tmp_path, capsys, extra, message):
    snapshot = str(default_snapshot_dir() / FRESH_SNAPSHOT)
    argv = ["build", "--snapshot", snapshot, "--n", "4", "--out", str(tmp_path / "x.npz")]
    assert main([*argv, *extra]) == 2
    assert message in capsys.readouterr().err
