"""The v3 ``collision-pair`` family: pre-registered, pinned, reproducible.

v3 was built and committed, with its diagnostics, before any solver ran on it.
The sha256 pins make that auditable: if a file here changes, the benchmark
numbers citing it no longer describe it.
"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pytest

from dextrivia.cli import main
from dextrivia.core import ProblemInstance
from dextrivia.costs.selection import PAIR_HORIZON_NOTE
from dextrivia.costs.validity import VALIDITY_HORIZON_DAYS, PropagationHorizonWarning
from dextrivia.snapshots import default_snapshot_dir

INSTANCES = default_snapshot_dir().parent / "instances"
V3_DIR = INSTANCES / "collisionpair-v3"
IRIDIUM, COSMOS = "iridium33_20260928.json", "cosmos2251_20260930.json"
DIAGNOSTICS = default_snapshot_dir().parent.parent / "docs" / "data" / "instance_diagnostics.json"

V3_SHA256 = {
    "n10_static": "2ed6cdd33e842529d1a2de956df763d427cc88502293fb6db9349ab5da84d922",
    "n10_td3d": "186a5b932c084a6725814e0936a3a219c71ff8293791313e5e47ab58f0f85b82",
    "n10_td7d": "4117204b10131d5528eb71e646ae8720728c25a3b8a54b039e983c9ca0423495",
    "n20_static": "568db9b51ac27739a1b3a15bf7ee971955182194347b19f959523def838e5dc0",
    "n20_td3d": "fd61d18fe988b1b6c68751f3af0a5aa7aecd993a96dc78eba047957b9d0c8c7e",
    "n20_td7d": "5054c16dfb0eb07c630fc5809b3c2df910a5b77b25bb3577003780ac4f4c8ff0",
    "n30_static": "f5951e5dd4fd5c0a8a886bc6083f420d864463cf89f936cb20bc520ed451fcb3",
    "n30_td3d": "fdcd421070b052465e568104bd2aa69eb2bf55c59db51eb27b8eb8d7934b149b",
    "n40_static": "8ebce1de93e5bc6fbd21674294b963483624fd4f40b725e840341734ef1f59e2",
    "n40_td3d": "8dbff4485c648893ae6b0ac45cfabc203cbe19f932b59bbb73fa74565e554263",
    "n50_static": "1329abf7ed5ee343eb8a21052f4565818afcd02b6d2649925dfed2e07b27cd97",
    "n50_td3d": "1550642e5e59d2f2dbae3fc4ff7f01503cd9ca7fb70c59188df1b1a80a45d1ce",
    "n60_static": "41e8449a3865f1649817e41bda55b22540caa11ceef5bb21d2ae2c3a6965eb5d",
}


def label(path) -> str:
    return path.stem[path.stem.rfind("_n") + 1 :]


def v3_family() -> dict[str, ProblemInstance]:
    return {label(p): ProblemInstance.load(p) for p in sorted(V3_DIR.glob("*.npz"))}


def test_v3_files_are_exactly_the_pre_registered_ones():
    hashes = {label(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in V3_DIR.glob("*.npz")}
    assert hashes == V3_SHA256


def test_v3_is_half_from_each_cloud_and_says_its_horizon_is_borrowed():
    for name, instance in v3_family().items():
        meta = instance.metadata
        assert meta["selection_rule"] == "collision-pair", name
        assert meta["snapshots"] == [IRIDIUM, COSMOS], name
        counts = meta["cloud_counts"]
        assert counts[IRIDIUM] == instance.n - instance.n // 2, name
        assert counts[COSMOS] == instance.n // 2, name
        assert meta["validity_horizon_note"] == PAIR_HORIZON_NOTE, name
        assert meta["exceeds_validity_horizon"] is False, name
        assert meta["propagation_span_days"] <= VALIDITY_HORIZON_DAYS, name


def test_v3_diagnostics_were_committed_for_every_file():
    diag = json.loads(DIAGNOSTICS.read_text())["v3"]
    assert {r["label"] for r in diag["instances"]} == set(V3_SHA256)
    assert diag["threshold_pct"] == 1.0


def test_v3_is_not_picked_up_by_the_flat_bench_glob():
    assert not any("collisionpair" in p.name for p in INSTANCES.glob("*.npz"))


@pytest.mark.parametrize(("n", "delta"), [(10, None), (20, 3.0)])
def test_cli_rebuilds_v3(tmp_path, n, delta):
    out = tmp_path / "rebuilt.npz"
    argv = [
        "build",
        "--snapshot",
        str(default_snapshot_dir() / IRIDIUM),
        "--pair-snapshot",
        str(default_snapshot_dir() / COSMOS),
        "--select",
        "collision-pair",
        "--cost-model",
        "impulsive-plane",
        "--n",
        str(n),
        "--family-version",
        "v3",
        "--out",
        str(out),
    ]
    if delta is not None:
        argv += ["--delta-days", f"{delta:g}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PropagationHorizonWarning)
        assert main(argv) == 0
    variant = "static" if delta is None else f"td{delta:g}d"
    original = v3_family()[f"n{n}_{variant}"]
    rebuilt = ProblemInstance.load(out)
    assert rebuilt.norad_ids == original.norad_ids
    np.testing.assert_allclose(rebuilt.costs, original.costs, rtol=1e-12, atol=1e-9)
