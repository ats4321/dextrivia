from __future__ import annotations

import pytest

from dextrivia.instances import build_instance
from dextrivia.snapshots import Snapshot, default_snapshot_dir

#: The snapshot the committed benchmarks are defined against.
BENCHMARK_SNAPSHOT = "iridium33_20260402.json"


@pytest.fixture(scope="session")
def snapshot() -> Snapshot:
    return Snapshot.load(default_snapshot_dir() / BENCHMARK_SNAPSHOT)


@pytest.fixture(scope="session")
def benchmark_instance(snapshot: Snapshot):
    """The canonical 10-object instance: first 10 objects, snapshot median epoch."""
    return build_instance(snapshot, n=10, rule="first")
