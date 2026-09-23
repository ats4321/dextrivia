"""CLI smoke test: build an instance, then solve it with both solvers."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dextrivia.cli import latest_snapshot, main
from dextrivia.snapshots import default_snapshot_dir

from .conftest import BENCHMARK_SNAPSHOT


def total_dv(captured: str) -> float:
    match = re.search(r"total dv\s+([0-9.]+) km/s", captured)
    assert match, captured
    return float(match.group(1))


def test_build_then_solve_greedy_equals_exact(tmp_path, capsys):
    snapshot = str(default_snapshot_dir() / BENCHMARK_SNAPSHOT)
    instance = tmp_path / "instance.npz"

    assert main(["build", "--snapshot", snapshot, "--n", "10", "--out", str(instance)]) == 0
    capsys.readouterr()

    totals = {}
    for solver in ("greedy", "exact"):
        assert main(["solve", "--instance", str(instance), "--solver", solver]) == 0
        totals[solver] = total_dv(capsys.readouterr().out)

    assert totals["greedy"] == totals["exact"]


def test_random_selection_requires_a_seed(tmp_path, capsys):
    snapshot = str(default_snapshot_dir() / BENCHMARK_SNAPSHOT)
    code = main(
        [
            "build",
            "--snapshot",
            snapshot,
            "--n",
            "5",
            "--select",
            "random",
            "--out",
            str(tmp_path / "i.npz"),
        ]
    )
    assert code == 2
    assert "requires a seed" in capsys.readouterr().err


def make_snapshots(directory, *names):
    for name in names:
        (directory / name).write_text("{}", encoding="utf-8")
    return directory


def test_latest_snapshot_orders_by_date_not_alphabetically(tmp_path):
    """The newer snapshot here sorts FIRST alphabetically, so name order lies."""
    make_snapshots(tmp_path, "cosmos2251_20260405.json", "iridium33_20260402.json")

    assert latest_snapshot(None, tmp_path).name == "cosmos2251_20260405.json"
    assert latest_snapshot("cosmos-2251-debris", tmp_path).name == "cosmos2251_20260405.json"
    # The CLI default is the newest of the DEFAULT group, never the newest of
    # any group: switching datasets has to be an explicit choice.
    assert latest_snapshot(snapshot_dir=tmp_path).name == "iridium33_20260402.json"


def test_latest_snapshot_prefers_the_same_day_suffix_written_second(tmp_path):
    make_snapshots(tmp_path, "iridium33_20260402.json", "iridium33_20260402T181124Z.json")
    assert latest_snapshot(snapshot_dir=tmp_path).name == "iridium33_20260402T181124Z.json"


def test_latest_snapshot_ignores_unparseable_names_and_reports_an_empty_group(tmp_path):
    make_snapshots(tmp_path, "notasnapshot.json", "iridium33_20260402.json")
    assert latest_snapshot(snapshot_dir=tmp_path).name == "iridium33_20260402.json"
    with pytest.raises(SystemExit, match="cosmos-2251-debris"):
        latest_snapshot("cosmos-2251-debris", tmp_path)


def test_build_prints_the_path_it_actually_wrote(tmp_path, capsys):
    """--out without a .npz suffix still has to name a file that exists."""
    snapshot = str(default_snapshot_dir() / BENCHMARK_SNAPSHOT)
    out = tmp_path / "instance"
    assert main(["build", "--snapshot", snapshot, "--n", "4", "--out", str(out)]) == 0

    written = re.search(r"wrote (\S+)", capsys.readouterr().out)
    assert written, "build did not report a path"
    assert Path(written.group(1)).exists()
