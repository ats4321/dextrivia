"""The harness additions: resumable runs, bound-aware references, seed policy."""

from __future__ import annotations

import csv
import json

import pytest

from dextrivia import bench
from dextrivia.bench import (
    CHECKPOINT,
    add_references,
    load_checkpoint,
    run_bench,
    seeds_for,
)
from tests.test_qubo_formulation import random_instance


@pytest.fixture
def instance_dir(tmp_path):
    out = tmp_path / "instances"
    out.mkdir()
    random_instance(4, 1).save(out / "snap_planecluster-v9_n4_static.npz")
    random_instance(4, 2, time_dependent=True).save(out / "snap_planecluster-v9_n4_td30d.npz")
    return out


def _run(instance_dir, out, **kwargs):
    return run_bench(
        instance_dir=instance_dir,
        out_dir=out,
        solvers=kwargs.pop("solvers", ["greedy", "localsearch", "sa-perm"]),
        seeds=(1, 2),
        run_penalty_study=False,
        verbose=False,
        **kwargs,
    )


def test_a_resumed_run_skips_completed_rows_and_finishes_the_rest(
    instance_dir, tmp_path, monkeypatch
):
    out = tmp_path / "run"
    calls = []
    real = bench.run_one

    def counting(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 4:
            raise KeyboardInterrupt  # killed mid-run
        return real(*args, **kwargs)

    monkeypatch.setattr(bench, "run_one", counting)
    with pytest.raises(KeyboardInterrupt):
        _run(instance_dir, out)
    assert len(load_checkpoint(out)) == 3  # the three that finished were kept

    calls.clear()
    monkeypatch.setattr(bench, "run_one", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    _run(instance_dir, out)
    # 2 instances x (greedy once + localsearch x2 + sa-perm x2) = 10 runs, 3 done before
    assert len(calls) == 7
    rows = list(csv.DictReader((out / "results.csv").open()))
    assert len(rows) == 10
    assert len({(r["instance"], r["solver"], r["seed"]) for r in rows}) == 10

    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["segments"]) == 2
    assert sum(s["runs"] for s in manifest["segments"]) == 10


def test_a_complete_run_resumed_runs_nothing(instance_dir, tmp_path, monkeypatch):
    out = _run(instance_dir, tmp_path / "run")
    monkeypatch.setattr(bench, "run_one", lambda *a, **k: pytest.fail("re-ran a finished row"))
    _run(instance_dir, out)


def test_a_torn_checkpoint_line_is_dropped_not_fatal(tmp_path):
    (tmp_path / CHECKPOINT).write_text('{"instance_file": "a", "solver": "g", "seed": 1}\n{"ins')
    assert len(load_checkpoint(tmp_path)) == 1


def test_cpsat_runs_one_seed_above_n20_and_all_seeds_below():
    seeds = (1, 2, 3, 4, 5)
    assert seeds_for("cpsat", 20, seeds, False) == seeds
    assert seeds_for("cpsat", 25, seeds, False) == (1,)
    assert seeds_for("ils", 40, seeds, False) == seeds
    assert seeds_for("highs", 40, seeds, True) == (None,)


def _row(solver, dv, bound=None, feasible=True):
    return {
        "instance": "a",
        "solver": solver,
        "feasible": feasible,
        "total_dv_kms": dv,
        "num_samples": None,
        "certified_lower_bound_kms": bound,
    }


def test_best_known_meeting_a_certified_bound_is_proven():
    rows = [
        _row("exact", None, feasible=False),
        _row("greedy", 2.0),
        _row("highs", 2.0, bound=2.0 - 1e-6),
    ]
    add_references(rows, {"a": random_instance(4, 3)})
    assert {r["reference_kind"] for r in rows} == {"proven"}
    assert rows[0]["reference_bound_solver"] == "highs"


def test_best_known_above_its_bound_says_best_known_and_prints_the_gap():
    rows = [_row("greedy", 2.0), _row("cpsat", 2.0, bound=1.5), _row("highs", None, 1.6, False)]
    add_references(rows, {"a": random_instance(4, 3)})
    assert {r["reference_kind"] for r in rows} == {"best-known"}
    assert rows[0]["reference_lower_bound_kms"] == 1.6  # the best bound, even from a miss
    assert rows[0]["reference_bound_gap_pct"] == pytest.approx(20.0)


def test_a_bound_above_the_reference_is_a_bug_and_raises():
    rows = [_row("exact", 1.0), _row("cpsat", 1.0, bound=1.1)]
    with pytest.raises(ValueError, match="exceeds"):
        add_references(rows, {"a": random_instance(4, 3)})


@pytest.mark.parametrize("n", [40, 60])
def test_sampler_baselines_survive_n_where_2_to_the_n_squared_overflows(n):
    """The first canonical v2 run crashed here at N=40, after every solver had run.

    2**(N*N) overflows a float from N=32. The probability is reported in log10
    always, and as a float only where one can hold it -- never as a misleading 0.0.
    """
    import math

    from dextrivia.bench import log10_uniform_probability

    instance = random_instance(n, 1)
    rows = [
        _row("greedy", 5.0),
        {
            **_row("sa-qubo", 6.0),
            "num_samples": 500,
            "best_raw_dv_kms": 6.0,
            "_best_raw_sample_count": 3,
        },
    ]
    add_references(rows, {"a": instance})
    sampler = rows[1]
    expected = (math.lgamma(n + 1) - n * n * math.log(2)) / math.log(10)
    assert sampler["log10_uniform_feasible_probability"] == pytest.approx(expected)
    assert sampler["uniform_feasible_probability"] is None  # below float range, not 0.0
    assert sampler["num_optimal_sequences"] is None  # not enumerable at this N
    assert sampler["uniform_optimal_probability"] is None
    assert sampler["optimal_sample_probability"] == 0.0  # it did not hit the reference
    assert log10_uniform_probability(math.log(24), 4) == pytest.approx(math.log10(24 / 2**16))


def test_small_n_baselines_are_unchanged_by_the_log_space_rewrite():
    instance = random_instance(4, 3)
    rows = [{**_row("exact", 1.0)}, {**_row("qaoa", 1.0), "num_samples": 10}]
    rows[1].update(best_raw_dv_kms=1.0, _best_raw_sample_count=2)
    add_references(rows, {"a": instance})
    assert rows[1]["uniform_feasible_probability"] == pytest.approx(24 / 2**16)


def test_manifest_shows_src_changes_when_rows_come_from_an_older_commit(monkeypatch):
    seen = {}

    def fake_git(*args):
        seen["args"] = args
        return " src/dextrivia/bench.py | 3 ++-"

    monkeypatch.setattr(bench, "_git", fake_git)
    segments = [{"git_sha": "old"}, {"git_sha": "new"}]
    changes = bench.code_changes_since_runs(segments, {"sha": "new"})
    assert changes == [
        {"runs_at": "old", "written_at": "new", "src_diff_stat": " src/dextrivia/bench.py | 3 ++-"}
    ]
    assert seen["args"] == ("diff", "--stat", "old..new", "--", "src/")
    assert bench.code_changes_since_runs([{"git_sha": "new"}], {"sha": "new"}) == []
