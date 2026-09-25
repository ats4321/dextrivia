"""The benchmark harness. Its job is to record misses, not to crash on them.

Every test here uses tiny random instances written to a tmp_path: this file is
about the harness's bookkeeping, not about any solver's delta-v.
"""

from __future__ import annotations

import csv
import json

import pytest

from dextrivia.bench import (
    DEFAULT_SEEDS,
    add_references,
    discover_instances,
    instance_label,
    mean_random_permutation_dv,
    run_bench,
    summarize,
)
from tests.test_qubo_formulation import random_instance


@pytest.fixture
def instance_dir(tmp_path):
    """Two instances: one Held-Karp can solve, one static and one time-slotted."""
    out = tmp_path / "instances"
    out.mkdir()
    random_instance(4, 1).save(out / "snap_planecluster-v9_n4_static.npz")
    random_instance(4, 2, time_dependent=True).save(out / "snap_planecluster-v9_n4_td30d.npz")
    return out


def read_csv(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_instance_label_strips_the_snapshot_prefix():
    from pathlib import Path

    assert instance_label(Path("iridium33_20260402_planecluster-v1_n8_td30d.npz")) == "n8_td30d"


def test_discover_orders_by_n_then_name(tmp_path):
    random_instance(6, 1).save(tmp_path / "x_n6_static.npz")
    random_instance(3, 1).save(tmp_path / "x_n3_static.npz")
    assert [p.name for p in discover_instances(tmp_path)] == [
        "x_n3_static.npz",
        "x_n6_static.npz",
    ]


def test_instance_filter_is_a_substring_match(instance_dir):
    assert [instance_label(p) for p in discover_instances(instance_dir, ["td30d"])] == ["n4_td30d"]


def test_run_writes_every_artifact(instance_dir, tmp_path):
    out = run_bench(
        instance_dir=instance_dir,
        out_dir=tmp_path / "run",
        solvers=["greedy", "exact"],
        seeds=(1,),
        run_penalty_study=False,
        verbose=False,
    )
    assert (out / "results.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "manifest.json").exists()


def test_manifest_records_what_would_change_the_numbers(instance_dir, tmp_path):
    out = run_bench(
        instance_dir=instance_dir,
        out_dir=tmp_path / "run",
        solvers=["greedy"],
        seeds=(1, 2),
        run_penalty_study=False,
        verbose=False,
    )
    manifest = json.loads((out / "manifest.json").read_text())

    assert manifest["git"]["sha"]
    # Listed, not summarised to a boolean: uncommitted solver code voids a run,
    # an uncommitted README does not, and only the paths distinguish them.
    assert isinstance(manifest["git"]["dirty_paths"], list)
    assert manifest["git"]["dirty"] == bool(manifest["git"]["dirty_paths"])
    assert manifest["seeds"] == [1, 2]
    assert manifest["versions"]["numpy"]
    assert manifest["platform"]["python"]
    # Every instance is pinned by content hash, so a regenerated family that
    # happens to keep its filename cannot silently restate old numbers.
    assert len(manifest["instances"]) == 2
    for record in manifest["instances"]:
        assert len(record["sha256"]) == 64


def test_a_solver_that_cannot_run_becomes_a_row_with_a_reason(instance_dir, tmp_path):
    """The contract the whole harness exists for: record the miss, keep going."""
    out = run_bench(
        instance_dir=instance_dir,
        out_dir=tmp_path / "run",
        solvers=["greedy", "ortools"],
        seeds=(1,),
        run_penalty_study=False,
        verbose=False,
    )
    rows = read_csv(out / "results.csv")
    refused = [r for r in rows if r["solver"] == "ortools" and r["instance"] == "n4_td30d"]

    assert len(refused) == 1
    assert refused[0]["feasible"] == "False"
    assert "time-slotted" in refused[0]["reason"]
    assert refused[0]["total_dv_kms"] == ""
    # and the run as a whole still produced results for everything else
    assert any(r["feasible"] == "True" for r in rows)


def test_deterministic_solvers_run_once_whatever_the_seed_list(instance_dir, tmp_path):
    out = run_bench(
        instance_dir=instance_dir,
        out_dir=tmp_path / "run",
        solvers=["exact", "local-search"],
        seeds=(1, 2, 3),
        run_penalty_study=False,
        verbose=False,
    )
    rows = read_csv(out / "results.csv")
    exact = [r for r in rows if r["solver"] == "exact" and r["instance"] == "n4_static"]
    search = [r for r in rows if r["solver"] == "local-search" and r["instance"] == "n4_static"]

    assert len(exact) == 1
    assert exact[0]["seed"] == ""
    assert exact[0]["deterministic"] == "True"
    assert len(search) == 3


def test_reference_is_exact_when_held_karp_ran():
    instance = random_instance(4, 3)
    rows = [
        {
            "instance": "a",
            "solver": "exact",
            "feasible": True,
            "total_dv_kms": 1.0,
            "num_samples": None,
        },
        {
            "instance": "a",
            "solver": "greedy",
            "feasible": True,
            "total_dv_kms": 1.5,
            "num_samples": None,
        },
    ]
    add_references(rows, {"a": instance})

    assert rows[0]["reference_kind"] == "exact"
    assert rows[1]["gap_pct"] == pytest.approx(50.0)


def test_reference_falls_back_to_best_known_and_says_so():
    """Beyond Held-Karp range a gap is measured against the best thing found.

    It is not an optimality gap and the label has to make that impossible to
    miss, because it is the number most likely to be quoted out of context.
    """
    instance = random_instance(4, 3)
    rows = [
        {
            "instance": "a",
            "solver": "exact",
            "feasible": False,
            "total_dv_kms": None,
            "num_samples": None,
        },
        {
            "instance": "a",
            "solver": "greedy",
            "feasible": True,
            "total_dv_kms": 2.0,
            "num_samples": None,
        },
        {
            "instance": "a",
            "solver": "sa-perm",
            "feasible": True,
            "total_dv_kms": 2.5,
            "num_samples": None,
        },
    ]
    add_references(rows, {"a": instance})

    assert {r["reference_kind"] for r in rows} == {"best-known"}
    assert rows[1]["gap_pct"] == pytest.approx(0.0)
    assert rows[2]["gap_pct"] == pytest.approx(25.0)


def test_sampler_rows_get_the_uniform_baselines():
    """A best-of-shots result means nothing without the chance baseline beside it."""
    instance = random_instance(4, 3)
    optimum = min(instance.path_cost(p) for p in __import__("itertools").permutations(range(4)))
    rows = [
        {
            "instance": "a",
            "solver": "exact",
            "feasible": True,
            "total_dv_kms": optimum,
            "num_samples": None,
        },
        {
            "instance": "a",
            "solver": "qaoa",
            "feasible": True,
            "total_dv_kms": optimum,
            "num_samples": 1000,
            "best_raw_dv_kms": optimum,
            "_best_raw_sample_count": 40,
        },
    ]
    add_references(rows, {"a": instance})
    sampler = rows[1]

    # 24 permutations among 2**16 bitstrings.
    assert sampler["uniform_feasible_probability"] == pytest.approx(24 / 65536)
    assert sampler["uniform_optimal_probability"] == pytest.approx(
        sampler["num_optimal_sequences"] / 65536
    )
    assert sampler["optimal_sample_probability"] == pytest.approx(0.04)
    assert sampler["random_permutation_method"].startswith("exact")


def test_a_sampler_that_never_found_the_optimum_scores_zero_probability():
    instance = random_instance(4, 3)
    rows = [
        {
            "instance": "a",
            "solver": "exact",
            "feasible": True,
            "total_dv_kms": 0.1,
            "num_samples": None,
        },
        {
            "instance": "a",
            "solver": "sa-qubo",
            "feasible": True,
            "total_dv_kms": 0.5,
            "num_samples": 100,
            "best_raw_dv_kms": 0.5,
            "_best_raw_sample_count": 60,
        },
    ]
    add_references(rows, {"a": instance})
    assert rows[1]["optimal_sample_probability"] == 0.0


def test_mean_random_permutation_is_enumerated_exactly_at_small_n():
    instance = random_instance(4, 3)
    value, method = mean_random_permutation_dv(instance)
    expected = [instance.path_cost(p) for p in __import__("itertools").permutations(range(4))]
    assert value == pytest.approx(sum(expected) / len(expected))
    assert method == "exact (24 permutations)"


def test_summary_carries_a_spread_not_just_a_mean():
    rows = [
        {
            "instance": "a",
            "n": 4,
            "variant": "static",
            "solver": "s",
            "feasible": True,
            "total_dv_kms": 1.0,
            "gap_pct": 0.0,
            "runtime_s": 1.0,
            "feasibility_rate": 1.0,
            "runtime_note": "wall clock",
            "reference_dv_kms": 1.0,
            "reference_kind": "exact",
        },
        {
            "instance": "a",
            "n": 4,
            "variant": "static",
            "solver": "s",
            "feasible": True,
            "total_dv_kms": 2.0,
            "gap_pct": 100.0,
            "runtime_s": 3.0,
            "feasibility_rate": 1.0,
            "runtime_note": "wall clock",
            "reference_dv_kms": 1.0,
            "reference_kind": "exact",
        },
    ]
    summary = summarize(rows)[0]

    assert summary["runs"] == 2
    assert summary["gap_mean_pct"] == pytest.approx(50.0)
    assert summary["gap_std_pct"] == pytest.approx(50.0)
    assert summary["gap_best_pct"] == pytest.approx(0.0)
    assert summary["gap_worst_pct"] == pytest.approx(100.0)
    assert summary["runtime_mean_s"] == pytest.approx(2.0)


def test_summary_keeps_the_reason_when_nothing_succeeded():
    rows = [
        {
            "instance": "a",
            "n": 20,
            "variant": "td",
            "solver": "exact",
            "feasible": False,
            "total_dv_kms": None,
            "gap_pct": None,
            "runtime_s": 0.0,
            "feasibility_rate": None,
            "runtime_note": "wall clock",
            "reason": "N=20 exceeds held-karp limit 18",
        },
    ]
    summary = summarize(rows)[0]
    assert summary["feasible_runs"] == 0
    assert "held-karp" in summary["reason"]


def test_dirty_paths_survive_the_first_line(monkeypatch):
    """git status --porcelain writes " M path"; the captured output is stripped.

    Fixed-width slicing therefore ate a character off the first path only,
    which is how a manifest came to record "esults/canonical/manifest.json".
    A provenance field that quietly corrupts its own contents is worse than
    not having it.
    """
    from dextrivia import bench

    porcelain = " M results/canonical/manifest.json\n?? docs/figures/new.png"
    monkeypatch.setattr(bench, "_git", lambda *a: porcelain.strip() if a[0] == "status" else "x")

    state = bench.git_state()

    assert state["dirty_paths"] == [
        "docs/figures/new.png",
        "results/canonical/manifest.json",
    ]
    assert state["dirty"] is True


def test_default_seed_count_supports_a_spread():
    """Five seeds, because a mean of one run is not a mean."""
    assert len(DEFAULT_SEEDS) >= 5


def test_unknown_solver_is_rejected_before_anything_runs(instance_dir, tmp_path):
    with pytest.raises(SystemExit, match="unknown solver"):
        run_bench(
            instance_dir=instance_dir,
            out_dir=tmp_path / "run",
            solvers=["greedy", "quantum-supremacy"],
            seeds=(1,),
            run_penalty_study=False,
            verbose=False,
        )
