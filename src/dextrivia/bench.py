"""The benchmark harness: every solver, every instance, every seed, one CSV.

    dextrivia bench                                   # the whole family
    dextrivia bench --solvers greedy,exact --seeds 1  # a quick one

Writes ``results/<UTC timestamp>/`` containing

    results.csv        one row per (instance, solver, seed) run
    summary.csv        one row per (instance, solver): mean, spread, best
    penalty_sweep.csv  QUBO feasibility and quality vs penalty weight
    manifest.json      git SHA, instance hashes, library versions, hardware

Three rules this harness exists to enforce:

1. **A miss is a row, not a crash.** A solver that cannot handle an instance
   returns ``feasible=False`` with a reason; that becomes a row with the reason
   in it. The only thing worse than a benchmark with holes is a benchmark that
   hides them.
2. **The oracle is labelled.** ``reference_kind`` is ``exact`` when Held-Karp
   ran; ``proven`` when Held-Karp could not but a certified lower bound
   (``highs``, ``cpsat``) meets the best path found; and ``best-known``
   otherwise, printed with its bound and the gap to it. A gap against a
   best-known value is not an optimality gap and is never called one.
3. **Costs are read through ``instance.leg_costs``**, never ``instance.costs``.

Stochastic solvers are run once per seed and reported with a mean and a spread.
Deterministic ones (``dextrivia.solvers.DETERMINISTIC``) are run once: three
identical Held-Karp runs measure nothing and cost oracle time. ``cpsat`` is
run at one seed above N=20 (``SINGLE_SEED_ABOVE_N``): it burns its full limit
there, and five 60 s runs of a time-limited MIP is five hours of a full run.

A run is **resumable**. Every finished (instance, solver, seed) row is appended
to ``runs.jsonl`` in the output directory as it completes; re-running with the
same ``--out`` skips rows already there. Each row records the git state of the
segment that produced it, so a resumed run cannot hide a code change between
segments -- the manifest lists every segment.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from dextrivia.core import ProblemInstance, Solution
from dextrivia.solvers import DETERMINISTIC, SOLVERS

__all__ = ["run_bench", "discover_instances", "add_references", "summarize"]

#: Five seeds, so every stochastic number in the README has a spread behind it.
DEFAULT_SEEDS = (1, 2, 3, 4, 5)

#: Multipliers on ``path_upper_bound`` for the penalty study. Spans the
#: provably-insufficient region (< 1.0), the shipped default (1.1), and far
#: enough above it to show quality degrading without feasibility improving.
DEFAULT_PENALTY_FACTORS = (0.5, 1.0, 1.05, 1.1, 1.5, 2.0, 4.0, 8.0)

#: Above this, "mean delta-v of a random permutation" is sampled, not enumerated.
EXACT_ENUMERATION_MAX_N = 8

#: Monte Carlo draws when enumeration is out of reach.
RANDOM_PERMUTATION_SAMPLES = 100_000

#: What a runtime in this table actually measures. A configured time limit is
#: not a measurement, and a simulator's wall clock is not a quantum runtime.
RUNTIME_NOTES = {
    "qaoa": "classical statevector simulation time",
    "qaoa-swap": "classical subspace simulation time",
    "ortools": "configured time limit, not a measurement",
    "cpsat": "configured time limit unless proven_optimal",
    "highs": "configured time limit unless proven_optimal",
}

#: Solvers run at the first seed only above this N. The summary's ``runs``
#: column shows the effect, and the manifest records this table.
SINGLE_SEED_ABOVE_N = {"cpsat": 20}

#: A best-known value within this of a certified bound is ``proven``. Bounds
#: already carry their own safety margin (1e-6 km/s for highs, (N-1)*0.5 mm/s
#: rounding for cpsat); this is 1 cm/s, still far below any reported gap.
PROVEN_TOLERANCE_KMS = 1e-5

#: Checkpoint file, one JSON row per completed run.
CHECKPOINT = "runs.jsonl"

DEFAULT_RUNTIME_NOTE = "wall clock"

#: Packages whose version changes a number in this table.
TRACKED_PACKAGES = (
    "numpy",
    "sgp4",
    "dimod",
    "dwave-samplers",
    "qiskit",
    "scipy",
    "ortools",
    "matplotlib",
)

CSV_COLUMNS = (
    "instance",
    "instance_file",
    "n",
    "variant",
    "time_dependent",
    "solver",
    "seed",
    "deterministic",
    "feasible",
    "reason",
    "total_dv_kms",
    "reference_dv_kms",
    "reference_kind",
    "gap_pct",
    "runtime_s",
    "runtime_note",
    "feasibility_rate",
    "feasibility_is_structural",
    "best_raw_dv_kms",
    "best_repaired_dv_kms",
    "best_was_raw_sample",
    "mean_feasible_dv_kms",
    "mean_random_permutation_dv_kms",
    "random_permutation_method",
    "optimal_sample_probability",
    "uniform_optimal_probability",
    "uniform_feasible_probability",
    "num_optimal_sequences",
    "num_samples",
    "num_variables",
    "penalty",
    "proven_optimal",
    "certified_lower_bound_kms",
    "reference_lower_bound_kms",
    "reference_bound_solver",
    "reference_bound_gap_pct",
    "sequence_norad",
    "git_sha",
)

PENALTY_COLUMNS = (
    "instance",
    "n",
    "num_variables",
    "seed",
    "penalty_factor",
    "penalty",
    "path_upper_bound_kms",
    "provably_sufficient",
    "feasibility_rate",
    "best_raw_dv_kms",
    "best_repaired_dv_kms",
    "mean_feasible_dv_kms",
    "reference_dv_kms",
    "gap_pct",
    "reason",
)

SUMMARY_COLUMNS = (
    "instance",
    "n",
    "variant",
    "solver",
    "runs",
    "feasible_runs",
    "reference_dv_kms",
    "reference_kind",
    "reference_lower_bound_kms",
    "reference_bound_solver",
    "reference_bound_gap_pct",
    "dv_best_kms",
    "dv_mean_kms",
    "dv_std_kms",
    "gap_best_pct",
    "gap_mean_pct",
    "gap_std_pct",
    "gap_worst_pct",
    "runtime_mean_s",
    "runtime_std_s",
    "runtime_note",
    "feasibility_rate_mean",
    "reason",
)


def default_instance_dir() -> Path:
    from dextrivia.snapshots import default_snapshot_dir

    return default_snapshot_dir().parent / "instances"


def default_results_dir() -> Path:
    return default_instance_dir().parent.parent / "results"


def instance_label(path: Path) -> str:
    """Short name for tables: ``..._planecluster-v1_n8_td30d.npz`` -> ``n8_td30d``."""
    stem = path.stem
    marker = "_n"
    index = stem.rfind(marker)
    return stem[index + 1 :] if index >= 0 else stem


def discover_instances(instance_dir: Path, patterns: list[str] | None = None) -> list[Path]:
    """Every ``.npz`` under ``instance_dir``, filtered by substring, N then variant."""
    paths = sorted(Path(instance_dir).glob("*.npz"))
    if patterns:
        paths = [p for p in paths if any(pattern in p.name for pattern in patterns)]

    def order(path: Path) -> tuple[int, str]:
        instance = ProblemInstance.load(path)
        return (instance.n, path.name)

    return sorted(paths, key=order)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    """Repository state, captured at the START of a run.

    Captured at the start rather than the end because that is the state that
    actually produced the numbers -- a file edited while a 60-minute benchmark
    runs did not influence it. ``dirty_paths`` is listed rather than summarised
    to a boolean so a reader can see whether the uncommitted work was solver
    code or a README, which is the difference between a void result and a
    cosmetic one.
    """
    porcelain = _git("status", "--porcelain") or ""
    # Split on the status field rather than slicing a fixed 3 characters:
    # porcelain writes " M path" with a leading space, and _git strips its
    # output, so the FIRST line is one character shorter than the rest. Fixed
    # slicing silently turned "results/..." into "esults/..." -- a corrupted
    # path in the one field whose whole job is to be auditable.
    dirty = sorted(
        line.strip().split(maxsplit=1)[-1] for line in porcelain.splitlines() if line.strip()
    )
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_paths": dirty,
    }


def warm_backends() -> dict[str, str]:
    """Import optional backends before anything is timed.

    The first ``ortools`` import costs about 13 seconds on this machine. Paid
    inside a measured solve it would land entirely on whichever instance
    happened to run first, and look like that instance being hard.
    """
    status = {}
    for label, module in (
        ("dwave-samplers", "dwave.samplers"),
        ("qiskit", "qiskit"),
        ("ortools-routing", "ortools.constraint_solver.pywrapcp"),
        ("ortools-cpsat", "ortools.sat.python.cp_model"),
        ("scipy", "scipy.optimize"),
    ):
        try:
            __import__(module)
            status[label] = "available"
        except ImportError as exc:
            status[label] = f"unavailable: {exc}"
    return status


def mean_random_permutation_dv(instance: ProblemInstance, seed: int = 0) -> tuple[float, str]:
    """Mean path cost of a uniformly random visiting order.

    The baseline any sampler has to beat before its mean means anything.
    Enumerated exactly for small N -- at N=4 that is 24 orders, so the number is
    the truth rather than an estimate of it.
    """
    n = instance.n
    if n <= EXACT_ENUMERATION_MAX_N:
        costs = [instance.path_cost(p) for p in itertools.permutations(range(n))]
        return float(np.mean(costs)), f"exact ({len(costs)} permutations)"
    rng = np.random.default_rng(seed)
    costs = [instance.path_cost(rng.permutation(n)) for _ in range(RANDOM_PERMUTATION_SAMPLES)]
    return float(np.mean(costs)), f"sampled ({RANDOM_PERMUTATION_SAMPLES} draws)"


def count_optimal_sequences(instance: ProblemInstance, optimum: float) -> int | None:
    """How many orders achieve ``optimum``. ``None`` when N is too big to enumerate.

    Usually more than one: reversing an open path costs the same whenever the
    cost matrix is symmetric and static, so a "unique optimal bitstring" is an
    assumption, not a fact, and the uniform baseline has to use the real count.
    """
    if instance.n > EXACT_ENUMERATION_MAX_N:
        return None
    return sum(
        1
        for p in itertools.permutations(range(instance.n))
        if instance.path_cost(p) <= optimum + 1e-12
    )


def run_one(
    solver_name: str,
    instance: ProblemInstance,
    path: Path,
    seed: int | None,
    deterministic: bool,
) -> dict[str, Any]:
    """One solver on one instance at one seed. Never raises on a solver miss."""
    solution: Solution = SOLVERS[solver_name]().solve(instance, seed=seed)
    meta = solution.metadata
    row: dict[str, Any] = {
        "instance": instance_label(path),
        "instance_file": path.name,
        "n": instance.n,
        "variant": "td" if instance.time_dependent else "static",
        "time_dependent": instance.time_dependent,
        "solver": solver_name,
        "seed": seed,
        "deterministic": deterministic,
        "feasible": solution.feasible,
        "reason": meta.get("reason"),
        "total_dv_kms": solution.total_dv_kms if solution.feasible else None,
        "runtime_s": solution.runtime_s,
        "runtime_note": RUNTIME_NOTES.get(solver_name, DEFAULT_RUNTIME_NOTE),
        "feasibility_rate": meta.get("feasibility_rate"),
        "feasibility_is_structural": meta.get("feasibility_is_structural"),
        "best_raw_dv_kms": meta.get("best_raw_dv_kms"),
        "best_repaired_dv_kms": meta.get("best_repaired_dv_kms"),
        "best_was_raw_sample": meta.get("best_was_raw_sample"),
        "mean_feasible_dv_kms": meta.get("mean_feasible_dv_kms"),
        "num_samples": meta.get("num_samples"),
        "num_variables": meta.get("num_variables") or meta.get("qubits"),
        "penalty": meta.get("penalty"),
        "proven_optimal": meta.get("proven_optimal"),
        # Was meta["best_objective_bound_kms"], a key no solver ever wrote:
        # this column was empty in every row of the first canonical run.
        "certified_lower_bound_kms": meta.get("lower_bound_kms"),
        "sequence_norad": " ".join(str(i) for i in solution.norad_order(instance)),
        # Kept out of the CSV, used by add_references to turn a best-of-shots
        # count into a probability once the optimum is known.
        "_best_raw_sample_count": meta.get("best_raw_sample_count"),
    }
    return row


def add_references(rows: list[dict[str, Any]], instances: dict[str, ProblemInstance]) -> None:
    """Fill in reference, gap, and the sampler-vs-chance columns. In place.

    The reference is Held-Karp where Held-Karp ran, and otherwise the best
    delta-v any solver in this run achieved. That is ``proven`` when the best
    certified lower bound any row reported meets it (within
    ``PROVEN_TOLERANCE_KMS``) and ``best-known`` when it does not -- a gap
    measured against the best thing we happened to find is not an optimality
    gap and must not be printed as one; the bound and the gap to it are
    printed beside it instead.

    A bound above the reference is a bug in a bound, not a result, and raises.
    """
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], []).append(row)

    for label, group in by_instance.items():
        instance = instances[label]
        feasible = [r for r in group if r["feasible"] and r["total_dv_kms"] is not None]

        bounds = [r for r in group if r.get("certified_lower_bound_kms") is not None]
        best_bound = max(bounds, key=lambda r: r["certified_lower_bound_kms"], default=None)
        bound = best_bound["certified_lower_bound_kms"] if best_bound else None

        exact = [r for r in feasible if r["solver"] == "exact"]
        if exact:
            reference, kind = min(r["total_dv_kms"] for r in exact), "exact"
        elif feasible:
            reference = min(r["total_dv_kms"] for r in feasible)
            proven = bound is not None and reference - bound <= PROVEN_TOLERANCE_KMS
            kind = "proven" if proven else "best-known"
        else:
            reference, kind = None, "none"
        if bound is not None and reference is not None and bound > reference + 1e-9:
            raise ValueError(
                f"{label}: certified bound {bound} from {best_bound['solver']} exceeds "
                f"the {kind} reference {reference} -- a bound is wrong"
            )

        random_mean, random_method = (None, None)
        optima = None
        if reference is not None and any(r["num_samples"] for r in group):
            random_mean, random_method = mean_random_permutation_dv(instance)
            optima = count_optimal_sequences(instance, reference)

        for row in group:
            row["reference_dv_kms"] = reference
            row["reference_kind"] = kind
            row["reference_lower_bound_kms"] = bound
            row["reference_bound_solver"] = best_bound["solver"] if best_bound else None
            row["reference_bound_gap_pct"] = (
                max(0.0, (reference - bound) / reference * 100.0)
                if bound is not None and reference
                else None
            )
            row["gap_pct"] = (
                (row["total_dv_kms"] - reference) / reference * 100.0
                if reference and row["total_dv_kms"] is not None and reference > 0
                else None
            )
            if not row["num_samples"]:
                continue
            # Sampler-only columns. The uniform baselines are what turn a
            # best-of-shots result into a claim about the algorithm: at N=4
            # there are 24 permutations among 2**16 bitstrings, and best-of-4096
            # plus repair finds the optimum from pure noise.
            total_bitstrings = 2.0 ** (instance.n * instance.n)
            permutations = float(math.factorial(instance.n))
            row["mean_random_permutation_dv_kms"] = random_mean
            row["random_permutation_method"] = random_method
            row["uniform_feasible_probability"] = permutations / total_bitstrings
            row["num_optimal_sequences"] = optima
            row["uniform_optimal_probability"] = (
                optima / total_bitstrings if optima is not None else None
            )
            hit_optimum = (
                row["best_raw_dv_kms"] is not None
                and reference is not None
                and row["best_raw_dv_kms"] <= reference + 1e-12
            )
            row["optimal_sample_probability"] = (
                (row["_best_raw_sample_count"] or 0) / row["num_samples"] if hit_optimum else 0.0
            )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse seeds into a mean, a spread and a best, per (instance, solver)."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["instance"], row["solver"]), []).append(row)

    summary = []
    for (label, solver), group in grouped.items():
        head = group[0]
        feasible = [r for r in group if r["feasible"] and r["total_dv_kms"] is not None]
        gaps = [r["gap_pct"] for r in feasible if r["gap_pct"] is not None]
        dvs = [r["total_dv_kms"] for r in feasible]
        runtimes = [r["runtime_s"] for r in group]
        rates = [r["feasibility_rate"] for r in group if r["feasibility_rate"] is not None]
        summary.append(
            {
                "instance": label,
                "n": head["n"],
                "variant": head["variant"],
                "solver": solver,
                "runs": len(group),
                "feasible_runs": len(feasible),
                "reference_dv_kms": head.get("reference_dv_kms"),
                "reference_kind": head.get("reference_kind"),
                "reference_lower_bound_kms": head.get("reference_lower_bound_kms"),
                "reference_bound_solver": head.get("reference_bound_solver"),
                "reference_bound_gap_pct": head.get("reference_bound_gap_pct"),
                "dv_best_kms": min(dvs) if dvs else None,
                "dv_mean_kms": float(np.mean(dvs)) if dvs else None,
                "dv_std_kms": float(np.std(dvs, ddof=0)) if dvs else None,
                "gap_best_pct": min(gaps) if gaps else None,
                "gap_mean_pct": float(np.mean(gaps)) if gaps else None,
                "gap_std_pct": float(np.std(gaps, ddof=0)) if gaps else None,
                "gap_worst_pct": max(gaps) if gaps else None,
                "runtime_mean_s": float(np.mean(runtimes)),
                "runtime_std_s": float(np.std(runtimes, ddof=0)),
                "runtime_note": head["runtime_note"],
                "feasibility_rate_mean": float(np.mean(rates)) if rates else None,
                # Only meaningful when nothing succeeded; that is exactly when
                # a reader needs to know why the cell is empty.
                "reason": None if feasible else head.get("reason"),
            }
        )
    return sorted(summary, key=lambda r: (r["n"], r["variant"], r["solver"]))


def _sample_at(solver: Any, seed: int, qubo: Any) -> np.ndarray:
    """Bound sampler for ``sweep_penalty``. A module function, not a closure over
    the loop variables, which is the classic way to sweep every factor with the
    last seed."""
    return solver.sample(qubo, seed=seed)


def penalty_study(
    instances: dict[str, ProblemInstance],
    seeds: tuple[int, ...],
    factors: tuple[float, ...],
    references: dict[str, float | None],
) -> list[dict[str, Any]]:
    """``sweep_penalty`` across the time-dependent instances, at every seed.

    Time-dependent only: the static instances rank nothing (greedy already ties
    Held-Karp on every one of them), so sweeping a penalty on them would measure
    the encoding's overhead on a problem with no headroom.
    """
    from dextrivia.qubo import sweep_penalty
    from dextrivia.solvers.quantum_annealing import SimulatedAnnealingSolver

    rows = []
    for label, instance in instances.items():
        if not instance.time_dependent:
            continue
        for seed in seeds:
            solver = SimulatedAnnealingSolver()
            try:
                sample_fn = functools.partial(_sample_at, solver, seed)
                records = sweep_penalty(instance, sample_fn, factors=factors)
            except ImportError as exc:
                rows.append({"instance": label, "n": instance.n, "seed": seed, "reason": str(exc)})
                break
            reference = references.get(label)
            for record in records:
                rows.append(
                    {
                        "instance": label,
                        "n": instance.n,
                        "num_variables": instance.n**2,
                        "seed": seed,
                        "penalty_factor": record["penalty_factor"],
                        "penalty": record["penalty"],
                        "path_upper_bound_kms": record["path_upper_bound_kms"],
                        "provably_sufficient": record["provably_sufficient"],
                        "feasibility_rate": record["feasibility_rate"],
                        "best_raw_dv_kms": record["best_raw_dv_kms"],
                        "best_repaired_dv_kms": record["best_repaired_dv_kms"],
                        "mean_feasible_dv_kms": record["mean_feasible_dv_kms"],
                        "reference_dv_kms": reference,
                        "gap_pct": (
                            (record["best_repaired_dv_kms"] - reference) / reference * 100.0
                            if reference
                            else None
                        ),
                        "reason": None,
                    }
                )
    return rows


def seeds_for(
    name: str, n: int, seeds: tuple[int, ...], deterministic: bool
) -> tuple[int | None, ...]:
    """Which seeds ``name`` runs at on an N-object instance. See ``SINGLE_SEED_ABOVE_N``."""
    if deterministic:
        return (None,)
    limit = SINGLE_SEED_ABOVE_N.get(name)
    return seeds[:1] if limit is not None and n > limit else seeds


def run_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    return (row["instance_file"], row["solver"], row["seed"])


def append_checkpoint(out_dir: Path, row: dict[str, Any]) -> None:
    """Append one finished run. Flushed per row, so a killed run loses at most one."""
    with (Path(out_dir) / CHECKPOINT).open("a") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(out_dir: Path) -> list[dict[str, Any]]:
    """Rows already completed in ``out_dir``. A torn last line (killed mid-write) is dropped."""
    path = Path(out_dir) / CHECKPOINT
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per run segment (a resume starts a new one), with its git state."""
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("_segment_started_utc") or "unknown"
        entry = seen.setdefault(
            key,
            {
                "started_utc": key,
                "git_sha": row.get("git_sha"),
                "dirty_paths": row.get("_git_dirty_paths"),
                "runs": 0,
            },
        )
        entry["runs"] += 1
    return sorted(seen.values(), key=lambda e: e["started_utc"])


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    instance_paths: list[Path],
    instances: dict[str, ProblemInstance],
    solvers: list[str],
    seeds: tuple[int, ...],
    factors: tuple[float, ...],
    argv: list[str],
    backends: dict[str, str],
    started: datetime,
    elapsed_s: float,
    git: dict[str, Any],
) -> dict[str, Any]:
    """Everything needed to say what produced these numbers, and on what."""
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"

    snapshots = sorted(
        {str(i.metadata.get("snapshot")) for i in instances.values() if i.metadata.get("snapshot")}
    )
    return {
        "timestamp_utc": started.isoformat(),
        "elapsed_s": elapsed_s,
        "command": " ".join(argv),
        # A benchmark run from a dirty tree is not reproducible from the SHA
        # alone, and the manifest is the only place that can admit it.
        "git": git,
        "snapshots": snapshots,
        "instances": [
            {
                "file": path.name,
                "sha256": _sha256(path),
                "n": instances[instance_label(path)].n,
                "time_dependent": instances[instance_label(path)].time_dependent,
                "metadata": instances[instance_label(path)].metadata,
            }
            for path in instance_paths
        ],
        # Configuration, not repr(): a repr carries a memory address, which
        # would make two identical runs produce different manifests.
        "solvers": {
            name: {"class": SOLVERS[name].__name__, **vars(SOLVERS[name]())} for name in solvers
        },
        "deterministic_solvers": sorted(set(solvers) & set(DETERMINISTIC)),
        "seeds": list(seeds),
        "penalty_factors": list(factors),
        "backends": backends,
        "versions": versions,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
        },
    }


def run_bench(
    instance_dir: Path | None = None,
    out_dir: Path | None = None,
    solvers: list[str] | None = None,
    patterns: list[str] | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    factors: tuple[float, ...] = DEFAULT_PENALTY_FACTORS,
    run_penalty_study: bool = True,
    argv: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """Run the benchmark and write a results directory. Returns its path."""
    started = datetime.now(UTC)
    clock = started.timestamp()
    git = git_state()
    instance_dir = Path(instance_dir) if instance_dir else default_instance_dir()
    paths = discover_instances(instance_dir, patterns)
    if not paths:
        raise SystemExit(f"no instances matching {patterns or '*'} in {instance_dir}")

    names = solvers or list(SOLVERS)
    unknown = [name for name in names if name not in SOLVERS]
    if unknown:
        raise SystemExit(f"unknown solver(s): {', '.join(unknown)}; have {', '.join(SOLVERS)}")

    backends = warm_backends()
    instances = {instance_label(p): ProblemInstance.load(p) for p in paths}

    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_dir) if out_dir else default_results_dir() / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_checkpoint(out_dir)
    done = {run_key(r) for r in rows}
    if verbose and rows:
        print(f"resuming {out_dir}: {len(rows)} runs already done")

    for path in paths:
        instance = instances[instance_label(path)]
        for name in names:
            deterministic = name in DETERMINISTIC
            for seed in seeds_for(name, instance.n, seeds, deterministic):
                if (path.name, name, seed) in done:
                    continue
                row = run_one(name, instance, path, seed, deterministic)
                row.update(git_sha=git["sha"], _segment_started_utc=started.isoformat())
                row["_git_dirty_paths"] = git["dirty_paths"]
                append_checkpoint(out_dir, row)
                rows.append(row)
                if verbose:
                    outcome = (
                        f"{row['total_dv_kms']:.4f} km/s"
                        if row["feasible"]
                        else f"skipped ({row['reason']})"
                    )
                    seed_label = "-" if seed is None else seed
                    print(
                        f"  {row['instance']:<12} {name:<12} seed={seed_label:<4} "
                        f"{row['runtime_s']:7.2f}s  {outcome}"
                    )

    add_references(rows, instances)
    summary = summarize(rows)
    references = {label: None for label in instances}
    for row in rows:
        references[row["instance"]] = row["reference_dv_kms"]

    sweep_rows: list[dict[str, Any]] = []
    if run_penalty_study:
        if verbose:
            print("penalty study (time-dependent instances)...")
        sweep_rows = penalty_study(instances, seeds, factors, references)

    _write_csv(out_dir / "results.csv", CSV_COLUMNS, rows)
    _write_csv(out_dir / "summary.csv", SUMMARY_COLUMNS, summary)
    if sweep_rows:
        _write_csv(out_dir / "penalty_sweep.csv", PENALTY_COLUMNS, sweep_rows)

    elapsed = datetime.now(UTC).timestamp() - clock
    manifest = build_manifest(
        paths, instances, names, seeds, factors, argv or sys.argv, backends, started, elapsed, git
    )
    manifest["segments"] = segments(rows)
    manifest["single_seed_above_n"] = SINGLE_SEED_ABOVE_N
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    if verbose:
        print(f"\nwrote {out_dir}")
        print(f"  {len(rows)} runs, {sum(1 for r in rows if not r['feasible'])} skipped")
        print(f"  elapsed {elapsed:.1f}s")
    return out_dir


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--instance-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: results/<timestamp>")
    parser.add_argument(
        "--solvers", default=None, help=f"comma-separated; default all of {','.join(SOLVERS)}"
    )
    parser.add_argument(
        "--instances", default=None, help="comma-separated substrings, e.g. n4,n8_td"
    )
    parser.add_argument("--seeds", default=None, help="comma-separated; default 1,2,3,4,5")
    parser.add_argument(
        "--no-penalty-study",
        action="store_true",
        help="skip the QUBO penalty sweep (it is the slowest part of a full run)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(args: argparse.Namespace) -> int:
    run_bench(
        instance_dir=args.instance_dir,
        out_dir=args.out,
        solvers=args.solvers.split(",") if args.solvers else None,
        patterns=args.instances.split(",") if args.instances else None,
        seeds=tuple(int(s) for s in args.seeds.split(",")) if args.seeds else DEFAULT_SEEDS,
        run_penalty_study=not args.no_penalty_study,
        verbose=not args.quiet,
    )
    return 0
