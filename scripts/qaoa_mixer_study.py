"""Does a feasibility-preserving mixer make QAOA optimise *within* the feasible set?

    uv run python scripts/qaoa_mixer_study.py            # ~10 min
    uv run --with qiskit-aer python scripts/qaoa_mixer_study.py --aer-only

The penalty-QUBO ``qaoa`` concentrates amplitude on permutation matrices at
N=4 but its feasible samples are only ~6.6% better than a random permutation.
This script measures ``qaoa-swap`` (uniform start) against the same chance
baselines, at depths 1..5, 5 seeds each, on the committed family:

* raw feasibility -- MEASURED from bitstrings of the synthesised Qiskit circuit
  at the optimised angles where N <= 4; 1.0 by construction above that;
* P(optimal) vs the uniform-over-feasible baseline #optimal / N!;
* mean delta-v vs the exact mean over all N! permutations;
* **captured** = (random mean - QAOA mean) / (random mean - optimum): the share
  of the possible improvement over chance. 0 is chance, 1 is always optimal.

The same "captured" figure is computed for the X-mixer ``qaoa`` from the
committed canonical run, so the comparison is like for like. The optimum comes
from Held-Karp, outside the solver.

Optimisation runs in the subspace simulator (exact; see
tests/test_qaoa_swap.py). ``--aer-only`` instead checks whether Qiskit Aer's
matrix-product-state simulator reproduces the subspace result at N=5 (25
qubits), and how long it takes. qiskit-aer is not a project dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from dextrivia.core import ProblemInstance
from dextrivia.qubo import library_versions
from dextrivia.qubo.permutation_qaoa import PermutationSpace
from dextrivia.snapshots import default_snapshot_dir
from dextrivia.solvers import ExactSolver
from dextrivia.solvers.quantum_qaoa_swap import QAOASwapSolver

SIZES = (4, 5, 8)
VARIANTS = ("static", "td30d")
DEPTHS = (1, 2, 3, 4, 5)
SEEDS = (1, 2, 3, 4, 5)
SHOTS = 4096
MAXITER = 1000
RESTARTS = 3
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "qaoa_mixer_study.json"
CANONICAL = ROOT / "results" / "canonical" / "results.csv"
#: Raw rows, written after every instance; gitignored scratch space.
CHECKPOINT = ROOT / ".context" / "qaoa_mixer_study.rows.json"


def load(n: int, variant: str) -> ProblemInstance:
    name = f"iridium33_20260402_planecluster-v1_n{n}_{variant}.npz"
    return ProblemInstance.load(default_snapshot_dir().parent / "instances" / name)


def measured_feasibility(instance: ProblemInstance, meta: dict, seed: int) -> float:
    """Sample the real, synthesised Qiskit circuit at the optimised angles."""
    from qiskit.quantum_info import Statevector

    from dextrivia.solvers.quantum_qaoa_swap import swap_qaoa_circuit, synthesize

    space = PermutationSpace(instance)
    circuit, g, b = swap_qaoa_circuit(instance, meta["reps"])
    bound = synthesize(circuit).assign_parameters(
        {**dict(zip(g, meta["gammas"], strict=True)), **dict(zip(b, meta["betas"], strict=True))}
    )
    state = Statevector(space.embed(space.uniform_state())).evolve(bound)
    state.seed(seed)
    counts = state.sample_counts(SHOTS)
    n = instance.n
    feasible = 0
    for key, count in counts.items():
        bits = np.array([int(c) for c in key[::-1]]).reshape(n, n)
        if (bits.sum(axis=0) == 1).all() and (bits.sum(axis=1) == 1).all():
            feasible += count
    return feasible / SHOTS


def study() -> dict:
    rows = []
    for n in SIZES:
        for variant in VARIANTS:
            instance = load(n, variant)
            space = PermutationSpace(instance)
            optimum = ExactSolver().solve(instance).total_dv_kms
            assert abs(optimum - space.costs.min()) < 1e-9
            n_opt = int((space.costs <= optimum + 1e-12).sum())
            random_mean = float(space.costs.mean())
            for reps in DEPTHS:
                for seed in SEEDS:
                    t0 = time.perf_counter()
                    solver = QAOASwapSolver(
                        reps=reps, shots=SHOTS, maxiter=MAXITER, restarts=RESTARTS
                    )
                    sol = solver.solve(instance, seed=seed)
                    meta = sol.metadata
                    hit = meta["best_raw_dv_kms"] <= optimum + 1e-12
                    row = {
                        "instance": f"n{n}_{variant}",
                        "n": n,
                        "reps": reps,
                        "seed": seed,
                        "initial_state": "uniform",
                        "feasibility_rate": (
                            measured_feasibility(instance, meta, seed) if n <= 4 else 1.0
                        ),
                        "feasibility_measured": n <= 4,
                        "optimum_kms": optimum,
                        "num_optimal_sequences": n_opt,
                        "uniform_optimal_probability": n_opt / math.factorial(n),
                        "sampled_optimal_probability": (
                            meta["best_raw_sample_count"] / SHOTS if hit else 0.0
                        ),
                        "exact_optimal_probability": meta["exact_optimal_probability"],
                        "random_mean_dv_kms": random_mean,
                        "sampled_mean_dv_kms": meta["mean_feasible_dv_kms"],
                        "exact_mean_dv_kms": meta["exact_mean_dv_kms"],
                        "captured": (random_mean - meta["exact_mean_dv_kms"])
                        / (random_mean - optimum),
                        "best_raw_dv_kms": meta["best_raw_dv_kms"],
                        "objective_evaluations": meta["objective_evaluations"],
                        "restart_expectations": meta["restart_expectations"],
                        "runtime_s": time.perf_counter() - t0,
                    }
                    rows.append(row)
                    print(
                        f"{row['instance']:10s} p={reps} seed={seed} "
                        f"feas={row['feasibility_rate']:.4f} "
                        f"P*={row['exact_optimal_probability']:.4f} "
                        f"(unif {row['uniform_optimal_probability']:.4f}) "
                        f"captured={row['captured']:+.3f} {row['runtime_s']:.1f}s",
                        flush=True,
                    )
            # Checkpoint: a crash in post-processing must not cost the runs.
            CHECKPOINT.write_text(json.dumps(rows, default=float))
    return {"rows": rows, "summary": summarise(rows), "x_mixer_qaoa": x_mixer_baseline()}


def summarise(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["instance"], r["n"], r["reps"]) for r in rows})
    out = []
    for instance, n, reps in keys:
        group = [r for r in rows if (r["instance"], r["reps"]) == (instance, reps)]

        def stat(field: str, group=group) -> tuple[float, float]:
            values = [r[field] for r in group]
            return statistics.mean(values), statistics.stdev(values)

        out.append(
            {
                "instance": instance,
                "n": n,
                "reps": reps,
                "seeds": len(group),
                "feasibility_rate_mean": stat("feasibility_rate")[0],
                "feasibility_measured": group[0]["feasibility_measured"],
                "uniform_optimal_probability": group[0]["uniform_optimal_probability"],
                "exact_optimal_probability": stat("exact_optimal_probability"),
                "sampled_optimal_probability": stat("sampled_optimal_probability"),
                "random_mean_dv_kms": group[0]["random_mean_dv_kms"],
                "exact_mean_dv_kms": stat("exact_mean_dv_kms"),
                "captured": stat("captured"),
                "optimum_kms": group[0]["optimum_kms"],
            }
        )
    return out


def x_mixer_baseline() -> list[dict]:
    """The penalty-QUBO ``qaoa`` rows from the committed canonical run, in the
    same terms. Its P(optimal) is conditioned on feasibility to compare with a
    solver whose samples are all feasible."""
    rows = [r for r in csv.DictReader(CANONICAL.open()) if r["solver"] == "qaoa"]
    out = []
    for instance in sorted({r["instance"] for r in rows}):
        group = [r for r in rows if r["instance"] == instance and r["feasible"] == "True"]
        if not group:  # past the qubit wall: qaoa recorded a miss there
            continue
        n = int(group[0]["n"])
        optimum = float(group[0]["reference_dv_kms"])
        random_mean = float(group[0]["mean_random_permutation_dv_kms"])
        n_opt = int(group[0]["num_optimal_sequences"])
        captured = [
            (random_mean - float(r["mean_feasible_dv_kms"])) / (random_mean - optimum)
            for r in group
        ]
        conditional = [
            float(r["optimal_sample_probability"]) / float(r["feasibility_rate"]) for r in group
        ]
        out.append(
            {
                "instance": instance,
                "source": str(CANONICAL.relative_to(ROOT)),
                "seeds": len(group),
                "feasibility_rate_mean": statistics.mean(
                    float(r["feasibility_rate"]) for r in group
                ),
                "optimal_probability_given_feasible": (
                    statistics.mean(conditional),
                    statistics.stdev(conditional),
                ),
                "uniform_optimal_probability_given_feasible": n_opt / math.factorial(n),
                "captured": (statistics.mean(captured), statistics.stdev(captured)),
            }
        )
    return out


def first_k(instance: ProblemInstance, k: int) -> ProblemInstance:
    """First ``k`` objects of an instance (time-slotted: first k-1 slots)."""
    costs = instance.costs[: k - 1, :k, :k] if instance.time_dependent else instance.costs[:k, :k]
    return ProblemInstance(
        norad_ids=instance.norad_ids[:k],
        names=instance.names[:k],
        epoch=instance.epoch,
        costs=np.array(costs),
        metadata={**instance.metadata, "derivation": f"first {k} objects"},
    )


AER_SHOTS = 20_000
#: Wall-clock budget per Aer circuit. MPS cost grows with entanglement, i.e.
#: with depth; a run that blows through this is recorded, not waited on.
AER_BUDGET_S = 600


def _aer_worker(compiled, seed: int, queue) -> None:
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="matrix_product_state", seed_simulator=seed)
    queue.put(sim.run(compiled, shots=AER_SHOTS).result().get_counts())


def _run_with_budget(compiled, seed: int) -> dict | None:
    """Run in a child process so an over-budget simulation can be killed."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_aer_worker, args=(compiled, seed, queue))
    proc.start()
    try:
        return queue.get(timeout=AER_BUDGET_S)
    except Exception:  # queue.Empty: over budget
        return None
    finally:
        proc.terminate()
        proc.join()


def aer_check() -> dict:
    """Aer MPS against the exact subspace distribution at N=5 (25 qubits) and
    N=7 (49 qubits, first 7 objects of n8 -- no dense statevector can exist),
    plus the attempt at N=8 (64 qubits), which Aer refuses outright.

    Compared by SAMPLING: total variation distance between AER_SHOTS MPS shots
    and the exact distribution, next to the TVD of an equally large sample drawn
    from the exact distribution itself (the shot-noise floor). Not by reading
    amplitudes: qiskit-aer 0.17.2's MPS ``save_amplitudes`` returns wrong values
    on this circuit, recorded below as ``save_amplitudes_bug``, while its
    ``save_statevector`` and its sampling agree with the exact answer.
    """
    from qiskit import transpile
    from qiskit.transpiler.exceptions import CircuitTooWideForTarget
    from qiskit_aer import AerSimulator

    from dextrivia.solvers.quantum_qaoa_swap import swap_qaoa_circuit

    results = []
    for n, reps_list in ((5, (1, 2)), (7, (1, 3)), (8, (1,))):
        for variant in VARIANTS:
            instance = load(n, variant) if n != 7 else first_k(load(8, variant), 7)
            label = f"n{n}_{variant}" if n != 7 else f"n8_{variant} first 7"
            space = PermutationSpace(instance)
            sequence = tuple(range(n))
            # Python ints: at N=8 bit indices reach 63 and overflow int64.
            lookup = {
                sum(1 << (int(obj) * n + p) for p, obj in enumerate(perm)): k
                for k, perm in enumerate(space.perms)
            }
            for reps in reps_list:
                rng = np.random.default_rng(reps)
                gammas, betas = rng.uniform(0, 3, reps), rng.uniform(0, 1.5, reps)
                circuit, g, b = swap_qaoa_circuit(instance, reps, initial_sequence=sequence)
                bound = circuit.assign_parameters(
                    {**dict(zip(g, gammas, strict=True)), **dict(zip(b, betas, strict=True))}
                )
                bound.measure_all()
                sim = AerSimulator(method="matrix_product_state")
                t0 = time.perf_counter()
                try:
                    compiled = transpile(bound, sim, optimization_level=0)
                except CircuitTooWideForTarget as exc:
                    results.append({"instance": label, "qubits": n * n, "error": str(exc)})
                    print(results[-1], flush=True)
                    continue
                t_compile = time.perf_counter() - t0
                t0 = time.perf_counter()
                counts = _run_with_budget(compiled, reps)
                t_run = time.perf_counter() - t0
                if counts is None:
                    results.append(
                        {
                            "instance": label,
                            "qubits": n * n,
                            "reps": reps,
                            "error": f"did not finish within the {AER_BUDGET_S} s budget",
                        }
                    )
                    print(results[-1], flush=True)
                    continue
                t0 = time.perf_counter()
                exact = np.abs(space.evolve(gammas, betas, space.basis_state(sequence))) ** 2
                t_subspace = time.perf_counter() - t0
                empirical, leaked = np.zeros(space.dim), 0
                for key, count in counts.items():
                    k = lookup.get(int(key, 2))
                    if k is None:
                        leaked += count
                    else:
                        empirical[k] += count
                ideal = rng.multinomial(AER_SHOTS, exact / exact.sum()) / AER_SHOTS
                results.append(
                    {
                        "instance": label,
                        "qubits": n * n,
                        "reps": reps,
                        "initial_state": "basis",
                        "shots": AER_SHOTS,
                        "tvd_mps_vs_exact": 0.5
                        * (np.abs(empirical / AER_SHOTS - exact).sum() + leaked / AER_SHOTS),
                        "tvd_ideal_sample_vs_exact": 0.5 * np.abs(ideal - exact).sum(),
                        "leaked_fraction": leaked / AER_SHOTS,
                        "cx_count": int(compiled.count_ops().get("cx", 0)),
                        "aer_transpile_s": t_compile,
                        "aer_run_s": t_run,
                        "subspace_evolve_s": t_subspace,
                    }
                )
                print(results[-1], flush=True)
    return {
        "rows": results,
        "save_amplitudes_bug": _amplitude_readout_bug(),
        "versions": library_versions("qiskit", "qiskit-aer"),
    }


def _amplitude_readout_bug() -> dict:
    """The N=3 reproduction: same circuit, three readouts, one wrong."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from dextrivia.solvers.quantum_qaoa_swap import swap_qaoa_circuit

    instance = first_k(load(4, "static"), 3)
    space = PermutationSpace(instance)
    sequence = (0, 1, 2)
    circuit, g, b = swap_qaoa_circuit(instance, 1, initial_sequence=sequence)
    circuit = circuit.assign_parameters({g[0]: 0.7, b[0]: 0.4})
    indices = [int(i) for i in space.bitstring_indices()]
    exact = space.evolve([0.7], [0.4], space.basis_state(sequence))
    errors = {}
    for method in ("statevector", "matrix_product_state"):
        for readout in ("save_amplitudes", "save_statevector"):
            c = circuit.copy()
            getattr(c, readout)(*([indices] if readout == "save_amplitudes" else []))
            sim = AerSimulator(method=method)
            data = sim.run(transpile(c, sim, optimization_level=0)).result().data()
            got = (
                np.asarray(data["amplitudes"])
                if readout == "save_amplitudes"
                else np.asarray(data["statevector"])[indices]
            )
            errors[f"{method}/{readout}"] = float(np.max(np.abs(got - exact)))
    return {"instance": "n4_static first 3", "max_amplitude_error": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aer-only", action="store_true")
    parser.add_argument(
        "--summarise-only", action="store_true", help="rebuild summaries from the row checkpoint"
    )
    args = parser.parse_args()

    # Merge at write time, not read time, so an --aer-only run and a study run
    # cannot overwrite each other's keys.
    data: dict = {}
    if args.summarise_only:
        rows = json.loads(CHECKPOINT.read_text())
        data.update(rows=rows, summary=summarise(rows), x_mixer_qaoa=x_mixer_baseline())
    elif args.aer_only:
        data["aer_mps_check"] = aer_check()
    else:
        data.update(study())
        data["config"] = {
            "sizes": SIZES,
            "variants": VARIANTS,
            "depths": DEPTHS,
            "seeds": SEEDS,
            "shots": SHOTS,
            "maxiter": MAXITER,
            "restarts": RESTARTS,
            "optimizer": "COBYLA (scipy)",
            "initial_state": "uniform over all N! permutations",
            "backend": "subspace (exact); feasibility re-measured on the Qiskit circuit for N<=4",
            "versions": library_versions("qiskit", "scipy", "numpy"),
        }
    data = {**(json.loads(OUT.read_text()) if OUT.exists() else {}), **data}
    OUT.write_text(json.dumps(data, indent=1, default=float) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
