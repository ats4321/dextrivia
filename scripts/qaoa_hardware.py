"""Run ``qaoa-swap`` once on IBM Quantum hardware, next to the noiseless answer.

    uv sync --all-extras                    # or: uv sync --extra quantum --extra ibm
    uv run python scripts/qaoa_hardware.py

Setup (the token is NEVER printed, logged or written anywhere by this script):

1. Create an IBM Quantum Platform account and API key at
   https://quantum.cloud.ibm.com (the Open Plan's free monthly QPU minutes are
   plenty: each circuit here is 4096 shots).
2. Put it in the repo root's ``.env`` (gitignored; a Conductor workspace
   symlinks it, see conductor.json):

       IBM_QUANTUM_TOKEN=<your API key>
       IBM_QUANTUM_INSTANCE=<optional instance CRN; omit to let the service pick>

3. Run this script. It writes ``docs/data/qaoa_hardware.json``.

What it runs -- the BASIS-START variant, which answers a different question
from the uniform-start quality study and must not share a table with it:

* N=3, taken as the first 3 objects of the committed ``n4_static`` and
  ``n4_td30d`` instances (td: the first 2 leg slots). 9 qubits.
* reps=1 with fixed angles (see ``prepare``: from a basis state one layer has
  nothing to optimise, so this is a fidelity test of one full layer). Sent
  once, no optimiser loop on the device.
* Start in the identity order (object p at position p) via X gates.
* N=4 (16 qubits) is transpiled too, and submitted only if the estimated
  circuit fidelity -- the product of (1 - error) over every two-qubit gate and
  measurement in the transpiled circuit, from the backend's own calibration --
  is at least ``MIN_ESTIMATED_FIDELITY``. Below that the output is noise by the
  backend's own numbers, and spending QPU time to confirm it proves nothing.

Without a token it records ``status: skipped`` with the reason, and still
reports the transpiled cost against ``FakeFez`` (a snapshot of a Heron r2
device's calibration shipped with qiskit-ibm-runtime), so the depth question
is answered either way. Nothing here is a quantum-advantage claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dextrivia.core import ProblemInstance
from dextrivia.qubo import library_versions
from dextrivia.qubo.permutation_qaoa import PermutationSpace
from dextrivia.snapshots import default_snapshot_dir
from dextrivia.solvers.quantum_qaoa_swap import swap_qaoa_circuit

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "qaoa_hardware.json"
SHOTS = 4096
REPS = 1
MIN_ESTIMATED_FIDELITY = 0.01
TOKEN_VAR = "IBM_QUANTUM_TOKEN"
INSTANCE_VAR = "IBM_QUANTUM_INSTANCE"


def load_env(path: Path = ROOT / ".env") -> None:
    """Read IBM_QUANTUM_* from ``.env`` into the environment, without echoing.

    A dozen lines instead of a python-dotenv dependency. Existing environment
    variables win.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.startswith("IBM_QUANTUM_") and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("'\""))


def sub_instance(variant: str, n: int) -> ProblemInstance:
    """First ``n`` objects of the committed N=4 instance. Recorded in metadata."""
    name = f"iridium33_20260402_planecluster-v1_n4_{variant}.npz"
    base = ProblemInstance.load(default_snapshot_dir().parent / "instances" / name)
    costs = base.costs[: n - 1, :n, :n] if base.time_dependent else base.costs[:n, :n]
    return ProblemInstance(
        norad_ids=base.norad_ids[:n],
        names=base.names[:n],
        epoch=base.epoch,
        costs=np.array(costs),
        metadata={
            **base.metadata,
            "derived_from": name,
            "derivation": f"first {n} objects; time-slotted costs keep the first {n - 1} slots",
        },
    )


def prepare(instance: ProblemInstance) -> dict:
    """Fix the angles; return the measured circuit and the exact noiseless answer.

    At reps=1 from a basis state there is nothing to optimise. The phase
    separator is diagonal, so on a single basis state it is a global phase: the
    output distribution depends on beta alone. (A free optimisation of beta
    returns ~0 or ~pi -- stay put -- and the transpiler then deletes the
    near-identity rotations, leaving an empty circuit to test.) So this is a
    FIDELITY test of one full QAOA layer, not an optimisation test:

    * beta = pi/4 makes every partial swap a 50/50 superposition -- the mixer
      is maximally non-trivial;
    * gamma = pi / (mean path cost) is arbitrary but not small, so the device
      has to execute the phase-separator gates and get them right enough to
      leave only a global phase.
    """
    n = instance.n
    start = tuple(range(n))
    space = PermutationSpace(instance)
    psi0 = space.basis_state(start)
    meta = {"gammas": [math.pi / float(space.costs.mean())], "betas": [math.pi / 4]}
    probs = np.abs(space.evolve(meta["gammas"], meta["betas"], psi0)) ** 2
    optimum = float(space.costs.min())
    circuit, g, b = swap_qaoa_circuit(instance, REPS, initial_sequence=start)
    bound = circuit.assign_parameters(
        {**dict(zip(g, meta["gammas"], strict=True)), **dict(zip(b, meta["betas"], strict=True))}
    )
    bound.measure_all()
    return {
        "circuit": bound,
        "space": space,
        "noiseless": {
            "feasibility_rate": 1.0,
            "optimal_probability": float(probs[space.costs <= optimum + 1e-12].sum()),
            "mean_dv_kms": float(probs @ space.costs),
            "source": "exact subspace simulation of the same circuit and angles",
        },
        "gammas": meta["gammas"],
        "betas": meta["betas"],
        "angle_rule": "beta = pi/4, gamma = pi / mean path cost; fixed, not optimised",
        "distribution": probs,
        "start_state_dv_kms": float(space.costs[psi0.real > 0][0]),
        "initial_sequence": start,
        "optimum_kms": optimum,
        "random_mean_dv_kms": float(space.costs.mean()),
        "uniform_bitstring_feasibility": math.factorial(n) / 2 ** (n * n),
    }


def transpile_report(circuit, backend) -> tuple[object, dict]:
    from qiskit.transpiler import generate_preset_pass_manager

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    isa = pm.run(circuit)
    ops = dict(isa.count_ops())
    two_q = {name for name in ops if name in ("cz", "ecr", "cx")}
    fidelity = 1.0
    for item in isa.data:
        name = item.operation.name
        if name in two_q or name == "measure":
            qargs = tuple(isa.find_bit(q).index for q in item.qubits)
            props = backend.target[name].get(qargs)
            if props is not None and props.error is not None:
                fidelity *= 1.0 - props.error
    return isa, {
        "backend": backend.name,
        "transpiled_depth": isa.depth(),
        "two_qubit_depth": isa.depth(filter_function=lambda i: i.operation.num_qubits == 2),
        "two_qubit_gates": sum(ops[name] for name in two_q),
        "count_ops": ops,
        "estimated_fidelity": fidelity,
        "optimization_level": 3,
    }


def score(counts: dict[str, int], n: int, prep: dict) -> dict:
    """Hardware counts in the same terms as the noiseless answer, plus the total
    variation distance between the two distributions over all 2**(N**2) outcomes
    (infeasible outcomes have noiseless probability 0)."""
    space, optimum = prep["space"], prep["optimum_kms"]
    total = sum(counts.values())
    feasible, optimal, dv = 0, 0, 0.0
    empirical = np.zeros(space.dim)
    for key, count in counts.items():
        bits = np.array([int(c) for c in key.replace(" ", "")[::-1][: n * n]]).reshape(n, n)
        if (bits.sum(axis=0) == 1).all() and (bits.sum(axis=1) == 1).all():
            sequence = tuple(int(np.argmax(bits[:, p])) for p in range(n))
            k = int(np.flatnonzero((space.perms == sequence).all(axis=1))[0])
            cost = space.costs[k]
            empirical[k] += count
            feasible += count
            dv += count * cost
            optimal += count if cost <= optimum + 1e-12 else 0
    return {
        "shots": total,
        "feasibility_rate": feasible / total,
        "optimal_probability": optimal / total,
        "optimal_probability_given_feasible": optimal / feasible if feasible else None,
        "mean_feasible_dv_kms": dv / feasible if feasible else None,
        "tvd_to_noiseless": 0.5
        * (np.abs(empirical / total - prep["distribution"]).sum() + (total - feasible) / total),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    load_env()
    token = os.environ.get(TOKEN_VAR)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    record: dict = {
        "date_utc": now,
        "initial_state": "basis (identity order, X gates)",
        "reps": REPS,
        "shots": SHOTS,
        "min_estimated_fidelity_to_submit": MIN_ESTIMATED_FIDELITY,
        "versions": library_versions("qiskit", "qiskit-ibm-runtime", "numpy", "scipy"),
        "runs": [],
    }

    if token:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token,
            instance=os.environ.get(INSTANCE_VAR) or None,
        )
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=16)
        record["status"] = "ran"
    else:
        from qiskit_ibm_runtime.fake_provider import FakeFez

        backend = FakeFez()
        record["status"] = "skipped"
        record["reason"] = (
            f"no {TOKEN_VAR} in the environment or in {ROOT.name}/.env; "
            "transpiled cost is reported against FakeFez, nothing was submitted"
        )
    del token

    for n in (3, 4):
        for variant in ("static", "td30d"):
            instance = sub_instance(variant, n) if n == 3 else _full(variant)
            prep = prepare(instance)
            isa, report = transpile_report(prep["circuit"], backend)
            run = {
                "instance": f"n{n}_{variant}" + (" (first 3 of n4)" if n == 3 else ""),
                "n": n,
                "qubits": n * n,
                **{
                    k: prep[k]
                    for k in (
                        "gammas",
                        "betas",
                        "angle_rule",
                        "initial_sequence",
                        "start_state_dv_kms",
                        "optimum_kms",
                    )
                },
                "random_mean_dv_kms": prep["random_mean_dv_kms"],
                "uniform_bitstring_feasibility": prep["uniform_bitstring_feasibility"],
                "noiseless": prep["noiseless"],
                "noiseless_distribution": prep["distribution"].tolist(),
                "transpiled": report,
            }
            submit = record["status"] == "ran" and (
                n == 3 or report["estimated_fidelity"] >= MIN_ESTIMATED_FIDELITY
            )
            if submit:
                job = SamplerV2(mode=backend).run([isa], shots=SHOTS)
                counts = job.result()[0].data.meas.get_counts()
                run["job_id"] = job.job_id()
                run["hardware"] = score(counts, n, prep)
            elif record["status"] == "ran":
                run["hardware"] = None
                run["not_submitted"] = (
                    f"estimated fidelity {report['estimated_fidelity']:.2e} "
                    f"< {MIN_ESTIMATED_FIDELITY}"
                )
            record["runs"].append(run)
            print(
                f"{run['instance']:24s} depth={report['transpiled_depth']} "
                f"2q={report['two_qubit_gates']} est.fidelity={report['estimated_fidelity']:.2e} "
                f"hardware={'yes' if submit else 'no'}",
                flush=True,
            )

    OUT.write_text(json.dumps(record, indent=1, default=float) + "\n")
    print(f"status={record['status']}; wrote {OUT.relative_to(ROOT)}")


def _full(variant: str) -> ProblemInstance:
    name = f"iridium33_20260402_planecluster-v1_n4_{variant}.npz"
    return ProblemInstance.load(default_snapshot_dir().parent / "instances" / name)


if __name__ == "__main__":
    main()
