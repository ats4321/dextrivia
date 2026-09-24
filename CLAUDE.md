# Dextrivia — working notes for agents

## Goal

Sequence the removal of Iridium-33 collision debris, and turn that into an
**honest benchmark of classical vs quantum / quantum-inspired solvers**. The
deliverable is the comparison, not any one solver. A number is only worth
reporting if the snapshot, epoch, selection rule and cost model that produced it
are recorded alongside it.

## The problem

Given N catalogued objects and a delta-v cost for each transfer, find the
visiting order that minimises total delta-v.

**It is an OPEN PATH, not a TSP tour.** The servicer starts at the first object
of the sequence and stops at the last. There is no return leg to the start and
no depot. A sequence over N objects has exactly **N-1 legs**. Every solver, cost
model and QUBO formulation must assume this.

## Architecture

```
data/snapshots/*.json              immutable, timestamped TLE sets (committed)
        |  Snapshot.load / .select(n, rule, seed)
        v
src/dextrivia/propagation.py       SGP4 wrapper; mean semi-major axis at an epoch
        v
src/dextrivia/costs/*.py           CostModel -> C[i,j] or C[t,i,j] in km/s
        v
src/dextrivia/instances.py         build_instance(...) -> ProblemInstance (+provenance)
        v
src/dextrivia/solvers/*.py         Solver -> Solution
        v
src/dextrivia/cli.py               dextrivia fetch | build | solve
```

## Interfaces (`src/dextrivia/core.py`)

| Type | Shape |
|---|---|
| `DebrisObject` | `norad_id: int`, `name`, `line1`, `line2` |
| `ProblemInstance` | `norad_ids`, `names`, `epoch`, `costs`, `metadata`; `n`, `time_dependent`, `leg_costs(step)`, `leg_cost(step,i,j)`, `path_cost(seq)`, `save/load` |
| `CostModel` (Protocol) | `name`; `build(objects, epoch) -> np.ndarray` |
| `Solver` (Protocol) | `name`; `solve(instance, seed=None) -> Solution` |
| `Solution` | `sequence`, `total_dv_kms`, `runtime_s`, `solver_name`, `feasible`, `metadata` |

**Costs are either static or time-slotted:**

* `C[i, j]` — shape `(N, N)`. What the Hohmann model produces today.
* `C[t, i, j]` — shape `(N-1, N, N)`. `t` is the **position of the leg in the
  sequence** (leg `t` departs the `t`-th object), *not* wall-clock time. Chosen
  so the problem stays a finite DP: Held-Karp gets `t` for free from the popcount
  of its visited-set state. A cost model needing wall-clock time must map it onto
  these slots itself and record the mapping in `metadata`.

Solvers must handle **both** shapes — use `instance.leg_costs(step)`, never index
`instance.costs` directly.

A solver that cannot handle an instance (too large, backend missing) returns
`feasible=False` with a reason in `metadata`. It does not raise: the benchmark
needs to record misses.

## Conventions

* **Units:** km, km/s. Runtimes in seconds.
* **Time:** timezone-aware UTC `datetime` only. Naive datetimes are rejected at
  every boundary. No hardcoded calendar dates anywhere — the default epoch is the
  snapshot's median TLE epoch.
* **Identity: NORAD catalog IDs (int), everywhere.** 106 of the 107 objects in
  the snapshot are named `IRIDIUM 33 DEB`; names are display-only.
* **Earth constants:** WGS72 (`R = 6378.135 km`, `mu = 398600.8`), because that is
  what SGP4 uses internally. Do not mix in 6371 or WGS84.
* **Altitude:** mean semi-major axis (`Satrec.am`) at the epoch, never
  `|r| - R`. The instantaneous value swings ~285 km per orbit for the most
  eccentric object here.
* **Selection is explicit:** `n` + rule (`first` | `random`) + `seed`, recorded
  in `ProblemInstance.metadata`. Never silently slice a catalogue.
* **Snapshots are immutable.** `dextrivia fetch` always writes a new file and
  never overwrites. Benchmarks cite a snapshot filename.
* **Snapshot ordering comes from the filename**, `<group-stem>_<YYYYMMDD>[T<time>Z].json`,
  never from `fetched_utc` (which is null for the legacy snapshot) and never from
  alphabetical order. `dextrivia build` without `--snapshot` takes the newest
  snapshot *of the default group only*; another dataset needs an explicit flag.

## Known limitations (as of this foundation workspace)

1. **The benchmark is degenerate.** The Hohmann model derives every cost from one
   scalar per object, so the objects lie on a line, the optimum is just
   "visit in altitude order", and greedy already ties the exact optimum
   (0.1262 km/s on the 10-object instance). There is no headroom for any solver
   to demonstrate an advantage until the cost model improves. Documented and
   locked down by `tests/test_degeneracy.py` — when a real cost model lands,
   `test_greedy_ties_the_exact_optimum_on_the_real_10_object_instance` should
   start failing and be rewritten, not deleted.
2. No plane changes, no RAAN drift, no phasing, no finite burns, no J2 — the
   physics that actually dominates a debris-removal delta-v budget.
3. No time windows, propellant budget, or vehicle constraints.
4. No conjunction/collision-risk weighting.
5. The committed snapshot's `fetched_utc` is `null` — the pre-package fetch
   script never recorded a download time, and inventing one would be worse than
   admitting the gap (`fetched_utc_note` says so in the file). Snapshots fetched
   with `dextrivia fetch` carry a real timestamp. Anything reading
   `Snapshot.fetched_utc` must handle `None`.

## Directory ownership (parallel workspaces)

| Path | Owner |
|---|---|
| `src/dextrivia/core.py`, `instances.py`, `snapshots.py`, `propagation.py`, `cli.py`, `solvers/greedy.py`, `solvers/exact.py` | foundation (this workspace) |
| `src/dextrivia/costs/`, `data/instances/` | **physics workspace** |
| `src/dextrivia/qubo/`, `src/dextrivia/solvers/quantum*` | **QUBO workspace** |
| `tests/` | shared — add files, don't rewrite others' |

**Rule: nobody changes an interface in `core.py` without recording why in this
file**, under "Interface change log", in the same commit. Additive changes
(new optional field, new helper) still get a line. Both parallel workspaces are
written against these signatures.

Optional heavy dependencies go in `[project.optional-dependencies]` extras
(`quantum`, `physics`) — never in the core dependency list.

### Interface change log

* 2026-09-22 — initial `core.py`. Costs accept `(N,N)` or `(N-1,N,N)` from day
  one so the physics workspace can add time-dependent costs without a breaking
  change; slot index = position in sequence, so Held-Karp keeps working.
* 2026-09-23 — `ProblemInstance.save()` now normalises the path to end in
  `.npz` and returns the path it actually wrote. `np.savez` appends the suffix
  itself, so the old return value named a non-existent file whenever the caller
  passed anything else, and `load(save(p))` failed.
* 2026-09-23 — `Snapshot.fetched_utc` is now `datetime | None` (additive to
  callers that only read it, breaking for callers that assumed non-null). The
  committed legacy snapshot has no recorded fetch time; see limitation 5.
* 2026-09-23 — plane-aware cost models landed (physics workspace) with **no
  change to `core.py`**. Recorded here because it is the interesting case: the
  time-slotted `(N-1, N, N)` shape and the position-in-sequence slot index were
  enough, and `dextrivia.costs.selection.build_cluster_instance` assembles
  `ProblemInstance` directly rather than through `instances.build_instance`,
  whose `rule` argument is `first`/`random` only. If a `plane-cluster` rule
  should become a first-class `Snapshot.select` option, that is a foundation
  change and needs its own entry.

## Build & test

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .

uv run dextrivia build --n 10
uv run dextrivia solve --instance data/instances/iridium33_20260402_n10_first.npz --solver exact
```

## Physics

Owned by the physics workspace. Full write-up: **`docs/physics.md`**.

### What exists

| Module | What it is |
|---|---|
| `costs/hohmann.py` | altitude-only baseline. Unchanged, still the default for `dextrivia build`, still degenerate on purpose. |
| `costs/realistic.py` | `ImpulsiveCostModel` (Hohmann + optimally split plane change) and `EdelbaumCostModel` (low-thrust), plus `plane_angle`, `nodal_precession_deg_per_day`, `mean_elements`. |
| `costs/selection.py` | `plane-cluster` target selection + `build_cluster_instance`. |
| `scripts/build_instance_family.py` | regenerates the committed instance family and prints the greedy/exact gap table. |
| `scripts/plot_cloud.py` | `docs/figures/raan_altitude.png`. Needs the `physics` extra (matplotlib). |

### Facts that drive every design choice here

* Inclinations span 0.51°, RAANs span the full circle. The **median pairwise
  plane angle is 45.5°**, costing **5.8 km/s** impulsively — against 0.126 km/s
  for the entire 10-object mission under the coplanar model.
* Exchange rate at 700 km: **1° of plane ≈ 0.131 km/s, 100 km of altitude ≈
  0.053 km/s.** Both matter, which is what makes sequencing non-trivial.
* J2 nodal drift is **common-mode**: −0.405 to −0.505 °/day, so it mostly
  cancels in ΔΩ. The **differential** rate for a median pair is 0.016 °/day —
  **612 days to change their separation by 10°**. Waiting buys very little.
  Do not claim otherwise.

### Conventions this workspace adds

* **Wall-clock mapping: leg `k` departs at `epoch + k · Δ`**, `Δ =
  delta_per_leg_days` (transfer + rendezvous + capture + release). Recorded in
  metadata as `delta_per_leg_days` and `leg_departure_rule`. Slots stay
  position-in-sequence; this is the only place wall clock enters.
* `C[k,i,j]` is priced at the **nominal** departure time. A min-over-loiter-window
  variant exists (`window_days`) and is **off by default** — it buys ~0 km/s at
  86° and it grants the drift discount without the objective paying for the
  wait. See `docs/physics.md` §5.
* **`plane-cluster` selection is fixed before any solver runs.** Never pick
  instances by which solver wins on them.
* Edelbaum is only valid to θ ≈ 114.6°; above that the effective angle passes π
  and the formula inverts. It is clamped, and clamped values are ceilings, not
  quantities.
* Costs that are upper bounds (two-burn above ~60° of plane, `separate_burn_dv`)
  say so in their docstring.

### The committed instance family

`data/instances/iridium33_20260402_planecluster-v1_n{4,5,8,10,15,20}_{static,td30d}.npz`
— 12 files, committed (a `!`-negation in `.gitignore`), cited by
`tests/test_instance_family.py`. Δ = 30 days.

Headline, from `scripts/build_instance_family.py`:

* **Altitude order is no longer optimal anywhere** — 56–339% worse than the
  Held-Karp optimum. The degeneracy in "Known limitations" item 1 is gone for
  these instances (`tests/test_degeneracy.py` still passes because the default
  cost model is untouched, and should be rewritten, not deleted, if the default
  ever changes).
* **Greedy still ties exact on all six static instances.** Say this plainly; a
  plane-aware static matrix is still near-metric.
* **Time dependence is what breaks greedy**: +5.37% at N=8, +5.18% at N=15.
  That is the headroom available to a quantum / quantum-inspired solver.
* N=20 is past the Held-Karp limit and records a miss, by design.

## QUBO

Owned by the **QUBO workspace**. Full write-up in `docs/qubo.md`.

**No `core.py` change was needed.** The interfaces as merged carry this work
unaltered, so there is no entry in the interface change log above — costs
accepting `(N,N)` or `(N-1,N,N)` from day one is exactly what let the QUBO be
written once for both shapes.

### Files claimed

| Path | Note |
|---|---|
| `src/dextrivia/qubo/` | formulation, decoding, repair, penalty sweep, run bookkeeping |
| `src/dextrivia/solvers/quantum_annealing.py` | `sa-qubo`, dwave-samplers |
| `src/dextrivia/solvers/quantum_qaoa.py` | `qaoa`, Qiskit statevector |
| `src/dextrivia/solvers/ortools_routing.py` | `ortools` — **not a quantum file**, see below |
| `tests/test_qubo_formulation.py`, `tests/test_qubo_solvers.py` | added, nothing else in `tests/` touched |
| `docs/qubo.md` | new |

`ortools_routing.py` is the one name that does not match the
`solvers/quantum*` pattern this workspace was assigned. Calling a Google
constraint-programming solver `quantum_ortools.py` to satisfy a glob would put a
lie in the filename, and the benchmark's whole point is that the labels are
honest. It is a new file, so it collides with nobody — the ownership table above
names `greedy.py` and `exact.py` individually, not the directory. Claimed here
instead.

`src/dextrivia/solvers/__init__.py` gained three entries in `SOLVERS` plus their
imports, so the CLI can reach them. That file was unowned; this line is the
record.

### Rules for anyone touching this code

* **Never index `instance.costs`.** Use `instance.leg_costs(p)`. The QUBO's
  position index `p` and the instance's slot index `t` are the same integer, so
  time-slotted costs need no extra machinery — but only if everything goes
  through the accessor.
* **Optional backends are imported inside `solve()`, never at module scope.**
  `dextrivia.solvers` must stay importable without the `quantum` extra, or the
  CLI breaks on a core install. Enforced by `test_no_backend_is_imported_at_module_scope`,
  which reads the source with `ast` rather than importing (an import-based check
  would pass on CI, where every extra is installed) and by the `core-only` CI job.
* **Raw and repaired results stay separate.** `repair()` cannot fail, so a
  repaired delta-v always exists; quoting it as a sampler result would make a
  sampler that never satisfied a single constraint look successful. See the
  `N=4` random-noise control in `docs/qubo.md` §6 for how badly this misleads at
  small `N`.
* **The penalty default is a measurement, not a preference.**
  `DEFAULT_PENALTY_SAFETY = 1.1` comes from the sweep table in `docs/qubo.md` §3.
  Raising it does not buy feasibility and does cost solution quality.

### Known limitations added by this workspace

6. **QAOA tops out at N=4.** The position encoding needs `N**2` qubits and a
   statevector is `2**(N**2)` amplitudes: N=5 is 512 MB, N=6 is 1 TB. The exact
   Held-Karp oracle reaches N=18. The quantum solver's ceiling is an order of
   magnitude below the classical oracle's on the same problem — that gap is a
   finding, not a footnote.
7. **No result at N<=4 distinguishes a solver from noise.** With 6 or 24
   possible sequences, best-of-shots plus repair finds the optimum from uniform
   random bits. Only the raw feasibility rate carries any signal at those sizes.
8. **OR-Tools cannot take time-slotted costs.** A routing arc-cost callback sees
   `(from, to)` only and has no idea how many legs have been flown. Such
   instances get `feasible=False` rather than a quietly wrong answer. When the
   physics workspace lands `C[t,i,j]`, the strong classical baseline above
   Held-Karp range disappears and something else will have to fill that role.
9. **Only the time-dependent instances rank anything.** Superseded the original
   "benchmark is degenerate" note when the physics workspace landed: on the
   plane-cluster family, greedy still ties exact on *every* static instance
   (N=4..15), so those tables still rank nothing. The `td30d` instances at N=8
   and N=15 are the first with real headroom (greedy +5.37% and +5.18%). Quote
   a static-instance result only as a sanity check, never as a comparison.
10. **The one win and the one collapse are both at `sa-qubo`.** On `n8_td30d`
   it reaches the exact optimum where greedy loses 5.37% -- the first result in
   this project where the QUBO route beats the classical heuristic on a
   non-degenerate instance. On `n15_td30d` (225 variables) it lands +41.6%,
   far worse than greedy. Two data points are not a scaling law; do not report
   the first without the second. The N=15 gap is a ceiling, not a budget
   problem: 20x the sampling effort takes it to +17.1%, a further 5x (458 s,
   against greedy's sub-millisecond) lands +19.6%. It plateaus at 17-20%,
   three to four times worse than greedy. More compute does not lift it.
11. **The strong classical baseline is missing exactly where it is needed.**
   `ortools` refuses both `td30d` instances (limitation 8), which are the only
   ones with headroom. So the N=15 comparison currently has no good classical
   bar above Held-Karp range at all.
