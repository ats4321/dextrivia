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
| `src/dextrivia/bench.py`, `scripts/plot_benchmark.py`, `scripts/check_bench_output.py`, `results/`, `docs/figures/bench_*`, `README.md`, `dextrivia-landing.html` | **benchmark workspace** |
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
* 2026-09-24 — classical baselines (`localsearch`, `sa-perm`, `cpsat`) landed
  with **no change to `core.py`**. Logged because it is the third workspace in
  a row to need nothing: `leg_costs(step)` covered a MIP, a permutation
  annealer and a local search without alteration. The one thing worth flagging
  for future solver authors is that the position-in-sequence slot index makes
  2-opt **O(N) rather than O(1)** under time-slotted costs — see the QUBO
  section below.
* 2026-09-27 — `qaoa-swap` (constraint-preserving QAOA, QUBO workspace) landed
  with **no change to `core.py`**. Everything it records — backend, initial
  state, whether feasibility was measured or structural, `runtime_kind` — goes
  in `Solution.metadata`. `qubo.formulation` gained an additive
  `objective_qubo()` (penalty-free H_obj); `build_qubo` is unchanged in
  behaviour and shares its objective loop.
* 2026-09-28 — propagation validity horizon and v2 family (physics workspace,
  which for this change also held `snapshots.py`, `instances.py`, `cli.py`
  and `data/`). **No change to `core.py`.** Every new fact rides in `metadata`.
  Additive changes to the other files:
  * `dextrivia build` gained `--select plane-cluster`, `--cost-model`,
    `--delta-days`, `--raan-window`, `--inc-window`, `--alt-band`,
    `--seed-norad` and `--family-version`. The defaults are unchanged:
    `first`, `hohmann-coplanar`, static. This supersedes the 2026-09-23
    note: `plane-cluster` is now a CLI rule, dispatched to
    `build_cluster_instance`. It is still **not** a `Snapshot.select` rule,
    because it needs propagated elements.
  * New metadata keys on every freshly built instance:
    `propagation_span_days`, `validity_horizon_days`,
    `validity_horizon_source` and `exceeds_validity_horizon`. Plane-cluster
    instances also carry `excluded_decayed`. Old files lack them; use
    `costs.validity.check_instance`.
  * `select_plane_cluster(..., times=...)` screens out objects SGP4 cannot
    propagate. The `ClusterSelection` dataclass gained an
    `excluded_decayed` field with a default.
  * A second snapshot, `iridium33_20260928.json`, is now the `dextrivia build`
    default. Every quoted CLI example pins `--snapshot`, and
    `tests/test_readme_example.py` runs the README's.
* 2026-10-05 — docs only, **no change to `core.py`**: the 2026-09-23 single-seed
  `sa-qubo` effort sweep on v1 `n15_td30d` (+17–20% at 20–100× budget), salvaged
  from unmerged PR #4 into `docs/qubo.md` §7 as a dated historical measurement.

* 2026-09-30 — headroom workspace: certified lower bounds, `ils`,
  `sa-perm-cold`, v3 `collision-pair` family, resumable bench. **No change to
  `core.py`.** Every new fact rides in `Solution.metadata` or instance
  metadata. Additive changes to files other workspaces own:
  * **`solvers/highs_mip.py` (new)**: `highs`, a time-indexed MIP whose LP is
    strengthened with max-flow subtour cuts on aggregate arcs, run through
    HiGHS (`scipy.optimize.milp`). `lower_bound_kms` is certified to HiGHS
    tolerance minus 1e-6 km/s. It is checked against Held-Karp on all 29
    committed N ≤ 15 instances. SciPy is imported inside `solve()`. The time
    limit is enforced inside each LP, and an unfinished LP is discarded, never
    used as a bound. `HIGHS_MAX_N = 60`.
  * **`solvers/cpsat.py` (QUBO workspace)**: localsearch warm start
    (`AddHint`) and a redundant `AddCircuit` over aggregate arcs, which moves
    the v2 `n30_static` bound gap from 50% to 0%. `lower_bound_kms` now
    subtracts `(N-1)·0.5/COST_SCALE`, because the old value bounded the
    *rounded* problem. The raw value is kept as `raw_bound_kms`.
    `CPSAT_MAX_N` goes from 40 to 50: at N=50 it proves static v3 using 2.8 GB,
    at N=60 it needs 4.3 GB and leaves a 20% gap (`docs/bounds.md`).
  * **`solvers/permutation.py` (QUBO workspace)**: `ils` (localsearch +
    double-bridge + batched full-repricing descent + restarts) and
    `sa-perm-cold` (`start_from_greedy=False`). `localsearch` and `sa-perm`
    are unchanged.
  * **`solvers/__init__.py`**: three `SOLVERS` entries; `highs` is added to
    `DETERMINISTIC`.
  * **`bench.py` (benchmark workspace)**:
    * `reference_kind` gains `proven`, meaning the best-known value meets a
      certified bound. `best-known` references carry their bound, the solver
      that produced it, and the gap.
    * A bound above a reference raises.
    * Rows are checkpointed to `runs.jsonl`, so a run resumes with the same
      `--out`, and the manifest lists the git state of each segment.
    * `cpsat` runs one seed above N=20.
    * **Bug fixed:** `certified_lower_bound_kms` read a key no solver writes,
      so it was empty in every row of the first canonical run.
  * **`costs/selection.py`, `cli.py`, `scripts/build_instance_family.py`
    (physics / foundation)**:
    * The `collision-pair` rule, `merge_snapshots`, and `dextrivia build
      --select collision-pair --pair-snapshot`.
    * `family_name` / `family_path` take an optional `rule`.
    * v3 is built diagnostics-only, with no solver.
    * `--check` no longer crashes on v1 files, which lack span metadata.
  * **`data/snapshots/cosmos2251_20260930.json` (new)**: fetched with
    `dextrivia fetch`. The v3 family lives in
    `data/instances/collisionpair-v3/`. Its sha256 values are pinned in
    `tests/test_collision_pair_family.py`, and it was committed before any
    solver ran on it.
  * **`scripts/plot_benchmark.py`**: the new solvers take their parent's hue,
    dashed and hollow, instead of a ninth hue. It adds `--prefix` and a
    per-instance gap/bound figure.

## Benchmark

Owned by the **benchmark workspace**. Full write-up: **`README.md`** (the
public-facing one) — this section is the ownership and interface record.

```bash
dextrivia bench                                    # everything, ~30 min
dextrivia bench --solvers greedy,exact --seeds 1   # a quick one
uv run python scripts/plot_benchmark.py results/<run>
```

### Files claimed

| Path | Note |
|---|---|
| `src/dextrivia/bench.py` | the harness: discovery, runs, references, summaries, manifest |
| `scripts/plot_benchmark.py` | every figure, from a results directory |
| `results/` | committed canonical run, cited by the README |
| `tests/test_bench.py` | added, nothing else in `tests/` touched |

### Changes to files this workspace does not own

* **`core.py`: none.** The harness reads costs only through
  `instance.leg_costs(step)` and records misses through the existing
  `feasible=False` contract. No interface change was needed, so there is no
  entry in the interface change log.
* **`cli.py` (foundation)** gained a fourth verb, `bench`, delegating to
  `dextrivia.bench.add_arguments` / `.main`. Additive; `fetch|build|solve` are
  untouched.
* **`solvers/__init__.py`** (unowned, same as the QUBO workspace's note) gained
  three `SOLVERS` entries and a new `DETERMINISTIC` frozenset. The benchmark
  runs deterministic solvers once rather than once per seed — three identical
  Held-Karp runs measure nothing and cost oracle time.
* **`qubo/formulation.py` (QUBO workspace)** — `summarize_samples` returns two
  additional keys, `mean_feasible_dv_kms` and `best_raw_sample_count`. Purely
  additive; every existing key and value is unchanged. They exist so the
  harness can ask whether a sampler beat chance (mean of the *distribution*,
  not of the luckiest shot) and how often it landed on its own best answer —
  without the oracle ever being imported into a solver. The harness combines
  them with the optimum it already knows.

### Rules this workspace adds

* **Every number in `README.md` comes from the committed run** under
  `results/`, cited by path. If a number cannot be traced to a CSV cell it does
  not go in.
* **`reference_kind` is always printed.** `exact` means Held-Karp ran;
  `best-known` means it could not and the gap is measured against the best
  result anything achieved. A best-known gap is not an optimality gap.
* **A configured time limit is not a runtime measurement.** `ortools` and
  `cpsat` burn their limits; `qaoa`'s wall clock is **classical statevector
  simulation time** and is labelled that way everywhere it appears.
* **The solvers are not this workspace's.** `cpsat`, `localsearch` and
  `sa-perm` came from the classical-baselines workspace and landed on `main`
  first. This workspace built a duplicate set and **deleted it on merge** --
  two implementations of one solver in one registry is indefensible, and the
  merged ones are good. `solvers/__init__.py` keeps one addition of ours, the
  `DETERMINISTIC` set, which the harness uses to run seed-independent solvers
  once instead of five times.
* **`bench` is the CLI harness; `scripts/benchmark_family.py` is the other
  workspace's.** They overlap and that is worth resolving, but not by one of
  them silently deleting the other. See the open question in the PR.

## Build & test

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .

uv run dextrivia build --snapshot data/snapshots/iridium33_20260402.json --n 10
uv run dextrivia solve --instance data/instances/iridium33_20260402_n10_first.npz --solver exact
```

## Physics

Owned by the physics workspace. Full write-up: **`docs/physics.md`**.

### What exists

| Module | What it is |
|---|---|
| `costs/hohmann.py` | altitude-only baseline. Unchanged, still the default for `dextrivia build`, still degenerate on purpose. |
| `costs/realistic.py` | `ImpulsiveCostModel` (Hohmann + optimally split plane change) and `EdelbaumCostModel` (low-thrust), plus `plane_angle`, `nodal_precession_deg_per_day`, `mean_elements`. |
| `costs/selection.py` | `plane-cluster` target selection + `build_cluster_instance`, decay screen applied before the seed is chosen. `family_name` / `family_path`. |
| `costs/validity.py` | `VALIDITY_HORIZON_DAYS`, `screen_decay`, `horizon_metadata`, `check_instance`, `compare_snapshots`, `PropagationHorizonWarning`. |
| `scripts/build_instance_family.py` | `--version v2` (default) or `v1`: regenerates a family, prints the CLI command for each file and the gap table. |
| `scripts/validate_propagation.py` | April snapshot propagated against the September one → `data/validation/*.json`, `docs/figures/propagation_validation.png`. |
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
* **v1 is frozen and partly past the validity horizon**: `n8/n10/n15/n20_td30d`
  propagate 183–545 days, against a 177-day horizon. They are kept
  byte-identical (sha256 pinned in `tests/test_propagation_validity.py`)
  because the README's canonical run cites them. Do not regenerate them in
  place.

### Propagation validity horizon (docs/physics.md §9)

* **`VALIDITY_HORIZON_DAYS = 177`, measured, not assumed.** It comes from
  propagating the April snapshot against the real September TLEs. **This is a
  single-horizon measurement (two snapshots), not an error-vs-time curve.** The
  TLE fit-noise floor is unmeasured, so some of the "error" may be fit noise.
  Say so wherever the number is quoted.
* **The rule, pre-registered:** the 90th-percentile in-cluster leg-cost error
  must stay ≤ 5% of the median in-cluster leg. Scale linearly down from the
  measured span and cap at it. Measured: 37 m/s against a 65 m/s tolerance, so
  the cap binds. A test recomputes it from the two committed snapshots.
* **Element error at ~177 days, p90:** RAAN 0.30°, inclination 0.011°, mean
  SMA 17.8 km. Per 30 days, if linear: 0.051°, 0.0019° and 3.0 km, i.e.
  ≲ 7 m/s of leg cost.
* **The error is drag, below ~650 km.** Pairs involving such an object have a
  p90 of 89 m/s, already over tolerance at the measured span (about 129 days
  when scaled). The horizon is a population statement; do not quote it as
  valid for every object.
* **What 30-day legs mean now:** the span is `(N−2)·Δ + TLE age`, so Δ = 30 d
  fits only up to N ≈ 7.
* **SGP4 is late about decay.** 35080 re-entered on 2026-07-10 (SATCAT); SGP4
  gave up on it 120 days later. Decayed objects are screened out at selection
  time and recorded in `excluded_decayed`, never allowed to crash
  `mean_elements`.
* **Every new instance records its span.** An instance past the horizon gets
  `exceeds_validity_horizon=True` and emits `PropagationHorizonWarning`.
  Don't silence the warning; either flag it or stay inside the horizon.

### The v2 instance family (docs/physics.md §10)

`data/instances/planecluster-v2/iridium33_20260928_planecluster-v2_n{N}_{static,td{Δ}d}.npz`
has 31 files. It lives in a **subdirectory on purpose**: `dextrivia bench` globs
`data/instances/*.npz` non-recursively, and its short labels (`n8_td30d`) would
collide with v1's.

* N ∈ {4, 5, 8, 10, 15, 20} use v1's window. N ∈ {25, 30, 40} use a 20° RAAN
  window, the smallest in {15, 20, 25, 30}° that holds 40 objects (chosen by
  count, never by Δv).
* Δ ∈ {3, 7, 14, 30} days, each kept only where the span fits the horizon.
  7 d is one full phasing cycle at a 50 km altitude difference (6.5 d).
* Every file can be rebuilt with `dextrivia build --select plane-cluster ...`;
  the script prints the exact command, and tests rebuild three of them.
* **Headline:** greedy misses the reference on 10 instances, by up to +4.66%.
  localsearch ties wherever an exact oracle exists (N ≤ 15), so small N still
  has no headroom. N = 25–40 exist so a later workspace can find out whether
  anything beats localsearch where no oracle can check.
* Low-altitude legs in `n20_td7d`, `n25_td7d` and `n40_td3d` are the
  least-validated numbers in v2.

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
| `src/dextrivia/solvers/quantum_qaoa_swap.py` | `qaoa-swap`, constraint-preserving mixer; Qiskit statevector or exact subspace backend |
| `src/dextrivia/solvers/ortools_routing.py` | `ortools` — **not a quantum file**, see below |
| `src/dextrivia/solvers/permutation.py` | `localsearch` + `sa-perm` — **classical controls**, see below |
| `src/dextrivia/solvers/cpsat.py` | `cpsat` — time-indexed MIP, also classical |
| `scripts/benchmark_family.py`, `scripts/penalty_study.py` | the multi-seed studies behind every number quoted here |
| `scripts/qaoa_mixer_study.py`, `scripts/qaoa_hardware.py` | the `qaoa-swap` quality study (+ Aer MPS check) and the IBM hardware run |
| `tests/test_qubo_formulation.py`, `tests/test_qubo_solvers.py`, `tests/test_classical_baselines.py`, `tests/test_qaoa_swap.py` | added, nothing else in `tests/` touched |
| `docs/qubo.md`, `docs/data/*.json` | new |

`permutation.py`, `cpsat.py` and `ortools_routing.py` are classical and do not
match the `solvers/quantum*` pattern this workspace was assigned. They are here
because a quantum benchmark without controls is not a benchmark — the whole
point of `sa-perm` is to sit next to `sa-qubo` and answer "is it the annealing
or the encoding?". All three are new files that collide with nobody; the
ownership table above names `greedy.py` and `exact.py` individually, not the
directory.

On the naming specifically: `ortools_routing.py` is the one name that does not
match the pattern and could have. Calling a Google
constraint-programming solver `quantum_ortools.py` to satisfy a glob would put a
lie in the filename, and the benchmark's whole point is that the labels are
honest. It is a new file, so it collides with nobody — the ownership table above
names `greedy.py` and `exact.py` individually, not the directory. Claimed here
instead.

`src/dextrivia/solvers/__init__.py` gained three entries in `SOLVERS` plus their
imports, so the CLI can reach them. That file was unowned; this line is the
record. It later gained a fourth, `qaoa-swap`. **Heads-up for the benchmark
workspace:** `dextrivia bench` with no `--solvers` runs every registered
solver, so `qaoa-swap` now joins a default run (seconds per run on the subspace
backend). `bench.py`'s runtime-label table does not know it; the solver records
`runtime_kind` ("classical subspace simulation time") in its own metadata. The
committed canonical run is unchanged until that workspace re-runs it.

Files outside the ownership table touched by the `qaoa-swap` work, all
additive: `.gitignore` (`.env`), `conductor.json` (setup symlinks the repo
root's `.env` into each workspace), `pyproject.toml` + `uv.lock` (new `ibm`
extra, `qiskit-ibm-runtime>=0.50`, used only by `scripts/qaoa_hardware.py`).

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
  `DEFAULT_PENALTY_SAFETY = 1.1` survived a 6-instance x 5-seed study
  (`scripts/penalty_study.py`) that found **no consistent winner**. Do not
  change it on the strength of one instance — an earlier spot check made 1.1
  look like the worst option by reading `n15_td30d` alone.
* **2-opt is O(N), not O(1), under time-slotted costs.** Reversing a segment
  does not just reverse those legs; it changes which leg *index* every later
  pair occupies, so the whole suffix re-prices. Or-opt shifts positions and has
  the same problem. Any move evaluation here re-costs the full path. The
  textbook O(1) delta is silently **wrong** on exactly the instances that
  matter.
* **`qaoa-swap` uniform-start and basis-start numbers never share a table.**
  The uniform start (an initial statevector; reps=0 = random permutation) is
  the quality study. The basis start (X gates) is the only thing hardware can
  run, and from a basis state the first phase separator is a global phase. They
  answer different questions.
* **Subspace-backend feasibility is structural, not measured.** The solver says
  so in `feasibility_measured`. Measured feasibility needs the statevector
  backend or a re-run of the synthesised circuit.
* **The subspace simulator's authority is one test.**
  `test_qiskit_circuit_matches_subspace_simulation_amplitude_by_amplitude`
  compares it with the synthesised Qiskit circuit at N=3 and 4. If the mixer
  schedule, the cost operator or the encoding changes, that test is what
  decides whether the subspace numbers still mean anything.
* **Never rank solvers by argmin without a tie rule.** The first penalty
  verdict claimed "factor 1.01 wins 4/6" purely because easy instances score
  +0.00% across the board and `min()` picks whatever sorts first. Ties must
  vote for nobody.

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
9. **The QUBO encoding is a net loss, and the "win" was retracted.** The
   control is `sa-perm`: the same simulated annealing applied to sequences
   instead of a penalty-encoded bit vector. Across all 12 family instances at
   5 seeds, **`sa-qubo` does not beat `sa-perm` on a single one.** It ties on
   the trivial ones and loses monotonically with N — on `td30d`: +1.07% at
   N=8, +5.18% at N=10, +22.33% at N=15, +105.77% at N=20, against +0.00%
   everywhere for `sa-perm`. The comparison is generous to `sa-qubo` twice
   over: `sa-perm` is interpreted Python against compiled C++, and `sa-qubo`'s
   reads x sweeps budget overran the matched 2 s wall clock at N=15 and N=20
   (4.5 s and 9.2 s). More time, still worse.
10. **An earlier single-seed claim was wrong and is withdrawn.** `sa-qubo` was
   reported at +0.00% on `n8_td30d` as "the first genuine win in this
   project". Over 5 seeds it averages **+1.07%**; the zero was one lucky draw.
   Never quote a stochastic solver from one run — the repo says so and it
   happened anyway.
11. **The family has no headroom left for anybody.** `localsearch` (greedy
   start, 2-opt + or-opt) matches the reference on **all 12 instances**,
   deterministically, in **1–6 ms**; `cpsat` proves optimality on 11 of 12.
   "Time dependence breaks greedy" is still true (+5.37% at N=8, +5.18% at
   N=15) but it measured greedy's weakness, not the problem's difficulty. Any
   further solver comparison on `planecluster-v1` can only measure how far
   something falls short of a 6 ms local search. **Harder instances are the
   prerequisite for the next comparison** — more objects, or constraints
   (propellant, time windows, conjunction risk) that 2-opt cannot trivially
   repair.
12. **`ortools` routing still refuses every time-dependent instance**, but this
   no longer leaves a hole: `cpsat` covers them, proves optimality to N=15 in
   2.2 s, and reports a lower bound when it cannot (N=20 left a gap after
   109 s, so that reference is the only unproven one in the family).
13. **The penalty default survived a real study and stays at 1.1.** Six
   instances, five seeds, eight factors: no consistent winner — only N=15 and
   N=20 discriminate and they pick different factors. Separately, the
   *provably sufficient* bound costs a factor of two to three in quality
   against the sub-threshold factor 0.5, which still samples 94–97% feasible.
   The guarantee is real and it is not free.
14. **`qaoa-swap` answers the within-feasible question at N=4, and the answer
   fades with N.** Uniform start, 5 seeds, depths 1–5
   (`docs/data/qaoa_mixer_study.json`). At N=4, measured feasibility is 1.000
   in every run, and it captures 0.68–0.90 (`n4_static`) and 0.57–0.74
   (`n4_td30d`) of the improvement over a random permutation, against 0.16 and
   0.14 for the X-mixer `qaoa`. At N=5 and N=8 the best depth captures only
   0.44–0.50, and P(optimal) at N=8 is <1%. Past p≈3 COBYLA exhausts its
   budget and the depth curve falls, which is optimiser failure, since depth
   p+1 contains depth p. Warm-starting from p−1 is untried.
15. **Reach past N=4 comes from the problem's size, not from quantum
   anything.** The constraint-preserving circuit lives in an N!-dimensional
   subspace, so it simulates exactly to N=9. Aer MPS is correct where it runs
   (TVD at the shot-noise floor), but took over 600 s at 49 qubits and p=3,
   and it refuses 64 qubits (N=8) outright. qiskit-aer 0.17.2's MPS
   `save_amplitudes` returns wrong amplitudes on this circuit; sample instead.
16. **No hardware result exists.** No IBM token was configured, so
   `docs/data/qaoa_hardware.json` records `status: skipped`. Against FakeFez,
   one layer is 368 two-qubit gates at N=3 (estimated fidelity 0.33) and 1212
   at N=4 (0.021). Instructions are in `docs/qubo.md` §10.
