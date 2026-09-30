# QUBO formulation of open-path debris sequencing

Owned by the QUBO workspace. Code: `src/dextrivia/qubo/formulation.py`,
`src/dextrivia/solvers/quantum_annealing.py`,
`src/dextrivia/solvers/quantum_qaoa.py`,
`src/dextrivia/solvers/quantum_qaoa_swap.py`,
`src/dextrivia/solvers/ortools_routing.py`.

This document records a formulation and a characterization. It is **not** a
quantum-advantage claim.

**The short version: the penalty-encoded QUBO is a net loss on this problem.**
The control for it is `sa-perm` — the same simulated annealing, applied to
sequences instead of a penalty-encoded bit vector. Across all twelve
`planecluster-v1` instances at five seeds each, `sa-perm` matches the reference
optimum everywhere and `sa-qubo` beats it nowhere, losing by up to 106% while
consuming *more* wall clock. A plain 2-opt/or-opt local search also matches the
reference on all twelve, deterministically, in 1–6 ms. See §7, which also
retracts an earlier single-seed claim that the QUBO route had won at N=8.

**`qaoa-swap` (§10)** drops the penalty entirely and uses Hadfield et al.'s
permutation-swap mixer, which cannot leave the feasible set. At N=4, every
measured shot of the real circuit is feasible, and it captures 57–90% of the
achievable improvement over a random permutation, against 14–16% for the
penalty `qaoa`. The margin shrinks at N=5 and N=8 (≤0.50), and past p≈3 the
optimiser, not the ansatz, is the limit. It is still a toy-size simulation. No
hardware was run (no token configured).

## 1. The problem, restated for a binary encoder

Given $N$ catalogued objects and a leg cost $C_p[i,j]$ in km/s, find the
visiting order minimising total delta-v. The mission is an **open path**: the
servicer starts at the first object of the sequence and stops at the last. A
sequence over $N$ objects has exactly $N-1$ legs. There is no return leg and no
depot, so this is *not* a TSP tour and the textbook TSP QUBO is the wrong
formulation — it would charge the mission for a leg it never flies.

## 2. Position encoding

One binary variable per (object, position) pair:

$$x_{i,p} \in \{0,1\}, \qquad x_{i,p}=1 \iff \text{object } i \text{ is visited } p\text{-th}$$

with $i, p \in \{0,\dots,N-1\}$, so **$N^2$ binary variables**. The flat index
used throughout the code is $u = iN + p$ (`PathQUBO.var(i, p)`).

A valid sequence is a permutation matrix: exactly one 1 per row and per column.

### Objective

$$H_{\text{obj}}(x) \;=\; \sum_{p=0}^{N-2}\; \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} C_p[i,j]\, x_{i,p}\, x_{j,p+1}$$

The outer sum runs to $N-2$, giving **$N-1$ terms** — the open path again. A
tour formulation would add a $p = N-1 \to 0$ wrap-around term; there is none here.

### Constraints

$$H_{\text{pen}}(x) \;=\; \underbrace{\sum_{i=0}^{N-1}\Bigl(1 - \sum_{p=0}^{N-1} x_{i,p}\Bigr)^{\!2}}_{\text{each object visited once}} \;+\; \underbrace{\sum_{p=0}^{N-1}\Bigl(1 - \sum_{i=0}^{N-1} x_{i,p}\Bigr)^{\!2}}_{\text{each position filled once}}$$

Both families are needed. Dropping either one admits assignments that are not
permutation matrices — visiting one object twice, or leaving a position empty.

### Full Hamiltonian

$$H(x) \;=\; H_{\text{obj}}(x) \;+\; A \cdot H_{\text{pen}}(x)$$

Expanding $(1-S)^2 = 1 - 2S + S^2$ moves the constant $2AN$ into an explicit
offset, the $-2AS$ terms onto the diagonal (legitimate because $x_u^2 = x_u$ for
binaries), and $S^2$ into the quadratic block. `PathQUBO` stores an upper
triangular $Q$ of shape $(N^2, N^2)$ plus that offset, with

$$H(x) = x^\top Q\, x + \text{offset}, \qquad \text{offset} = 2AN.$$

**On a feasible assignment $H_{\text{pen}} = 0$, so the energy is exactly the
path cost in km/s.** That is the property everything else is checked against
(`test_feasible_energy_is_the_path_cost_plus_the_known_offset`).

## 3. The penalty weight

### The bound actually used

$H_{\text{pen}}$ is a sum of squared integers, so it is $0$ on a permutation
matrix and $\ge 1$ on anything else. Leg costs are non-negative, so
$H_{\text{obj}} \ge 0$ everywhere. Therefore any infeasible assignment scores at
least $A$, while the best feasible one scores exactly the optimum $L^*$:

$$A > L^* \;\;\Longrightarrow\;\; \text{every infeasible assignment is worse than the optimum.}$$

$L^*$ is unknown up front, but it is bounded above by any feasible path, so the
greedy solver supplies a computable bound in $O(N^3)$:

$$A \;=\; \text{safety} \cdot \underbrace{L_{\text{greedy}}}_{\ge\, L^*}, \qquad \text{safety} > 1.$$

`default_penalty()` implements exactly this. The bound is verified by exhaustive
enumeration of all $2^{N^2}$ assignments at $N=3$ and $N=4$
(`test_every_infeasible_assignment_scores_above_the_best_feasible_one`), for
both static and time-slotted costs.

### Why not the Lucas bound

Lucas 2014, *Ising formulations of many NP problems*, §7.2, gives the TSP rule
$A > B \cdot \max_{ij} W_{ij}$. That is a statement about the cost of a single
variable flip. It does not by itself rule out an infeasible assignment that
dodges several expensive legs at once — and on a real instance here
$\max_{ij} C[i,j] \approx 0.05$ km/s while $L^* \approx 0.126$ km/s, so the
Lucas value is *below* the threshold that provably works. The global bound costs
one greedy run and is sound.

### Why `safety` is 1.1 and not 10

A bigger penalty is not a free safety margin. Leg costs on the committed
10-object instance span roughly 0.005–0.05 km/s while $A \approx 0.14$; raising
$A$ further compresses the entire objective into the numerical shadow of the
penalty terms, and the sampler loses the ability to tell a good path from a
mediocre one. Measured with `sweep_penalty` on the degenerate Hohmann N=10
instance, 500 reads × 1000 sweeps, seed 11:

| factor | $A$ | provably sufficient | raw feasibility | best delta-v | gap to optimum |
|---|---|---|---|---|---|
| 0.5 | 0.0631 | no | 0.68 | 0.126246 | +0.0% |
| 1.0 | 0.1262 | no | 1.00 | 0.132130 | +4.7% |
| **1.1** | **0.1389** | **yes** | **1.00** | **0.133822** | **+6.0%** |
| 1.5 | 0.1894 | yes | 1.00 | 0.160549 | +27.2% |
| 2.0 | 0.2525 | yes | 1.00 | 0.175991 | +39.4% |
| 4.0 | 0.5050 | yes | 1.00 | 0.181762 | +44.0% |
| 8.0 | 1.0100 | yes | 1.00 | 0.182771 | +44.8% |

Two things worth reading off this table:

1. **Large penalties buy nothing.** Feasibility is already 100% at the
   threshold; going eight times higher costs 45% in solution quality and gains
   no feasibility at all. The default `DEFAULT_PENALTY_SAFETY = 1.1` was chosen
   from this measurement, not from taste.
2. **The sub-threshold row found the optimum.** At factor 0.5 the penalty is
   *provably insufficient* — and it found the exact optimum anyway, at 68%
   feasibility. That is not a contradiction: the bound is worst-case, and this
   instance is degenerate (see §7). It is a reminder that "provably sufficient"
   and "works well" are different properties, which is why `sweep_penalty`
   reports `provably_sufficient` as a separate column rather than filtering.

### What a 6-instance study says about the default

The table above is one easy instance. `scripts/penalty_study.py` sweeps eight
factors across all six time-dependent instances at five seeds each; the data is
in `docs/data/penalty_study.json`. Two results.

**There is no consistent winner, so the default stays at 1.1.** Only two of the
six instances discriminate between the provably sufficient factors at all — on
N=4, 5, 8 and 10 they tie — and those two disagree:

| instance | 1.01 | 1.05 | **1.1** | 1.25 | 1.5 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| N=10 | +3.16 | +3.31 | +5.18 | +6.50 | +11.37 | +16.73 | +43.49 |
| N=15 | +32.02 | +32.21 | **+22.33** | +36.78 | +49.24 | +72.62 | +102.83 |
| N=20 | +81.64 | **+70.25** | +105.77 | +101.16 | +120.44 | +149.41 | +196.16 |

An earlier spot check on N=15 alone made 1.1 look like the worst of four
choices. It is the best of seven there. That is what tuning on one instance
buys you.

A methodological note, because the first version of this study got it wrong:
**ties must vote for nobody.** The initial run reported "factor 1.01 wins 4/6"
purely because the easy instances score +0.00% across the board and `min()`
awards the win to whichever factor sorts first. Four of those six wins were
instances that discriminated nothing.

**The provable bound is expensive.** The sub-threshold factor 0.5 — which voids
the guarantee of §3 — beats every provably sufficient factor on the instances
that are actually hard, while still sampling 94–97% raw feasible:

| instance | factor 0.5 (not sufficient) | best sufficient | raw feasibility at 0.5 |
|---|---|---|---|
| N=10 | +0.00% | +3.16% | 0.92 |
| N=15 | +6.99% | +22.33% | 0.97 |
| N=20 | +36.12% | +70.25% | 0.94 |

Being able to *prove* that no infeasible assignment can win costs a factor of
two to three in solution quality, and buys feasibility the sampler was already
achieving without it. The guarantee is worth having when you need a guarantee;
it is not free, and §3's derivation should be read with that price attached.

## 4. Time-slotted costs

$C_p$ above *is* `instance.leg_costs(p)`. No solver in this workspace indexes
`instance.costs` directly. For a static instance `leg_costs(p)` returns the same
$(N,N)$ matrix for every $p$; for a time-slotted $(N-1, N, N)$ instance it
returns a different one per leg.

The mapping is therefore trivially exact, and it is exact *because* the slot
index is defined as position-in-sequence rather than wall-clock time: the QUBO's
$p$ and the instance's $t$ are the same integer. The quadratic term coupling
position $p$ to position $p+1$ is precisely the leg that slot $p$ prices. No
extra variables, no discretisation, no change to the variable count or the
penalty argument — a time-dependent instance produces a QUBO of exactly the same
shape.

Every formulation test runs twice, once per cost shape.

## 5. Decoding, feasibility and repair

* `decode(x, n)` returns the sequence, or `None` if `x` is not a permutation matrix.
* `is_feasible(x, n)` is the permutation-matrix check.
* `repair(x, n)` forces any sample into a valid sequence: positions are filled in
  descending order of how strongly any object claims them, ties to the lowest
  index. Deterministic.

**Raw and repaired results are reported separately and must stay that way.**
`summarize_samples()` returns `feasibility_rate` and `best_raw_dv_kms`
(what the sampler produced unaided, `None` if it never produced a valid
permutation) alongside `best_repaired_dv_kms` and `best_was_raw_sample`. A
repaired result always exists — repair cannot fail — so quoting it as though it
were a raw sampler output would make a sampler that never once satisfied the
constraints look like it solved the problem.

## 6. Solvers and their limits

| solver | name | backend | ceiling |
|---|---|---|---|
| Simulated annealing on the QUBO | `sa-qubo` | `dwave-samplers` | $N \le 40$ by policy, a runtime wall not a correctness one |
| QAOA, statevector simulator | `qaoa` | `qiskit` ≥ 2.0 + `scipy` | $N \le 4$ — **hard**, see below |
| Constraint-preserving QAOA | `qaoa-swap` | `scipy` (+ `qiskit` for the statevector backend) | $N \le 4$ statevector, $N \le 9$ exact subspace — §10 |
| OR-Tools routing | `ortools` | `ortools` | no practical $N$ limit; **cannot do time-slotted costs** |
| 2-opt/or-opt local search | `localsearch` | none | no limit; optimal on all 12 family instances in 1–6 ms |
| Annealing over permutations | `sa-perm` | none | no limit; **the control for `sa-qubo`** |
| Time-indexed MIP | `cpsat` | `ortools` | $N \le 50$ (raised from 40, evidence in `docs/bounds.md`); localsearch warm start, reports a certified *lower bound* |
| Cut-strengthened time-indexed MIP | `highs` | `scipy` (HiGHS) | $N \le 60$; certified lower bound, see `docs/bounds.md` |

Oracles for measuring all three: Held-Karp (`exact`, $N \le 18$) and brute force
(`brute`, $N \le 8$). They are not competitors.

### The qubit wall

The position encoding needs $N^2$ qubits, and a dense statevector is
$2^{N^2}$ complex amplitudes at 16 bytes each:

| $N$ | qubits | statevector |
|---|---|---|
| 3 | 9 | 8 KB |
| 4 | 16 | 1 MB |
| 5 | 25 | 512 MB |
| 6 | 36 | 1 TB |

`QAOA_MAX_N = 4`. $N=5$ is 512 MB *before the optimiser evaluates anything*, and
each COBYLA iteration simulates the circuit again. $N=6$ is not a tuning
question. Meanwhile Held-Karp is exact to $N=18$ and OR-Tools does not care.

**That gap is the finding.** The quantum solver's ceiling sits an order of
magnitude below the exact classical oracle's, on the same problem, using the same
formulation. Anything quantum reported here is a measurement of a simulator at
toy sizes.

### What QAOA actually achieves at $N \le 4$

Measured at 4096 shots, 2 restarts, COBYLA `maxiter=200`, seed 5, on a random
static instance:

| $N$ | qubits | reps | raw feasibility | gap to optimum | runtime |
|---|---|---|---|---|---|
| 3 | 9 | 1 | 0.044 | +0.0% | 20 s |
| 3 | 9 | 2 | 0.039 | +0.0% | 1 s |
| 3 | 9 | 3 | 0.061 | +0.0% | 2 s |
| 4 | 16 | 1 | 0.021 | +0.0% | 7 s |
| 4 | 16 | 2 | 0.017 | +0.0% | 44 s |
| 4 | 16 | 3 | 0.048 | +0.0% | 81 s |

**The `+0.0%` column is not evidence of anything.** Here is the control —
uniform-random bitstrings at the same 4096-shot budget, no QAOA, no optimiser:

| $N$ | permutation fraction of $2^{N^2}$ | random raw feasibility | random best-of-shots gap |
|---|---|---|---|
| 3 | 0.0117 | 0.0103 | +0.0% (raw) |
| 4 | 0.00037 | 0.0000 | +0.0% (**repaired only** — random never produced one valid permutation) |

At these sizes there are 6 and 24 possible sequences. Best-of-4096-shots plus
repair recovers the optimum from pure noise, so any solver quoting a gap here is
reporting the shot budget, not the algorithm. This is precisely why
`summarize_samples` keeps `best_raw_dv_kms` and `best_repaired_dv_kms` apart, and
why the $N=4$ random row has to be labelled as repaired.

The one genuine signal is the **raw feasibility rate against the random
baseline**: QAOA concentrates amplitude on the feasible (permutation-matrix)
subspace by roughly 4x at $N=3$ (0.039–0.061 vs 0.010) and by a factor that is
formally infinite at $N=4$ (0.017–0.048 vs a measured 0.000, against a
0.00037 base rate). That is a real effect and a modest one. Note also that it is
not monotone in `reps` — depth 2 is worse than depth 1 at both sizes — which is
the expected behaviour of a shallow QAOA landscape riddled with local optima,
and the reason `restarts` and `restart_expectations` are recorded per run.

### Why not `qiskit-optimization`

It has not had a release since August 2025, and it pulls `qiskit-algorithms` in
behind it. The QUBO → Ising translation it would provide is a dozen lines
(`PathQUBO.to_ising`) which we own and test directly against `PathQUBO.energy`
over enumerated assignments. Two unmaintained dependencies is a worse trade than
twelve tested lines.

The implementation uses current Qiskit 2.x APIs throughout: V2 primitives
(`StatevectorEstimator`, `StatevectorSampler`), and the `qaoa_ansatz` **function**
rather than the `QAOAAnsatz` class, which Qiskit deprecated in 2.1 along with the
rest of the `NLocal` hierarchy. No `execute`, no `QuantumInstance` — both removed.

### Why not `dwave-neal`

`dwave-neal` was last released in 2022 and now only re-exports from
`dwave-samplers`. Same `SimulatedAnnealingSampler`, live package name.

### Why OR-Tools refuses time-slotted costs

An OR-Tools routing arc-cost callback is a function of `(from_node, to_node)`.
It has no access to how many legs have already been flown, so it **cannot**
express $C[t,i,j]$. Given such an instance the solver returns
`feasible=False` with that reason rather than silently optimising a different
problem. Static instances are scaled km/s → mm/s (int64 arc costs) for the
search, but the reported delta-v is recomputed from the unrounded instance, so
rounding costs search quality and never accuracy.

Open path, not tour, is expressed with a dummy depot joined to every object by a
zero-cost arc: a closed tour through that depot, with the depot deleted, is an
open path with a free start and a free end and $N-1$ real legs.

Note that OR-Tools' guided local search never converges on its own — it burns its
full time limit every run. **Its runtime is a configured knob, not a
measurement**, and comparing it to another solver's wall-clock is meaningless
unless the limit is quoted alongside.

## 7. What these numbers are worth

Re-measured across the whole `planecluster-v1` family, 5 seeds for every
stochastic solver, means with standard deviations. Reference is Held-Karp where
it reaches and CP-SAT above that. Full data in `docs/data/benchmark_family.json`,
regenerate with `scripts/benchmark_family.py`.

### Time-dependent (`td30d`) — the only instances with any headroom

| N | reference (km/s) | `greedy` | `localsearch` | `sa-perm` | `sa-qubo` |
|---|---|---|---|---|---|
| 4 | 0.5205 | +0.00% | +0.00% | +0.00% | +0.00% |
| 5 | 0.8314 | +0.00% | +0.00% | +0.00% | +0.00% |
| 8 | 1.1124 | +5.37% | +0.00% | +0.00% | +1.07% ±0.027 |
| 10 | 1.5165 | +0.00% | +0.00% | +0.00% | +5.18% ±0.069 |
| 15 | 2.1855 | +5.18% | +0.00% | +0.00% | +22.33% ±0.258 |
| 20 | 3.6723 † | +0.00% | +0.00% | +0.00% | +105.77% ±0.684 |

### Static

| N | reference (km/s) | `greedy` | `localsearch` | `sa-perm` | `sa-qubo` |
|---|---|---|---|---|---|
| 4–8 | 0.5946 / 0.8296 / 1.0824 | +0.00% | +0.00% | +0.00% | +0.00% |
| 10 | 1.3268 | +0.00% | +0.00% | +0.00% | +6.96% ±0.082 |
| 15 | 2.0151 | +0.00% | +0.00% | +0.00% | +44.12% ±0.320 |
| 20 | 2.4485 | +0.00% | +0.00% | +0.00% | +96.30% ±0.237 |

† CP-SAT's best after 109 s without proving optimality; its lower bound left a
gap. `localsearch`, `sa-perm` and CP-SAT independently land on the same value,
so it is very probably optimal — but "three methods agree" is not a proof and
the table says so. Every other reference row is proven optimal.

### Retraction

**The earlier claim that `n8_td30d` was "the first genuine win in this project"
was wrong, and is withdrawn.** It rested on a single seed. Over five seeds
`sa-qubo` averages **+1.07%** there, not +0.00%; the zero was one lucky draw.
Reporting a stochastic solver from one run is exactly the error this repository
exists to avoid, and it got into the documentation anyway.

### Does the QUBO encoding add anything? No.

That was the question the controls existed to answer, and the answer is clean
because `sa-perm` is the same algorithm as `sa-qubo` — simulated annealing,
comparable schedule — applied to sequences instead of a penalty-encoded bit
vector. The only difference is the encoding.

| N (td30d) | `sa-perm` | `sa-qubo` |
|---|---|---|
| 4 | +0.00% (2.0 s) | +0.00% (0.3 s) |
| 5 | +0.00% (2.0 s) | +0.00% (0.4 s) |
| 8 | +0.00% (2.0 s) | +1.07% (1.1 s) |
| 10 | +0.00% (2.0 s) | +5.18% (1.8 s) |
| 15 | +0.00% (2.0 s) | +22.33% (**4.5 s**) |
| 20 | +0.00% (2.0 s) | +105.77% (**9.2 s**) |

**`sa-qubo` does not beat `sa-perm` on a single instance at any size.** It ties
on the two trivial ones and loses everywhere else, and the loss grows
monotonically with N.

The comparison is also *generous* to `sa-qubo` in two ways worth stating, since
both cut against the conclusion:

1. `sa-perm` is interpreted Python; `sa-qubo` is compiled C++ inside
   `dwave-samplers`. A wall-clock match handicaps the permutation solver.
2. `sa-qubo`'s budget is reads x sweeps, not wall clock, so at N=15 and N=20 it
   actually *overran* the 2 s budget `sa-perm` was held to — 4.5 s and 9.2 s.
   It got more time and still lost by 22% and 106%.

So the penalty-encoded QUBO is not a neutral reformulation that a quantum
sampler might later exploit. On this problem it is a **net loss**: it takes an
objective that plain annealing optimises exactly and makes it one that the same
annealer, given more time, cannot.

### And the classical bar is higher than anything here reaches

`localsearch` — greedy start, 2-opt and or-opt to a local optimum — matches the
reference on **all twelve instances**, deterministically, in **1–6 ms**. CP-SAT
proves optimality on eleven of twelve.

That reframes the earlier "time dependence breaks greedy" finding. It does
break greedy (+5.37% at N=8, +5.18% at N=15) — but greedy being weak is not the
same as the problem being hard, and the difference was never measured until
now. Two-opt fixes it instantly. **The `planecluster-v1` family has no headroom
left for any solver**, classical or quantum; a benchmark on it can now only
measure how far a solver falls short of something a 6 ms local search already
achieves.

Harder instances are the prerequisite for any further comparison here — see the
limitations in `CLAUDE.md`.

## 8. Run records

Every solver returns a `Solution` whose `metadata` carries, at minimum: best
delta-v (`total_dv_kms` on the solution), `feasibility_rate`, time-to-solution
(`runtime_s`), `seed`, and `versions` from `library_versions()` — and per solver
the reads/shots/layers that define the run:

* `sa-qubo` — `num_reads`, `num_sweeps`, `num_variables`, `penalty`
* `qaoa` — `qubits`, `reps`, `shots`, `restarts`, `maxiter`,
  `objective_evaluations`, `final_expectation`, `restart_expectations`,
  `hamiltonian_terms`, `statevector_bytes`
* `ortools` — `time_limit_s`, `first_solution_strategy`, `metaheuristic`,
  `cost_scale`, `scaled_objective`

`versions` is not decoration. "Simulated annealing" is not a fixed algorithm; it
is whatever version of `dwave-samplers` was installed. A delta-v without it is
not reproducible. Missing packages are recorded as `"not installed"` rather than
omitted, so a run record never silently loses a row.

## 9. Scaling summary

| | variables / qubits | ceiling | why |
|---|---|---|---|
| QUBO construction | $N^2$ | dense $Q$ is $(N^2)^2$ floats: 6 MB at $N=30$ | memory |
| `qaoa` | $N^2$ qubits | $N=4$ | $2^{N^2}$ statevector |
| `qaoa-swap` | $N^2$ qubits, $N!$ reachable | $N=4$ / $N=9$ | statevector / exact feasible-subspace simulation (§10) |
| `sa-qubo` | $N^2$ spins | $N=40$ by policy | runtime |
| `exact` (oracle) | — | $N=18$ | $2^N N^2$ |
| `brute` (oracle) | — | $N=8$ | $N!$ |
| `ortools` | — | none reached | time-limited heuristic, static costs only |
| `cpsat` | $\approx N^3$ | $N=50$ by policy | see `docs/bounds.md` for N=40-60 |
| `localsearch` | — | none reached | full 2-opt + or-opt neighbourhood, $O(N^3)$ per pass |
| `sa-perm` | — | none reached | wall-clock budgeted |

Every ceiling is reported as `feasible=False` with a reason in `metadata`, never
as an exception — a benchmark has to record a miss, not crash on it.

## 10. `qaoa-swap`: a mixer that never leaves the feasible set

Code: `src/dextrivia/qubo/permutation_qaoa.py` (mixer schedule, subspace
simulator), `src/dextrivia/solvers/quantum_qaoa_swap.py` (the Qiskit circuit and
the solver), `tests/test_qaoa_swap.py`. Data: `docs/data/qaoa_mixer_study.json`,
`docs/data/qaoa_hardware.json`. Regenerate with `scripts/qaoa_mixer_study.py`
and `scripts/qaoa_hardware.py`.

§6 found that the penalty-QUBO `qaoa` finds *permutations* but not *good*
permutations. The textbook remedy is to stop spending the circuit on the
constraints at all: start inside the feasible subspace and use a mixer that
cannot leave it (Hadfield, Wang, O'Gorman, Rieffel, Venturelli, Biswas, *From
the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator
Ansatz*, Algorithms **12**(2):34, 2019, doi:10.3390/a12020034, arXiv:1709.03489v2).
Their orderings construction is **§5.1** (TSP).

### Why not an XY mixer

An XY (ring or complete) mixer on a one-hot register preserves that register's
Hamming weight. The position encoding has *two* families of one-hot constraints
— one object per position and one position per object — and an XY mixer on each
position's register preserves only the first. It will happily put one object in
two positions. XY mixers are the right tool when each variable has a single
one-hot register and nothing else, as in graph colouring (Wang, Rubin, Dominy,
Rieffel, *XY-mixers: analytical and numerical results for QAOA*, Phys. Rev. A
**101**, 012320, 2020, doi:10.1103/PhysRevA.101.012320). For permutations
Hadfield et al. use the ordering-swap mixer below, which is 4-local.

### The ansatz

Qubit $(u,p)$ is $x_{u,p}$, "object $u$ is visited $p$-th", index $uN+p$ as in
§2. With $S^+ = |1\rangle\langle 0| = (X-iY)/2$ and $S^- = (S^+)^\dagger$, the
**adjacent ordering-swap partial mixer** (Hadfield eqs. 46–47) for positions
$(i,i+1)$ and objects $\{u,v\}$ is

$$H_{i,\{u,v\}} \;=\; S^+_{u,i+1}\,S^+_{v,i}\,S^-_{u,i}\,S^-_{v,i+1} \;+\; \text{h.c.}$$

It swaps $u$ and $v$ between positions $i$ and $i+1$ if and only if they occupy
them, and annihilates every other basis state. One layer of the circuit is

$$U(\gamma,\beta) \;=\; \underbrace{\prod_{\text{odd } i}\;\prod_{\{u,v\}} e^{-i\beta H_{i,\{u,v\}}}\;\prod_{\text{even } i}\;\prod_{\{u,v\}} e^{-i\beta H_{i,\{u,v\}}}}_{\text{mixer}} \;\; \underbrace{e^{-i\gamma H_{\text{obj}}}}_{\text{phase separator}}$$

with $i \in \{0,\dots,N-2\}$ (open path: no wrap-around "last" part as in the
TSP tour). $H_{\text{obj}}$ is the objective of §2 alone, **with no penalty**:
on permutation matrices $H_{\text{pen}} \equiv 0$, so it would only add a
constant. Its leg-$p$ coupling is `instance.leg_costs(p)`, so time-slotted
costs work unchanged, exactly as in §4. Within one position the object pairs
are applied in Hadfield's colour-partition order (a proper edge colouring of
$K_N$, so each colour is a set of partial mixers on disjoint qubits that can run
as one parallel layer). Hadfield order the parts colour-major; we order them
parity-major. On the feasible subspace the two are the same operator — see the
proof below.

Initial states — **two, answering different questions, never mixed in a
table**:

* **uniform** — equal superposition over all $N!$ permutation matrices. At
  zero layers this *is* the uniform-random-permutation baseline, so any gain
  over random is attributable to the layers. It is supplied as an initial
  statevector, not synthesised as a gate: a state-preparation circuit over
  $N^2$ qubits is not something hardware can run. The quality study uses this.
* **basis** — one permutation (the identity order), prepared with $X$ gates.
  Hardware-runnable. The hardware run uses this.

### Proof: the ansatz preserves feasibility, and may be regrouped

Let $F$ be the span of the $N!$ permutation-matrix basis states. Four claims,
each checked by a test.

**(a) Each partial mixer preserves $F$.** On a basis state $|\sigma\rangle$ with
$\sigma$ a permutation, $H_{i,\{u,v\}}$ gives $|\sigma'\rangle$ — $\sigma$ with
positions $i,i+1$ exchanged — when $\{\sigma_i,\sigma_{i+1}\} = \{u,v\}$, and
$0$ otherwise. So $H_{i,\{u,v\}}F \subseteq F$. It is Hermitian, so it also
preserves $F^\perp$, and therefore so does every function of it, including
$e^{-i\beta H_{i,\{u,v\}}}$. The phase separator is diagonal in the
computational basis and preserves $F$ trivially. Hence the whole circuit maps
$F$ to $F$, for any angles and any ordering of the factors. Leakage out of $F$
is exactly zero in exact arithmetic.

**(b) The Pauli strings inside one partial mixer commute, so each factor is
exponentiated exactly.** Expanding the four ladder operators gives
$2^4 = 16$ strings, each with $X$ or $Y$ on all four qubits and coefficient
$\pm\tfrac{1}{16}\,i^{\#Y}$. Adding the Hermitian conjugate cancels the strings
with an odd number of $Y$s (their coefficients are imaginary) and doubles the
rest: **8 strings, coefficient $\pm\tfrac18$, each with an even number of $Y$s**.
Two single-qubit Paulis from $\{X,Y\}$ anticommute iff they differ, so two such
strings anticommute on exactly as many qubits as they differ on. Two strings with
the same $Y$-parity differ on an even number of qubits. The signs cancel and the
strings commute. Qiskit's default `PauliEvolutionGate` synthesis (Lie–Trotter,
one step) is therefore **exact** for each partial mixer, not a Trotter
approximation.
(`test_every_pauli_string_in_a_swap_term_commutes`.)

**(c) Partial mixers at the same position commute on $F$, although not on the
full space.** $H_{i,\{u,v\}}$ and $H_{i,\{u,w\}}$ share qubits $(u,i)$ and
$(u,i+1)$, and on the full $2^{N^2}$-dimensional space they do not commute in
general. So the ordering within a position does matter off $F$. Restricted to
$F$, however: $H_{i,\{u,v\}}|\sigma\rangle \neq 0$ requires
$\{\sigma_i,\sigma_{i+1}\} = \{u,v\}$, and the output $|\sigma'\rangle$ has the
same pair at those positions. In a permutation, *exactly one* unordered pair
occupies positions $(i,i+1)$. So for $\{u,v\} \neq \{u',v'\}$,

$$H_{i,\{u',v'\}}\,H_{i,\{u,v\}}\big|_F = 0 = H_{i,\{u,v\}}\,H_{i,\{u',v'\}}\big|_F .$$

The restricted operators commute, and because all their pairwise products
vanish,

$$\prod_{\{u,v\}} e^{-i\beta H_{i,\{u,v\}}}\Big|_F \;=\; e^{-i\beta H_i}\big|_F, \qquad H_i = \sum_{\{u,v\}} H_{i,\{u,v\}},$$

in any order. Here $H_i$ is Hadfield's *value-independent* adjacent swap (eq. 49).
By (a), the circuit acting on a state in $F$ equals the product of the restricted
factors, so the within-position ordering is irrelevant *for this ansatz*. The
disproof half matters too: a circuit that starts outside $F$, or a hardware
error that knocks the state out of $F$, sees an ordering-dependent operator.
(`test_partial_mixers_at_one_position_multiply_to_zero_on_the_feasible_subspace`
builds the restricted matrices and checks every pairwise product is zero.)

**(d) On $F$ the mixer is closed-form.** $H_i|_F$ is a perfect matching of
permutations (every permutation has exactly one partner, with positions $i,i+1$
swapped), so $H_i^2|_F = I$ and
$e^{-i\beta H_i}|_F = \cos\beta\, I - i\sin\beta\, H_i|_F$. $H_i$ and $H_{i'}$
act on disjoint qubits when $|i-i'|>1$, so all even positions commute with each
other, as do all odd ones. The mixer on $F$ is exactly
$e^{-i\beta H_{\text{odd}}}\,e^{-i\beta H_{\text{even}}}$, two blocks per layer.
Since adjacent transpositions generate $S_N$, repeated layers reach every
permutation.

**The final word is numerical**:
`test_qiskit_circuit_matches_subspace_simulation_amplitude_by_amplitude`
synthesises the Qiskit circuit to `cx/rz/sx/x`, evolves a dense statevector, and
compares it with the subspace simulator amplitude by amplitude at $N=3$ and
$N=4$, for static and time-slotted costs and for uniform and basis starts.
Measured agreement: **max |Δamplitude| ≤ 1.6e-10**, **leakage out of $F$ ≤
9e-14** (reps=2; 1,872 CNOTs at $N=4$). Synthesising first matters.
`Statevector.evolve` on an unsynthesised `PauliEvolutionGate` exponentiates the
whole operator as a matrix, which would check the mathematics but not the gate
sequence a device runs.

### The subspace simulator, and why it answers "can it go past N=4"

Claims (a)–(d) mean the circuit, started in $F$, lives in an $N!$-dimensional
space. There the phase separator is `psi *= exp(-1j*gamma*cost)` and each
position's mixer is `psi = cos(b)*psi - 1j*sin(b)*psi[partner_i]`. That is an
**exact** simulation of the same circuit, verified above, and not a
quantum-inspired approximation:

| $N$ | qubits | dense statevector | feasible subspace |
|---|---|---|---|
| 4 | 16 | 1 MB | 24 amplitudes |
| 5 | 25 | 512 MB | 120 |
| 8 | 64 | $2.9\times10^{20}$ bytes | 40,320 |
| 9 | 81 | — | 362,880 (~20 MB with index arrays) |

`SUBSPACE_MAX_N = 9`. The statevector backend keeps the qubit wall at
`STATEVECTOR_MAX_N = 4`. Past either limit the solver returns `feasible=False`
with the size in the reason. On the subspace backend `feasibility_rate` is 1.0
**by construction** (`feasibility_measured: false` in metadata); only the
statevector backend, and the study's re-measurement below, measure it.

Note what this says about the quantum solver. The reason the constraint-preserving
ansatz can be simulated to $N=9$ is that its reachable state space is only $N!$.
The feasible subspace is exactly as small classically as it is quantumly, so an
exact classical simulation of this circuit costs about as much as enumerating
every sequence, which Held-Karp already beats.

### Results, uniform start: does it optimise *within* the feasible set?

**Yes, at N=4, clearly. Less so as N grows. Depth past ~3 stops helping
because the optimiser does, not because the ansatz is exhausted.**

Protocol (`scripts/qaoa_mixer_study.py`, data `docs/data/qaoa_mixer_study.json`):
uniform start, depths 1–5, seeds 1–5, 4096 shots, COBYLA with 3 random restarts
and `maxiter=1000`. Optimisation runs on the exact subspace backend. At $N=4$
the optimised angles are then re-run on the **synthesised Qiskit circuit** and
feasibility is **measured** from 4096 sampled bitstrings. Above $N=4$
feasibility is 1.0 by construction (c). The optimum is Held-Karp, outside the
solver.

**captured** = (random mean − QAOA mean) / (random mean − optimum): the share of
the achievable improvement over a uniformly random permutation. 0 is chance, 1
is "always optimal". The X-mixer row uses the same formula on the committed
canonical run (`results/canonical/results.csv`, 5 seeds, reps=2). Its P(optimal)
is conditioned on feasibility, because every `qaoa-swap` sample is feasible.
Means ± standard deviation over 5 seeds.

| instance | solver | p | raw feasibility | P(optimal) | uniform over feasible | mean Δv (km/s) | random perm. mean | captured |
|---|---|---|---|---|---|---|---|---|
| `n4_static` | `qaoa` (X mixer, penalty) | 2 | 0.034 (m) | 0.133 ± 0.037 *given feasible* | 0.0833 | 0.9586 | 1.0261 | **0.16 ± 0.07** |
| `n4_static` | `qaoa-swap` | 1 | 1.000 (m) | 0.328 ± 0.000 | 0.0833 | 0.7340 ± 0.0000 | 1.0261 | **0.68 ± 0.00** |
| `n4_static` | `qaoa-swap` | 2 | 1.000 (m) | 0.455 ± 0.019 | 0.0833 | 0.6937 ± 0.0105 | 1.0261 | **0.77 ± 0.02** |
| `n4_static` | `qaoa-swap` | 3 | 1.000 (m) | 0.536 ± 0.069 | 0.0833 | 0.6763 ± 0.0096 | 1.0261 | **0.81 ± 0.02** |
| `n4_static` | `qaoa-swap` | 4 | 1.000 (m) | 0.707 ± 0.057 | 0.0833 | 0.6463 ± 0.0086 | 1.0261 | **0.88 ± 0.02** |
| `n4_static` | `qaoa-swap` | 5 | 1.000 (m) | 0.762 ± 0.078 | 0.0833 | 0.6367 ± 0.0153 | 1.0261 | **0.90 ± 0.04** |
| `n4_td30d` | `qaoa` (X mixer, penalty) | 2 | 0.027 (m) | 0.053 ± 0.024 *given feasible* | 0.0417 | 0.9103 | 0.9735 | **0.14 ± 0.08** |
| `n4_td30d` | `qaoa-swap` | 1 | 1.000 (m) | 0.121 ± 0.040 | 0.0417 | 0.7140 ± 0.0987 | 0.9735 | **0.57 ± 0.22** |
| `n4_td30d` | `qaoa-swap` | 2 | 1.000 (m) | 0.159 ± 0.035 | 0.0417 | 0.6521 ± 0.0339 | 0.9735 | **0.71 ± 0.08** |
| `n4_td30d` | `qaoa-swap` | 3 | 1.000 (m) | 0.175 ± 0.068 | 0.0417 | 0.6446 ± 0.0261 | 0.9735 | **0.73 ± 0.06** |
| `n4_td30d` | `qaoa-swap` | 4 | 1.000 (m) | 0.152 ± 0.081 | 0.0417 | 0.6400 ± 0.0334 | 0.9735 | **0.74 ± 0.07** |
| `n4_td30d` | `qaoa-swap` | 5 | 1.000 (m) | 0.236 ± 0.035 | 0.0417 | 0.6399 ± 0.0298 | 0.9735 | **0.74 ± 0.07** |

(m) = measured from sampled bitstrings. `qaoa-swap` P(optimal) and mean Δv are
exact values of the output distribution. The 4096-shot empirical P(optimal)
agrees with them to within 0.006 in every row of this table, and within 0.013
in every individual run, which is shot noise (both are in the JSON).

Past the old qubit wall, subspace backend:

| instance | p=1 | p=2 | p=3 | p=4 | p=5 | uniform P(opt) | best P(opt) seen |
|---|---|---|---|---|---|---|---|
| `n5_static` captured | 0.30 ± 0.10 | 0.39 ± 0.15 | 0.48 ± 0.14 | **0.50 ± 0.16** | 0.41 ± 0.17 | 0.0167 | 0.185 (p=4) |
| `n5_td30d` captured | 0.29 ± 0.08 | **0.47 ± 0.05** | 0.44 ± 0.09 | 0.47 ± 0.15 | 0.47 ± 0.11 | 0.0083 | 0.071 (p=3) |
| `n8_static` captured | 0.20 ± 0.10 | 0.39 ± 0.00 | 0.36 ± 0.17 | **0.44 ± 0.14** | 0.37 ± 0.13 | 0.00005 | 0.009 (p=5) |
| `n8_td30d` captured | 0.24 ± 0.00 | 0.31 ± 0.12 | **0.46 ± 0.00** | 0.37 ± 0.19 | 0.37 ± 0.19 | 0.00002 | 0.009 (p=5) |

Read honestly:

* **The within-feasible question has a clear answer at N=4.** The X-mixer
  `qaoa` captured 14–16% of the available improvement. `qaoa-swap` captures
  68–90% on `n4_static` and 57–74% on `n4_td30d`. Mean Δv is 38% and 34% below
  a random permutation at p=5, against 6.6% and 6.5% for `qaoa`. P(optimal)
  reaches 9× the uniform-over-feasible rate on `n4_static` and 5.7× on
  `n4_td30d`, against 1.6× and 1.3× for the X mixer's feasible samples. Every
  measured shot of the real circuit was feasible.
* **Time dependence costs about 15 points of captured improvement** at N=4 and
  flattens the depth curve. It has one optimal sequence instead of two
  (reversal symmetry is broken), so there is half the target to hit.
* **It fades with N.** Captured falls to ~0.45–0.50 at N=5 and ~0.44–0.46 at
  N=8, and P(optimal) at N=8 is <1%. That is still 180–350× the uniform
  rate, but in absolute terms most shots are not optimal.
* **Depth is optimiser-limited, not ansatz-limited.** A depth-$p+1$ circuit
  contains the depth-$p$ one (set the new angles to zero), so the true optimum
  of captured cannot fall with $p$. When the table falls (N=5 and N=8 at p=5),
  or the seed spread widens, COBYLA is failing, not the ansatz. The run records
  confirm it: at $p \ge 3$ nearly every run exhausted all 3 × 1000 evaluations.
  A warm start from the $p-1$ optimum (INTERP-style) is the obvious next step.
  It was not tried here.
* **Best-of-shots is still not evidence**, as §6 warned. The best sampled
  sequence was optimal in 100/100 runs at N=4–5 and 45/50 at N=8. Local search
  finds the same optimum in 1–6 ms, and at N=8 there are only 40,320 sequences.

This is a toy-size simulation result about an ansatz. It is not a speed or
quality advantage over anything classical. Every classical solver in §7 is
optimal on all of these instances, and `localsearch` does it in milliseconds.

### Beyond N=4: the subspace simulator vs Aer matrix-product states

The subspace backend is the answer to "can it go past N=4". It is exact (proof
above, tested amplitude by amplitude) and reaches N=9. The whole study above
runs in 1–27 s per run on a laptop. Aer's MPS simulator was measured against it
too (`--aer-only`, qiskit-aer 0.17.2, not a project dependency). Basis start,
20,000 shots, total variation distance (TVD) to the exact distribution, next to
the TVD of an equally large ideal sample (the shot-noise floor):

| instance | qubits | p | CX | TVD, Aer MPS | TVD, ideal sample | Aer run | subspace evolve |
|---|---|---|---|---|---|---|---|
| `n5_static` | 25 | 1 | 1920 | 0.0030 | 0.0024 | 0.57 s | 0.7 ms |
| `n5_static` | 25 | 2 | 3840 | 0.0050 | 0.0067 | 1.14 s | 0.3 ms |
| `n5_td30d` | 25 | 2 | 3840 | 0.0050 | 0.0075 | 1.14 s | 0.3 ms |
| `n8_*` first 7 objects | 49 | 1 | 6048 | 0.0033 | 0.0042 | 0.9–1.1 s | 0.7 ms |
| `n8_*` first 7 objects | 49 | 3 | — | **did not finish in 600 s** | | | |
| `n8_*` | 64 | 1 | — | **refused: Aer's limit is 63 qubits** | | | |

(`n5_td30d` p=1 and the N=7 static/td pair are identical to their siblings. From
a basis state the first phase separator is a global phase, so at p=1 the
distribution does not depend on the costs.)

* **Where it runs, Aer MPS is trustworthy**: its TVD sits at the shot-noise
  floor, with zero leakage out of $F$.
* **It does not extend reach.** Cost grows with entanglement, i.e. with depth:
  49 qubits went from 1 s at p=1 to over 10 minutes at p=3. And N=8 is simply
  over Aer's 63-qubit limit. Any state in $F$ has Schmidt rank at most
  $N!$ across any cut, so MPS is exact in principle. In practice that bound is
  the same $N!$ the subspace simulator pays directly, without the tensor
  overhead.
* **One Aer bug, recorded rather than worked around silently.** On this
  circuit, qiskit-aer 0.17.2's MPS `save_amplitudes` returns wrong amplitudes
  (max error 0.85 at N=3). The same method's `save_statevector` and its
  sampling are correct to 1e-11, as is `save_amplitudes` on the statevector
  method. The reproduction is in the JSON under `save_amplitudes_bug`. Hence
  the sampling comparison above.

So `qaoa-swap` keeps the refusal behaviour: `statevector` refuses N>4 and
`subspace` refuses N>9, each with `feasible=False` and the size in the reason.
Aer is not wired in as a backend, because it adds nothing the subspace backend
does not already do exactly and faster.

### Hardware (basis start): not run, and what it would cost

**No IBM Quantum token was configured in this workspace**, so nothing was
submitted. `docs/data/qaoa_hardware.json` records `status: skipped` with the
reason and date (2026-09-28 UTC). To run it: put `IBM_QUANTUM_TOKEN=<API key>`
(and optionally `IBM_QUANTUM_INSTANCE=<CRN>`) in the repo root's `.env`, which is
gitignored and symlinked into each Conductor workspace by `conductor.json`.
Then run `uv run python scripts/qaoa_hardware.py` with the `quantum` and `ibm`
extras installed. The script never prints the token.

This is a **different question** from the quality study, and its numbers must
not be read against the table above. A hardware circuit cannot prepare the
uniform superposition, so it starts in one basis permutation. From a basis state
the first phase separator is a global phase, so reps=1 has nothing to optimise.
The run is therefore a fidelity test of one full layer, with β = π/4 (every
partial swap a 50/50 superposition) and γ = π / mean path cost, both fixed. It
compares hardware feasibility, P(optimal), mean Δv and TVD against the exact
noiseless distribution of the same circuit. A free optimisation picks β ≈ 0,
"stay put", and the transpiler then deletes the near-identity rotations, which
leaves nothing to test. That is why β is not optimised.

Transpiled at `optimization_level=3` against `FakeFez` (a Heron r2 calibration
snapshot shipped with qiskit-ibm-runtime 0.50.0). Estimated fidelity is the
product of (1 − error) over every two-qubit gate and measurement, from that
calibration:

| circuit | qubits | depth | two-qubit gates | estimated fidelity | would submit |
|---|---|---|---|---|---|
| N=3 (first 3 of `n4_static` / `n4_td30d`), p=1 | 9 | ~925 | 368 | 0.33 | yes |
| N=4 (`n4_static` / `n4_td30d`), p=1 | 16 | ~1670 | 1212 | 0.021 | yes (threshold 0.01) |

A 33% estimated fidelity at N=3 means hardware feasibility will fall well below
the noiseless 1.0. How far it falls, relative to the 6/512 = 0.0117 uniform
bitstring rate, is the measurement the run would make. N=4 is marginal by the
backend's own numbers.

### Limitations specific to `qaoa-swap`

* The uniform start is supplied as an initial statevector, not as a gate. No
  hardware result can use it, so hardware and quality numbers answer different
  questions by construction.
* One layer costs (N−1)·C(N,2) four-local partial mixers, 8 Pauli rotations
  each: ~310 CX per layer at N=3 and ~940 at N=4 before routing. On heavy-hex
  hardware that is the binding constraint long before qubit count is.
* The subspace simulator is exact because the reachable space is only N!. That
  cuts both ways. Simulating the ansatz classically costs about as much as
  enumerating every sequence, and Held-Karp beats enumeration.
* COBYLA with random restarts does not converge at p ≥ 3 within 3 × 1000
  evaluations, so the depth trend above is a lower bound on what the ansatz can
  do. It is not a measurement of the ansatz's ceiling.
