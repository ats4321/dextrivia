# QUBO formulation of open-path debris sequencing

Owned by the QUBO workspace. Code: `src/dextrivia/qubo/formulation.py`,
`src/dextrivia/solvers/quantum_annealing.py`,
`src/dextrivia/solvers/quantum_qaoa.py`,
`src/dextrivia/solvers/ortools_routing.py`.

This document records a formulation and a characterization. It is **not** a
quantum-advantage claim.

The short version, measured against the plane-aware `planecluster-v1` instance
family (§7): on time-dependent costs at N=8, `sa-qubo` reaches the exact optimum
where greedy loses 5.37% — the first result in this project where the QUBO route
beats the classical heuristic on an instance with any headroom. At N=15 the same
solver lands +41.6%, eight times worse than greedy. One win and one collapse;
neither should be quoted without the other. `qaoa` cannot reach either size.

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

### The default is not confirmed on harder instances

The same sweep on `planecluster-v1_n15_td30d` (225 variables, 1000 reads × 2000
sweeps) is **not monotone and is bad everywhere**:

| factor | 1.01 | 1.05 | 1.1 | 1.5 |
|---|---|---|---|---|
| gap to optimum | +34.7% | +14.9% | **+41.6%** | +41.8% |
| raw feasibility | 1.00 | 1.00 | 1.00 | 1.00 |

The shipped default of 1.1 is the *worst* of the four on that instance, and 1.05
is nearly three times better. One instance is not enough to re-tune on — the
ordering is not monotone, so this may be sampler noise rather than signal — but
it is enough to say plainly that **`DEFAULT_PENALTY_SAFETY = 1.1` is calibrated
on one easy instance and should not be trusted at N=15 and above.** Sweep it per
instance rather than accepting the default, which is what `sweep_penalty` is for.

Note also what the table rules *out*: feasibility is 1.00 at every factor, so
the penalty is not the reason `sa-qubo` performs badly at N=15 (§7). The
constraint terms are working; the objective is simply not being optimised.

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
| OR-Tools routing | `ortools` | `ortools` | no practical $N$ limit; **cannot do time-slotted costs** |

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

The physics workspace has since landed plane-aware cost models and a committed
`planecluster-v1` instance family, so this section no longer rests on the
degenerate Hohmann instance. Held-Karp is the oracle throughout; `sa-qubo` at
1000 reads x 2000 sweeps, seed 11; `ortools` at a 2 s limit.

| instance | N | vars | optimum (km/s) | `greedy` | `sa-qubo` | `ortools` |
|---|---|---|---|---|---|---|
| `n8_static` | 8 | 64 | 1.0824 | +0.00% | +0.00% | +0.00% |
| `n10_static` | 10 | 100 | 1.3268 | +0.00% | +12.57% | +0.00% |
| `n8_td30d` | 8 | 64 | 1.1124 | +5.37% | **+0.00%** | **refused** |
| `n15_td30d` | 15 | 225 | 2.1855 | +5.18% | +41.56% | **refused** |

Raw feasibility was 1.00 on every row — the one-hot constraints are not the
difficulty here, the objective is.

Three things this table actually says:

1. **Static instances still rank nothing.** Greedy ties the exact optimum on
   *every* static instance in the family, N=4 through N=15. A plane-aware static
   cost matrix is still near-metric, so nearest-neighbour is already optimal and
   there is nothing for any solver to demonstrate. `sa-qubo` losing 12.6% at
   `n10_static` is a statement about the penalty encoding's overhead on an easy
   problem, not about annealing.

2. **`n8_td30d` is the first genuine win in this project.** Greedy loses 5.37%;
   `sa-qubo` reaches the exact optimum. Time dependence is what breaks
   nearest-neighbour — a cheap next leg now can strand the servicer in a plane
   that is expensive to leave three legs later — and that is exactly the
   structure a QUBO carries natively, because the cost of leg *p* is baked into
   the coupling between positions *p* and *p+1* rather than discovered greedily.

3. **It does not survive to N=15.** At 225 variables `sa-qubo` lands +41.56%,
   eight times worse than greedy. This is not a penalty-tuning artefact: the
   sweep at that instance gives +34.7%, +14.9%, +41.6%, +41.8% at factors 1.01,
   1.05, 1.1, 1.5 — non-monotonic, all bad, feasibility 1.00 throughout. The
   constraint terms are doing their job and the objective is simply not being
   optimised.

   Nor is it a budget artefact. Throwing effort at it helps once and then
   stops:

   | reads × sweeps | runtime | gap to optimum |
   |---|---|---|
   | 1000 × 2000 | 4 s | +41.56% |
   | 2000 × 20000 | 86 s | +17.12% |
   | 2000 × 100000 | 458 s | +19.55% |

   A 20x increase in sampling effort buys 41.6% → 17.1%. A further 5x — 115x
   the original budget, 458 seconds against greedy's sub-millisecond — buys
   nothing, and lands marginally worse, which is within sampler noise. **The
   gap plateaus around 17-20%, still three to four times worse than greedy at
   +5.18%.** This is a ceiling, not a budget problem, and it is the single most
   important number in this document: the penalty-encoded QUBO route has a
   quality floor at N=15 that more compute does not lift.

**Two data points are not a scaling law.** One win at N=8 and one collapse at
N=15 is the honest summary, and the win must not be quoted without the collapse.
What the effort sweep does establish is that the N=15 result is a property of
the formulation-plus-sampler, not of how long it was left to run.

### The baseline is missing where it matters

`ortools` refuses both `td30d` rows, because a routing arc-cost callback cannot
express `C[t,i,j]` (§6). Those are precisely the two instances with any headroom.
So the N=15 comparison currently has **no strong classical bar above Held-Karp
range at all** — `sa-qubo`'s +41.56% is measured against an exact oracle that
will itself run out at N=18, and against greedy. Filling that gap is the most
valuable next piece of work in this workspace: without it, the first instance
size where the quantum-inspired route might matter is also the first size where
there is nothing credible to compare it to.

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
| `sa-qubo` | $N^2$ spins | $N=40$ by policy | runtime |
| `exact` (oracle) | — | $N=18$ | $2^N N^2$ |
| `brute` (oracle) | — | $N=8$ | $N!$ |
| `ortools` | — | none reached | time-limited heuristic |

Every ceiling is reported as `feasible=False` with a reason in `metadata`, never
as an exception — a benchmark has to record a miss, not crash on it.
