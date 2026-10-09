# Lower bounds: how a best-known reference becomes proven

Code: `src/dextrivia/solvers/highs_mip.py` (`highs`, `lp_lower_bound`) and
`src/dextrivia/solvers/cpsat.py`. Tests: `tests/test_bounds.py`. Evidence
for the N caps: `scripts/bounds_study.py` → `docs/data/bounds_study.json`.

## Why this exists

Held-Karp stops at N=18. Above that, the benchmark's reference used to be
"the best any solver found", and nobody could say whether greedy was optimal
or simply unchallenged. The question that started this work was an N=30
static instance: greedy = localsearch = random-start sa-perm = 3.9970 km/s,
while a 30 s cold CP-SAT run returned 12.75. **`highs` proves 3.9970 optimal
in about 7 s** (`test_pinned_n30_april_40deg_window_greedy_is_provably_optimal`).
Greedy was optimal. Nothing was hiding.

Every reference in the benchmark is now one of:

| `reference_kind` | meaning |
|---|---|
| `exact` | Held-Karp ran (N ≤ 18). |
| `proven` | Held-Karp could not run, but the best path found is within 1e-5 km/s of a certified lower bound. |
| `best-known` | Neither. Printed with the bound, the solver that produced it, and the gap between them. |

A bound above a reference makes `add_references` raise. That would be a wrong
bound, not a weak one.

## The model

The time-indexed ("3-index") formulation for an open path with N−1 legs:

```
x[i,p]    object i is visited p-th
y[p,i,j]  leg p flies i -> j
z[i,j]    = sum_p y[p,i,j]           (aggregate arc, cost 0)

sum_p x[i,p] = 1, sum_i x[i,p] = 1
sum_j y[p,i,j] = x[i,p]        sum_i y[p,i,j] = x[j,p+1]
minimise sum_p sum_ij leg_costs(p)[i,j] * y[p,i,j]
```

Position index p is the slot index t, so time-slotted costs need nothing
extra. It is exact for integral x, because y is then forced.

**The plain LP relaxation is weak here**, 17–44% below the optimum on the
committed instances, because a fractional x can spread one object over several
positions. **Subtour cuts** close most of that gap. Close the path through a
depot d with arcs d→i of weight x[i,0]. Every integral solution then carries
one unit of flow from d to each object, so for every object set T:

```
sum_{k in T} x[k,0] + sum_{i not in T, j in T} z[i,j] >= 1
```

Separation is exact: one max-flow per object (`scipy.sparse.csgraph`). The
violated cuts are added and the LP is re-solved (HiGHS interior point) until
none is violated, a round limit is reached, or the time limit binds inside an
LP. A round whose LP did not finish is discarded, because an unfinished LP
proves nothing. `highs` then runs HiGHS branch-and-bound (`scipy.optimize.milp`,
`mip_rel_gap=1e-9`), with the cuts added, on the remaining time. It reports the
larger of the cut-LP value and `mip_dual_bound`.

**Certification.** Both the LP and the MIP bound are floating-point HiGHS
duals (feasibility tolerance 1e-7). Every reported bound has 1e-6 km/s
subtracted. That is enough for a benchmark and is stated in every record's
`bound_certification`. It is not an exact-rational proof.

**Validation.** On every committed instance with N ≤ 15 (29 files, v1 and v2,
both cost shapes), the cut-LP bound and the `highs` bound are both ≤ Held-Karp,
and `highs` reaches the Held-Karp optimum with `proven_optimal`. The same holds
on 12 random-cost instances (N=7, static and slotted) against brute force.
Random costs are the adversarial case, because they have no metric structure
for the cuts to exploit.

## CP-SAT changes

Measured with `uv run python scripts/bounds_study.py --ablation`, 30 s, seed 1,
8 workers (`docs/data/cpsat_ablation.json`). "Gap to best known" is measured
against the better of this path and a seed-1 `ils` run:

| instance | warm start | circuit | path (km/s) | certified bound (km/s) | gap to best known | wall (s) |
|---|---|---|---|---|---|---|
| v1 `n20_td30d` | no | no | 3.6723 | 2.8526 | 22.32% | 30.4 |
| v1 `n20_td30d` | yes | no | 3.6723 | 2.8255 | 23.06% | 30.6 |
| v1 `n20_td30d` | yes | yes | 3.6723 | 2.5815 | 29.70% | 30.5 |
| v2 `n20_td7d` | no | no | 2.6467 | 2.6466 | 0.00% | 19.6 |
| v2 `n20_td7d` | yes | no | 2.6467 | 2.6466 | 0.00% | 18.8 |
| v2 `n20_td7d` | yes | yes | 2.6467 | 2.6466 | 0.00% | 7.7 |
| v2 `n30_static` | no | no | 4.5135 | 2.6431 | 41.44% | 30.7 |
| v2 `n30_static` | yes | no | 4.5135 | 2.5713 | 43.03% | 30.9 |
| v2 `n30_static` | yes | yes | 4.5135 | 4.5135 | 0.00% | 5.6 |
| v2 `n40_td3d` | no | no | 8.7145 | 2.5030 | 56.32% | 31.5 |
| v2 `n40_td3d` | yes | no | 5.7308 | 2.3244 | 59.44% | 31.8 |
| v2 `n40_td3d` | yes | yes | 5.7308 | 4.0318 | 29.65% | 32.7 |

* **Warm start** (`AddHint` with the localsearch sequence, every variable
  hinted). On v2 `n40_td3d` the cold run returns 8.7145 km/s and the warm one
  5.7308 (= localsearch). By itself the warm start does nothing for the bound.
* **Redundant circuit.** `AddCircuit` over the aggregate arcs plus a depot
  gives CP-SAT's LP the subtour structure. It is what proves v2 `n30_static`
  and roughly halves the `n40_td3d` gap. It does not help v1 `n20_td30d`
  within 30 s. `highs` proves that instance optimal (see below). Both are on
  by default.
* **Rounding.** CP-SAT optimises costs rounded to mm/s, so its bound applies
  to the rounded problem. Each of the N−1 legs can round up by 0.5 mm/s, so
  `lower_bound_kms` now subtracts `(N−1)·0.5/COST_SCALE`. The unadjusted value
  is kept as `raw_bound_kms`. Before this change, a "bound" could exceed the
  true optimum by up to (N−1)·0.5 mm/s.
* **The bench bug.** `bench.py` read `meta["best_objective_bound_kms"]`, a key
  no solver writes. The `certified_lower_bound_kms` column of the first
  canonical run is empty in every row for that reason.

## The N caps, with evidence

`uv run python scripts/bounds_study.py` at 60 s, each run in its own process
(`docs/data/bounds_study.json`). **Wall times were measured on a shared, busy
laptop.** Load averages observed during the run ranged from about 7 to 38. Treat
the times as upper bounds; the memory and bound columns are the evidence.

| instance | N | solver | path (km/s) | certified bound (km/s) | gap to best known | peak RSS (MB) | wall (s) |
|---|---|---|---|---|---|---|---|
| v2 `n40_static` | 40 | `cpsat` | miss: no solution within 60.0s (status UNKNOWN) | - | - | 316 | 2.8 |
| v2 `n40_static` | 40 | `highs` | 5.5281 | 5.5281 | 0.00% | 461 | 27.1 |
| v2 `n40_td3d` | 40 | `cpsat` | 5.7308 | 3.9676 | 30.77% | 2086 | 35.9 |
| v2 `n40_td3d` | 40 | `highs` | 69.8402 | 4.9953 | 12.83% | 1458 | 60.5 |
| v3 `n40_static` | 40 | `cpsat` | 6.2776 | 6.2776 | 0.00% | 1621 | 14.3 |
| v3 `n40_static` | 40 | `highs` | 6.2776 | 6.2776 | 0.00% | 519 | 19.6 |
| v3 `n40_td3d` | 40 | `cpsat` | 13.4336 | 4.5279 | 66.29% | 2234 | 61.8 |
| v3 `n40_td3d` | 40 | `highs` | 49.1705 | 5.1076 | 61.98% | 1393 | 60.3 |
| v3 `n50_static` | 50 | `cpsat` | 7.2836 | 7.2835 | 0.00% | 2838 | 41.0 |
| v3 `n50_static` | 50 | `highs` | 7.2836 | 7.2836 | 0.00% | 821 | 56.1 |
| v3 `n50_td3d` | 50 | `cpsat` | 16.1235 | 2.5705 | 84.06% | 2941 | 63.3 |
| v3 `n50_td3d` | 50 | `highs` | 248.8684 | 5.8367 | 63.80% | 1786 | 60.3 |
| v3 `n60_static` | 60 | `cpsat` | 8.4777 | 6.7442 | 20.45% | 4319 | 69.3 |
| v3 `n60_static` | 60 | `highs` | 88.3465 | 7.8285 | 7.66% | 1123 | 60.4 |

* **`cpsat`: cap raised 40 → 50.** At N=50 it still proves the static v3
  instance optimal. At N=60 it needs 4.3 GB and leaves a 20.5% gap, where
  `highs` leaves 7.7% on 1.1 GB. The first row is a real miss: CP-SAT
  returned `UNKNOWN` after 2.8 s against a 60 s limit, with the machine at
  load average 38. Three later reruns each proved the instance optimal in
  10–15 s. The benchmark records such a miss as a row with its
  reason; it cannot affect a reference.
* **`highs` paths are not the point.** Where it cannot finish, its incumbent
  is whatever branch-and-bound last held: 69.84 km/s on v2 `n40_td3d`,
  against 5.73 for localsearch. It has no warm start, because
  `scipy.optimize.milp` cannot take one. Its bench rows are reported as they
  are; its job is the bound.
* **`highs`: cap 60.** Static instances are proven through N=50, and the
  static N=60 bound is within 7.7%. Time-dependent v3 is where both MIPs
  struggle: 62–84% between the best path found and the best certified bound at
  N ≥ 40. Whether that gap is headroom or bound weakness is exactly what the
  canonical benchmark's `ils` and multi-seed runs are there to test.
* **What the cut LP cannot do.** On time-slotted costs with large drift, a
  fractional solution can take a leg's cost from one slot and its structure
  from another. v1 `n20_td30d` shows it: the cut LP gives 2.4460 km/s, and only
  branch-and-bound closes the gap to the proven optimum of 3.6723.

