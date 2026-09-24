# Physics: what the cost models actually model

Scope: `src/dextrivia/costs/realistic.py` and `src/dextrivia/costs/selection.py`.
The baseline `costs/hohmann.py` is unchanged and still the default for
`dextrivia build`; everything here is opt-in and lives beside it, so the
"degenerate benchmark" control in `tests/test_degeneracy.py` keeps passing.

All numbers below are measured against the committed snapshot
`data/snapshots/iridium33_20260402.json` (107 objects) at its median TLE epoch,
2026-04-02T07:05:22Z. Rerun them with `scripts/build_instance_family.py` and
`scripts/plot_cloud.py`.

## 1. The problem with altitude-only costs

| Quantity | Value in this snapshot |
|---|---|
| Mean altitude (mean SMA − R⊕) | 513 – 892 km |
| Inclination | 85.96° – 86.47° (spread **0.51°**) |
| RAAN | 1.1° – 359.7° (**the full circle**) |
| Pairwise plane angle | median **45.5°**, max 172.8° |
| Impulsive plane change at the median angle, 700 km | **5.80 km/s** |
| Whole 10-object mission under `hohmann.py` | 0.126 km/s |

Inclination alone says the cloud is one orbit. RAAN says it is 107 of them. A
model that sees only altitude prices an entire mission at less than 3% of one
median leg's true plane change.

Exchange rate at 700 km, which is what makes the sequencing problem
two-dimensional and therefore non-trivial:

* **1° of plane ≈ 0.131 km/s**
* **100 km of altitude ≈ 0.053 km/s**

![RAAN vs altitude for the Iridium-33 cloud](figures/raan_altitude.png)

## 2. Plane angle

The true angle between two orbit planes, from inclination and RAAN:

```
cos θ = cos i₁ cos i₂ + sin i₁ sin i₂ cos(Ω₁ − Ω₂)
```

`|i₁ − i₂|` is not an approximation of this, it is a different quantity: it
reports ≤ 0.51° for every pair in this cloud while the truth reaches 172.8°.

## 3. Impulsive model — `ImpulsiveCostModel`

Two finite burns on a Hohmann transfer ellipse, with the plane change **folded
into the burns** rather than paid separately. Burn 1 does `s` of the rotation
and burn 2 the remaining `θ − s`; each burn's cost is the law of cosines
between the pre- and post-burn velocity vectors:

```
Δv(s) = √(v₁² + vₜ₁² − 2 v₁ vₜ₁ cos s) + √(v₂² + vₜ₂² − 2 v₂ vₜ₂ cos(θ − s))
```

`s` is chosen numerically to minimise the sum: the objective is smooth but not
convex, so the code does a 181-point global scan followed by golden-section
refinement inside the winning bracket (no scipy). The optimum is interior and
pushes most of the rotation onto the burn at the higher radius, where velocity
is lowest — but "most", not "all", and the share depends on the radius ratio,
which is why it is solved rather than assumed.

Implementation note: both terms are evaluated as `√((a−b)² + 4ab sin²(s/2))`.
That is algebraically the law of cosines, but the `a² + b² − 2ab cos s` form
loses ~9 significant digits exactly where this project needs them — small plane
angles between similar orbits, i.e. every leg inside a cluster.

**`separate_burn_dv` is the upper bound**: Hohmann + a standalone
`2 v sin(θ/2)` plane change at the higher orbit. It is never cheaper than the
combined burn (asserted for a grid of geometries in the tests). For a
300 km → GEO transfer with a 28.5° plane change: combined **4.23 km/s**,
separate **5.41 km/s**.

Validation (`tests/test_realistic_costs.py`, every case a number from outside
this repo):

| Case | Expected | Model |
|---|---|---|
| 300 km circular → GEO, coplanar | ~3.9 km/s | 3.893 km/s |
| θ = 0 | exactly `hohmann_dv` | exact to 1e-12 |
| r₁ = r₂ | `2v sin(θ/2)` | exact to 1e-9, for θ from 0.5° to 180° |
| GTO → GEO, 28.5° | ~4.2 km/s combined, ~5.4 km/s separate | 4.23 / 5.41 |

## 4. Low-thrust model — `EdelbaumCostModel`

Edelbaum (1961), the standard first-order estimate for an electric servicer in
active-debris-removal studies:

```
Δv = √(v₁² + v₂² − 2 v₁ v₂ cos(π/2 · θ))
```

Limiting cases, all asserted:

* θ = 0 → `|v₂ − v₁|`, the pure spiral.
* r₁ = r₂ → `2v sin(πθ/4)`, a pure low-thrust plane change — `π/2` times the
  impulsive `2v sin(θ/2)` for small angles.

**Low thrust never costs less Δv than impulsive for the same geometry.**
Rotating the velocity vector continuously is strictly worse than rotating it
once at the cheapest point. Its real advantage is propellant *mass* through
specific impulse, and this package measures delta-v only — so the Edelbaum
model will always lose here. That is a property of the metric, not of the
propulsion, and it is why the committed instance family uses the impulsive
model.

**Validity limit:** above θ ≈ 114.6° (2 rad) the effective angle `π/2·θ` passes
π and the closed form starts reporting *cheaper* transfers for *bigger* plane
changes. The effective angle is clamped at π, giving `v₁ + v₂` — stop dead and
re-accelerate. Treat any Edelbaum number above 114.6° as "too expensive to
matter" rather than as a quantity.

## 5. J2 nodal precession and the time-slot mapping

```
dΩ/dt = −3/2 · n · J₂ · (R⊕/p)² · cos i        p = a(1 − e²),  n = √(μ/a³)
```

WGS72 `J₂ = 1.082616e-3`, matching SGP4. Validated against published orbits:

| Orbit | Published | `nodal_precession_deg_per_day` |
|---|---|---|
| ISS-like, 400 km, 51.64° | ≈ −5.0 °/day | −5.00 °/day |
| Sun-synchronous, 700 km, 98.2° | +0.9856 °/day | +0.987 °/day |
| Polar, 90° | 0 | 0 |

and against SGP4's own node regression on real snapshot objects over 100 days:
**agreement within 0.3%** (test tolerance 2%).

### Slot semantics

`ProblemInstance` indexes time-dependent costs by *position in the sequence*,
not wall clock. This module supplies the missing map with one number:

> **leg _k_ departs at `t_k = epoch + k · Δ`**, where `Δ = delta_per_leg_days`
> is the whole per-object budget — transfer, rendezvous, capture, release.

Δ and the rule string are written into `ProblemInstance.metadata`
(`delta_per_leg_days`, `leg_departure_rule`). Δ is deliberately a single
constant: letting it depend on the leg would make departure time depend on
*which* objects were visited, which is the continuous-time scheduling problem
the slot convention exists to avoid.

`C[k, i, j]` is then the impulsive cost of i→j using each object's mean
elements propagated to `t_k`. Elements come from SGP4 itself (`Satrec.am/em/im/Om`
after propagation), so the node regression used in the costs is the
propagator's, including J4 and drag; the analytic `dΩ/dt` above exists to
explain and check that drift, not to replace it.

### How much does waiting actually buy? (Not much, and say so.)

The nodal drift of this cloud is **common-mode**: every object regresses at
−0.405 to −0.505 °/day, so the drift largely cancels in ΔΩ.

| | |
|---|---|
| Nodal rate | −0.405 to −0.505 °/day |
| **Differential** rate between a median pair | **0.016 °/day** |
| 90th percentile pair | 0.045 °/day |
| Fastest-separating pair | 0.100 °/day |
| Days to change a pair's RAAN separation by 10° | **612 days** (median), 100 days (fastest pair) |

So on a mission of weeks, differential drift is noise. On a *long* one it is
not: with Δ = 30 days over a 10-object mission (240 days), leg costs move by
0.44 km/s on average and up to 1.39 km/s between the first and last slot. That
is why the family ships time-dependent variants at Δ = 30 d, and why they are
the instances greedy loses on.

### The loiter window, and why it is off by default

`ImpulsiveCostModel(window_days=..., window_samples=...)` defines
`C[k,i,j]` as the **minimum** over sampled departure times in
`[t_k, t_k + window]`. It is implemented, tested, and **disabled by default**,
for two reasons:

1. **It buys almost nothing here.** On the N=10 cluster with a 20-day window:
   median saving per leg **0.000 km/s** (drift usually makes a pair worse, so
   the minimum is just the nominal time), best leg 0.052 km/s.
2. **It is dishonest as an objective.** The minimum hands the servicer the
   drift discount without the objective ever paying for the 20 days of waiting.
   A model that rewards loitering must also charge for it — with a propellant
   budget for station-keeping, a mission-duration penalty, or a deadline. None
   of those exist in this package yet.

Turn it on only with the schedule cost in hand.

## 6. Target selection — `plane-cluster`

Full-cloud tours are physically absurd: a median leg costs 5.8 km/s, more than
a launch to GEO, *per leg*. Real ADR studies pick targets that already share a
plane. The rule (`costs/selection.py`):

1. Keep objects within `raan_window_deg` (10°), `inc_window_deg` (2°) and
   `alt_band_km` (250 km) of a **seed** object.
2. Seed = the object with the most neighbours surviving that filter, ties
   broken by lowest NORAD id. Pinnable with `seed_norad`.
3. Take the N survivors closest to the seed in *true plane angle*, ordering the
   instance by NORAD id.

The window is a radius around the seed, so a cluster's own span can be twice as
wide — measured and recorded per instance as `cluster_max_plane_angle_deg`.
The inclination window is effectively inert for this cloud (0.51° total spread)
and is kept for mixed catalogues.

The rule is fixed before any solver runs and never looks at delta-v, let alone
at which solver wins. Selecting instances by "where greedy loses" would cook
the benchmark this repository exists to produce.

Instances are assembled here rather than through `instances.build_instance`,
whose `rule` argument is `first`/`random` only (foundation-owned). No interface
in `core.py` changed; `build_cluster_instance` populates the same
`ProblemInstance` with a superset of the same metadata keys.

## 7. The committed instance family

`data/instances/iridium33_20260402_planecluster-v1_n{N}_{static|td30d}.npz`,
N ∈ {4, 5, 8, 10, 15, 20}, 12 files, regenerated by
`uv run python scripts/build_instance_family.py`. Δ = 30 days. Each file
records the snapshot filename, selection rule and window, epoch and its source,
cost model, Δ and the leg→wall-clock rule, and the family version.

Greedy (best-of-all-starts nearest neighbour) against the Held-Karp optimum,
plus what visiting in altitude order costs — the strategy that *was* exactly
optimal under the coplanar model:

| N | variant | exact (km/s) | greedy (km/s) | greedy gap | altitude order (km/s) | vs exact |
|---|---------|--------------|---------------|-----|-----------------------|----------|
| 4 | static | 0.5946 | 0.5946 | +0.00% | 0.9641 | +62.1% |
| 4 | time-dependent | 0.5205 | 0.5205 | +0.00% | 0.9752 | +87.4% |
| 5 | static | 0.8296 | 0.8296 | +0.00% | 1.3820 | +66.6% |
| 5 | time-dependent | 0.8314 | 0.8314 | +0.00% | 1.2986 | +56.2% |
| 8 | static | 1.0824 | 1.0824 | +0.00% | 2.5544 | +136.0% |
| 8 | time-dependent | 1.1124 | 1.1720 | **+5.37%** | 2.3730 | +113.3% |
| 10 | static | 1.3268 | 1.3268 | +0.00% | 5.1450 | +287.8% |
| 10 | time-dependent | 1.5165 | 1.5165 | +0.00% | 5.0583 | +233.5% |
| 15 | static | 2.0151 | 2.0151 | +0.00% | 8.8540 | +339.4% |
| 15 | time-dependent | 2.1855 | 2.2988 | **+5.18%** | 8.8066 | +302.9% |
| 20 | static | — exceeds Held-Karp limit 18 — | 2.4485 | — | 13.1215 | — |
| 20 | time-dependent | — exceeds Held-Karp limit 18 — | 3.6723 | — | 17.2940 | — |

Read honestly:

* **Altitude order is dead.** It is 56–339% worse than optimal on every
  instance, and is never the optimal sequence. The degeneracy the old model had
  is gone.
* **Greedy still ties the optimum on every static instance.** A static
  plane-aware matrix is still symmetric and near-metric, and nearest-neighbour
  from all N starts finds the optimum at these sizes. Claiming otherwise would
  be the easy lie; the cost model got real, greedy did not get worse.
* **Time dependence is what breaks greedy** — 5.4% at N=8, 5.2% at N=15.
  Greedy commits to a cheap early leg and pays for it when the planes have
  drifted; Held-Karp plans around the drift. This is the headroom a
  quantum/quantum-inspired solver has to compete for.
* **N=20 records a miss, not a crash**, which is what the exact solver's
  `feasible=False` contract is for.

## 8. What is still missing

In rough order of how much delta-v it is worth:

1. **Phasing.** The servicer is assumed to arrive at the right point in the
   target's orbit for free. In reality that is paid in delta-v or in waiting,
   and for a dense plane usually waiting. The single largest omission.
2. **Three-burn (bi-elliptic) transfers.** Above ~60° of plane angle, raising
   apoapsis to rotate cheaply beats the two-burn manoeuvre. Every large-angle
   cost here is therefore an upper bound.
3. **Eccentricity.** Orbits are treated as circular at the mean semi-major
   axis. Real e is 8e-5 … 2.1e-2, biasing a transfer by ~13 m/s at the median
   and ~160 m/s for the worst object.
4. **Finite burns and gravity losses** — a few percent for a chemical stage.
5. **Argument-of-perigee drift**, J2 effects during the transfer itself,
   lunisolar and drag perturbations beyond what SGP4's mean elements carry.
6. **Time windows, propellant budget, vehicle constraints, conjunction risk
   weighting.** None of them exist.

A cost that is an upper bound is stated as one. Nothing here has been tuned to
make any solver look good.
