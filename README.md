# Dextrivia

**Orbital debris tracking and mission sequencing for low Earth orbit.**

Dextrivia ingests live Two-Line Element (TLE) data from Celestrak, propagates each debris object to a common epoch using SGP4, builds a delta-v cost matrix via Hohmann transfer approximations, and solves for an efficient removal sequence. It ships a greedy nearest-neighbor baseline and an exact Held-Karp solver, so every heuristic result can be compared against the true optimum rather than against hope. The long-term goal is a benchmarked comparison of classical and quantum / quantum-inspired solvers.

The sequencing problem here is an **open path**: the servicer starts at the first object and stops at the last, with no return leg.

---

## Background

The Iridium-Cosmos collision of 2009 produced more than 2,000 trackable fragments in low Earth orbit (LEO). These objects, concentrated between roughly 770 km and 800 km altitude, represent a well-studied and operationally significant debris cloud. Dextrivia uses this dataset as its primary case study.

Debris removal is fundamentally a sequencing problem: given N objects to visit and a cost (delta-v) to transfer between any pair, find the ordering that minimizes total propellant expenditure. This is structurally equivalent to the Traveling Salesman Problem, which is NP-hard in the general case and motivates the use of quantum-inspired optimization approaches.

---

## How It Works

```
Celestrak TLE feed
       |
  dextrivia fetch  -- writes a new immutable snapshot under data/snapshots/
       |
  dextrivia build  -- selects N objects (explicit rule + seed), propagates them
                      to the snapshot's median TLE epoch, builds the N x N
                      Hohmann delta-v matrix, saves a ProblemInstance
       |
  dextrivia solve  -- greedy (all start points), exact (Held-Karp), or brute
```

### Orbital Propagation

`src/dextrivia/propagation.py` wraps the `sgp4` Python library. Each object is propagated to a UTC epoch, returning position and velocity in the Earth-Centered Inertial (ECI) frame. The epoch is a parameter and defaults to the **median TLE epoch of the snapshot** — propagating far from the elements' own epoch quietly degrades every altitude.

Transfer costs use the **mean semi-major axis** SGP4 carries (`Satrec.am`), not the instantaneous `|r|`. The instantaneous altitude oscillates once per orbit — about 285 km peak-to-peak for the most eccentric object in this dataset — which would otherwise encode "where each object happened to be at the chosen epoch" into the cost matrix. All radii use the WGS72 Earth radius (6378.135 km), the constant SGP4 itself uses.

### Delta-V Cost Matrix

`src/dextrivia/costs/hohmann.py` models each debris object as being in a circular coplanar orbit at its mean semi-major axis. The cost to transfer between object i and object j is the two-burn Hohmann delta-v:

```
dv1 = |v_transfer_periapsis - v_circular_i|
dv2 = |v_circular_j - v_transfer_apoapsis|
total = dv1 + dv2
```

The coplanar assumption is an approximation. Real transfers between objects at different inclinations require a plane-change burn, which can dominate the delta-v budget. The Hohmann model is appropriate for initial feasibility screening and baseline comparison.

### Greedy Baseline

`src/dextrivia/solvers/greedy.py` implements the classical nearest-neighbor heuristic from every possible start point and retains the best result. `src/dextrivia/solvers/exact.py` solves the same instance exactly by Held-Karp dynamic programming (N <= 18), with a brute-force enumerator (N <= 8) used as its test oracle.

For the 10-object Iridium-33 instance, greedy finds a total delta-v of **0.1262 km/s** — and so does the exact solver. That tie is not a compliment to the heuristic: because the current cost model derives every cost from a single scalar per object (altitude), the optimal order is simply "visit in altitude order", which greedy trivially finds. **The benchmark is degenerate until the cost model models more than altitude.** `tests/test_degeneracy.py` documents this.

---

## Repository Structure

```
src/dextrivia/
|-- core.py                 ProblemInstance / CostModel / Solver / Solution
|-- propagation.py          SGP4 wrapper, mean semi-major axis at an epoch
|-- snapshots.py            fetch / load / select immutable TLE snapshots
|-- instances.py            snapshot + selection + cost model -> instance
|-- cli.py                  dextrivia fetch | build | solve
|-- costs/hohmann.py        coplanar Hohmann delta-v cost model
|-- solvers/greedy.py       nearest-neighbor from every start point
|-- solvers/exact.py        Held-Karp DP + brute-force oracle
data/snapshots/             committed, timestamped TLE sets
data/instances/             built problem instances (derived, gitignored)
tests/                      pytest suite
dextrivia-landing.html      project landing page
CLAUDE.md                   interfaces, conventions, directory ownership
```

---

## Installation

**Requirements:** Python 3.11 or later.

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run pytest
```

---

## Usage

### 1. Fetch fresh TLE data

```bash
uv run dextrivia fetch
```

Downloads the current Iridium-33 debris catalog from Celestrak into a **new** file under `data/snapshots/`. Existing snapshots are never overwritten.

### 2. Build a problem instance

```bash
uv run dextrivia build --n 10 --select first
```

Selects objects by an explicit rule (`first` or seeded `random`), propagates them to the snapshot's median TLE epoch, builds the delta-v matrix, and saves a `ProblemInstance` under `data/instances/`. Snapshot, selection rule, seed, epoch and cost model are all recorded in the instance's metadata.

Without `--snapshot`, `build` uses the newest snapshot **of the default `iridium-33-debris` group**, ordered by the date in the filename. Snapshots of another group are never picked implicitly — switching datasets always takes an explicit `--snapshot`.

### 3. Solve it

```bash
uv run dextrivia solve --instance data/instances/iridium33_20260402_n10_first.npz --solver greedy
uv run dextrivia solve --instance data/instances/iridium33_20260402_n10_first.npz --solver exact
```

**Sample output:**

```
solver     exact
instance   data/instances/iridium33_20260402_n10_first.npz (N=10, hohmann-coplanar)
epoch      2026-04-02T07:05:22.337088+00:00
total dv   0.126246 km/s
runtime    6.6 ms
sequence   (NORAD id, leg delta-v km/s)
   1.  33862   start
   2.  33860   +0.005137
   3.  33777   +0.022222
   ...
```

Objects are identified by NORAD catalog ID throughout: 106 of the 107 objects in this dataset share the name `IRIDIUM 33 DEB`.

---

## Data Sources

| Source | Description | License |
|--------|-------------|---------|
| [Celestrak](https://celestrak.org) | TLE catalog, Iridium-33 debris group | Free for non-commercial use |
| [Space-Track](https://www.space-track.org) | Official USSPACECOM catalog | Requires free account registration |

TLEs are fetched from Celestrak's `gp.php` endpoint using the `iridium-33-debris` group identifier. No API key is required for Celestrak. Space-Track is not currently used.

---

## Limitations and Future Work

- **Degenerate benchmark:** with altitude-only costs the optimum is just "visit in altitude order", so greedy already ties the exact solver and no solver can show an advantage. This is the first thing that needs fixing.
- **Coplanar assumption:** The Hohmann model ignores inclination differences between debris objects. A full 3D transfer cost accounting for plane changes will significantly change the cost matrix.
- **Epoch sensitivity:** Debris orbits decay over time. The cost matrix is only valid near the propagation epoch, which is why the epoch defaults to the snapshot's median TLE epoch. Long-horizon planning requires re-propagation at each step.
- **QUBO solver:** The next milestone is formulating the sequencing problem as a QUBO and solving it with a quantum annealer or simulated annealing backend (D-Wave, Qiskit, or Neal), benchmarked against greedy and the exact optimum.
- **Conjunction analysis:** The current tool does not model collision probabilities or conjunction events. Adding a conjunction screening step would enable risk-weighted prioritization.
- **Maneuver modeling:** Actual debris removal vehicles have propellant limits and attitude constraints. The cost matrix does not model these.

---

## License

MIT. See `LICENSE` for details.

---

## Acknowledgments

- [sgp4 Python library](https://pypi.org/project/sgp4/) by Brandon Rhodes, implementing the Vallado SGP4 formulation.
- [Celestrak](https://celestrak.org) maintained by T.S. Kelso for providing freely accessible orbital element sets.
- The Iridium-Cosmos collision dataset has been a standard reference for LEO debris research since 2009.
