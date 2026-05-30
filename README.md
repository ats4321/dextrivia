# Dextrivia

**Orbital debris tracking and mission sequencing for low Earth orbit.**

Dextrivia ingests live Two-Line Element (TLE) data from Celestrak, propagates each debris object to a common epoch using SGP4, builds a delta-v cost matrix via Hohmann transfer approximations, and solves for an efficient removal sequence using a greedy nearest-neighbor baseline. The greedy result establishes the benchmark that a future QUBO (quantum unconstrained binary optimization) solver will attempt to beat.

---

## Background

The Iridium-Cosmos collision of 2009 produced more than 2,000 trackable fragments in low Earth orbit (LEO). These objects, concentrated between roughly 770 km and 800 km altitude, represent a well-studied and operationally significant debris cloud. Dextrivia uses this dataset as its primary case study.

Debris removal is fundamentally a sequencing problem: given N objects to visit and a cost (delta-v) to transfer between any pair, find the ordering that minimizes total propellant expenditure. This is structurally equivalent to the Traveling Salesman Problem, which is NP-hard in the general case and motivates the use of quantum-inspired optimization approaches.

---

## How It Works

```
Celestrak TLE feed
       |
  [setup.py]  -- fetches live TLEs via HTTPS, saves to debris_tles.json
       |
  [cost_matrix.py]  -- propagates each TLE with SGP4, extracts altitude,
                       computes N x N Hohmann delta-v matrix
       |
  [greedy.py]  -- nearest-neighbor greedy solver, O(N^2) per start point
       |
  [test_greedy.py]  -- evaluates all N start points, reports best sequence
```

### Orbital Propagation

`propagation.py` wraps the `sgp4` Python library. Each object is propagated to a user-specified UTC epoch, returning position and velocity in the Earth-Centered Inertial (ECI) frame. Altitude is derived as the norm of the position vector minus Earth's mean radius (6371 km).

### Delta-V Cost Matrix

`cost_matrix.py` models each debris object as being in a circular coplanar orbit at its propagated altitude. The cost to transfer between object i and object j is the two-burn Hohmann delta-v:

```
dv1 = |v_transfer_periapsis - v_circular_i|
dv2 = |v_circular_j - v_transfer_apoapsis|
total = dv1 + dv2
```

The coplanar assumption is an approximation. Real transfers between objects at different inclinations require a plane-change burn, which can dominate the delta-v budget. The Hohmann model is appropriate for initial feasibility screening and baseline comparison.

### Greedy Baseline

`greedy.py` implements the classical nearest-neighbor heuristic: from the current object, always transfer to the cheapest unvisited next object. `test_greedy.py` runs the solver from every possible start point and retains the best result.

For the 10-object Iridium-33 sample, the greedy solver finds a full sequence with a total delta-v of approximately **0.098 km/s**. This is the benchmark for the QUBO solver.

---

## Repository Structure

```
dextrivia/
|-- setup.py                        fetch TLEs from Celestrak
|-- propagation.py                  SGP4 propagation wrapper
|-- cost_matrix.py                  Hohmann delta-v cost matrix builder
|-- greedy.py                       greedy nearest-neighbor solver
|-- test_greedy.py                  end-to-end greedy benchmark script
|-- text_cost_matrix.py             print cost matrix to stdout for inspection
|-- new.py                          alternate TLE fetch (development scratch)
|-- debris_tles.json                cached TLE data (Iridium-33 debris)
|-- greedy_baseline.json            saved best greedy result
|-- object_names.json               ordered list of debris object names
|-- space_track_credentials.example.json  template for Space-Track credentials
|-- dextrivia-landing.html          project landing page
```

---

## Installation

**Requirements:** Python 3.11 or later.

```bash
git clone https://github.com/yourusername/dextrivia.git
cd dextrivia
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install sgp4 numpy requests
```

---

## Usage

### 1. Fetch fresh TLE data

```bash
python setup.py
```

Downloads the current Iridium-33 debris catalog from Celestrak and writes it to `debris_tles.json`.

### 2. Build the cost matrix

```bash
python text_cost_matrix.py
```

Propagates all objects to a fixed epoch (2025-01-01 12:00 UTC), computes the delta-v matrix, and saves `cost_matrix.npy` and `object_names.json`.

### 3. Run the greedy baseline

```bash
python test_greedy.py
```

Solves the sequencing problem from every start point and prints the best removal order with per-step delta-v.

**Sample output:**

```
Running greedy solver on 10 debris objects...

Best greedy solution (starting from object 7):
  Total delta-v : 0.0984 km/s

  Sequence:
    Step  1: IRIDIUM 33 DEB               <- start
    Step  2: IRIDIUM 33 DEB               +0.0043 km/s
    Step  3: IRIDIUM 33 DEB               +0.0078 km/s
    ...
```

---

## Data Sources

| Source | Description | License |
|--------|-------------|---------|
| [Celestrak](https://celestrak.org) | TLE catalog, Iridium-33 debris group | Free for non-commercial use |
| [Space-Track](https://www.space-track.org) | Official USSPACECOM catalog | Requires free account registration |

TLEs are fetched from Celestrak's `gp.php` endpoint using the `iridium-33-debris` group identifier. No API key is required for Celestrak. Space-Track credentials are supported via `space_track_credentials.json` (see the `.example.json` template; never commit the real file).

---

## Limitations and Future Work

- **Coplanar assumption:** The Hohmann model ignores inclination differences between debris objects. A full 3D transfer cost accounting for plane changes will significantly change the cost matrix.
- **Epoch sensitivity:** Debris orbits decay over time. The cost matrix is only valid near the propagation epoch. Long-horizon planning requires re-propagation at each step.
- **QUBO solver:** The greedy baseline is a placeholder. The next milestone is formulating the sequencing problem as a QUBO and solving it with a quantum annealer or simulated annealing backend (D-Wave, Qiskit, or Neal).
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
