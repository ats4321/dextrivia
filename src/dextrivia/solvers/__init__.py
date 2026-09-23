"""Solvers. Every one implements ``dextrivia.core.Solver``.

Classical solvers live here. Quantum / quantum-inspired solvers go in
``dextrivia/solvers/quantum*.py`` and are OWNED BY THE QUBO WORKSPACE, as is
``ortools_routing.py`` -- see the QUBO section of CLAUDE.md for why a
non-quantum file ended up under that ownership.

The QUBO-workspace modules import their optional backends inside ``solve()``,
never at module scope, so this package stays importable -- and the CLI stays
usable -- on an install without the ``quantum`` extra. A solver whose backend
is missing reports ``feasible=False`` at solve time like any other miss.
"""

from dextrivia.solvers.exact import BruteForceSolver, ExactSolver
from dextrivia.solvers.greedy import GreedySolver
from dextrivia.solvers.ortools_routing import ORToolsRoutingSolver
from dextrivia.solvers.quantum_annealing import SimulatedAnnealingSolver
from dextrivia.solvers.quantum_qaoa import QAOASolver

#: Name -> solver class, for the CLI and benchmarks.
SOLVERS = {
    GreedySolver.name: GreedySolver,
    ExactSolver.name: ExactSolver,
    BruteForceSolver.name: BruteForceSolver,
    SimulatedAnnealingSolver.name: SimulatedAnnealingSolver,
    QAOASolver.name: QAOASolver,
    ORToolsRoutingSolver.name: ORToolsRoutingSolver,
}

__all__ = [
    "GreedySolver",
    "ExactSolver",
    "BruteForceSolver",
    "SimulatedAnnealingSolver",
    "QAOASolver",
    "ORToolsRoutingSolver",
    "SOLVERS",
]
