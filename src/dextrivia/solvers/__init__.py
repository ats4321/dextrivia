"""Solvers. Every one implements ``dextrivia.core.Solver``.

Classical solvers live here. Quantum / quantum-inspired solvers go in
``dextrivia/solvers/quantum*.py`` and are OWNED BY THE QUBO WORKSPACE.
"""

from dextrivia.solvers.exact import BruteForceSolver, ExactSolver
from dextrivia.solvers.greedy import GreedySolver

#: Name -> solver class, for the CLI and benchmarks.
SOLVERS = {
    GreedySolver.name: GreedySolver,
    ExactSolver.name: ExactSolver,
    BruteForceSolver.name: BruteForceSolver,
}

__all__ = ["GreedySolver", "ExactSolver", "BruteForceSolver", "SOLVERS"]
