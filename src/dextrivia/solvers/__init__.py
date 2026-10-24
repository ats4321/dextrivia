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

from dextrivia.solvers.cpsat import CPSATSolver
from dextrivia.solvers.exact import BruteForceSolver, ExactSolver
from dextrivia.solvers.greedy import GreedySolver
from dextrivia.solvers.highs_mip import HiGHSSolver
from dextrivia.solvers.ortools_routing import ORToolsRoutingSolver
from dextrivia.solvers.permutation import (
    ColdPermutationAnnealingSolver,
    IteratedLocalSearchSolver,
    LocalSearchSolver,
    PermutationAnnealingSolver,
)
from dextrivia.solvers.quantum_annealing import SimulatedAnnealingSolver
from dextrivia.solvers.quantum_qaoa import QAOASolver
from dextrivia.solvers.quantum_qaoa_swap import QAOASwapSolver

#: Name -> solver class, for the CLI and benchmarks.
SOLVERS = {
    GreedySolver.name: GreedySolver,
    ExactSolver.name: ExactSolver,
    BruteForceSolver.name: BruteForceSolver,
    SimulatedAnnealingSolver.name: SimulatedAnnealingSolver,
    QAOASolver.name: QAOASolver,
    QAOASwapSolver.name: QAOASwapSolver,
    ORToolsRoutingSolver.name: ORToolsRoutingSolver,
    LocalSearchSolver.name: LocalSearchSolver,
    PermutationAnnealingSolver.name: PermutationAnnealingSolver,
    CPSATSolver.name: CPSATSolver,
    HiGHSSolver.name: HiGHSSolver,
    IteratedLocalSearchSolver.name: IteratedLocalSearchSolver,
    ColdPermutationAnnealingSolver.name: ColdPermutationAnnealingSolver,
}

#: Solvers whose result does not depend on ``seed``. ``dextrivia bench`` runs
#: these once instead of once per seed -- three identical Held-Karp runs
#: measure nothing and cost oracle time.
#: ``highs`` is here because HiGHS branch-and-bound does not take a seed; its
#: answer can still vary by machine when the time limit binds, and it records
#: that (``proven_optimal``, ``mip_status``).
DETERMINISTIC = frozenset(
    {GreedySolver.name, ExactSolver.name, BruteForceSolver.name, HiGHSSolver.name}
)

__all__ = [
    "GreedySolver",
    "DETERMINISTIC",
    "ExactSolver",
    "BruteForceSolver",
    "SimulatedAnnealingSolver",
    "QAOASolver",
    "QAOASwapSolver",
    "ORToolsRoutingSolver",
    "LocalSearchSolver",
    "PermutationAnnealingSolver",
    "CPSATSolver",
    "HiGHSSolver",
    "IteratedLocalSearchSolver",
    "ColdPermutationAnnealingSolver",
    "SOLVERS",
]
