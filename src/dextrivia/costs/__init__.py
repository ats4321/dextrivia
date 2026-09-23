"""Cost models: catalogue objects -> delta-v arrays.

OWNED BY THE PHYSICS WORKSPACE. Anything added here must implement
``dextrivia.core.CostModel`` and return either (N, N) or (N-1, N, N) km/s.
"""

from dextrivia.costs.hohmann import HohmannCostModel, hohmann_dv

__all__ = ["HohmannCostModel", "hohmann_dv"]
