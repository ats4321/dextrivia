"""Cost models: catalogue objects -> delta-v arrays.

OWNED BY THE PHYSICS WORKSPACE. Anything added here must implement
``dextrivia.core.CostModel`` and return either (N, N) or (N-1, N, N) km/s.

``hohmann``    altitude-only baseline. Degenerate by construction; kept as the
               control the plane-aware models are measured against.
``realistic``  plane-aware impulsive and low-thrust Edelbaum models, with
               optional J2 nodal drift between legs.
``selection``  ``plane-cluster`` target selection, so instances are missions a
               servicer could actually fly.
"""

from dextrivia.costs.hohmann import HohmannCostModel, hohmann_dv
from dextrivia.costs.realistic import (
    EdelbaumCostModel,
    ImpulsiveCostModel,
    edelbaum_dv,
    impulsive_dv,
    nodal_precession_deg_per_day,
    plane_angle,
    separate_burn_dv,
)
from dextrivia.costs.selection import (
    ClusterWindow,
    build_cluster_instance,
    select_plane_cluster,
)

#: Name -> zero-argument factory, for scripts and benchmarks.
COST_MODELS = {
    HohmannCostModel.name: HohmannCostModel,
    ImpulsiveCostModel.base_name: ImpulsiveCostModel,
    EdelbaumCostModel.base_name: EdelbaumCostModel,
}

__all__ = [
    "COST_MODELS",
    "ClusterWindow",
    "EdelbaumCostModel",
    "HohmannCostModel",
    "ImpulsiveCostModel",
    "build_cluster_instance",
    "edelbaum_dv",
    "hohmann_dv",
    "impulsive_dv",
    "nodal_precession_deg_per_day",
    "plane_angle",
    "select_plane_cluster",
    "separate_burn_dv",
]
