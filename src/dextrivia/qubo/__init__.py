"""QUBO formulation of open-path debris sequencing, and its run bookkeeping.

Owned by the QUBO workspace. Nothing in here may import an optional backend at
module scope -- ``formulation`` has to stay usable, and tested, on an install
without the ``quantum`` extra.
"""

from __future__ import annotations

from importlib import metadata

from dextrivia.qubo.formulation import (
    DEFAULT_PENALTY_SAFETY,
    PathQUBO,
    build_qubo,
    decode,
    default_penalty,
    is_feasible,
    path_upper_bound,
    repair,
    summarize_samples,
    sweep_penalty,
)

__all__ = [
    "DEFAULT_PENALTY_SAFETY",
    "PathQUBO",
    "build_qubo",
    "decode",
    "default_penalty",
    "is_feasible",
    "path_upper_bound",
    "repair",
    "summarize_samples",
    "sweep_penalty",
    "library_versions",
]


def library_versions(*packages: str) -> dict[str, str]:
    """``{distribution: version}`` for whatever is actually installed.

    Recorded in every solver's ``Solution.metadata``. A delta-v from a sampler
    is not reproducible without it -- "simulated annealing" is not a fixed
    algorithm, it is whatever version of dwave-samplers was on the machine.
    Missing packages are reported as "not installed" rather than omitted, so a
    run record never silently loses a row.
    """
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions
