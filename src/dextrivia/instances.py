"""Assembly: snapshot + selection + cost model -> ``ProblemInstance``.

This is the one place that decides what goes into ``metadata``, and therefore
the one place that decides whether a benchmark number can be reproduced later.
"""

from __future__ import annotations

from datetime import datetime

from dextrivia.core import CostModel, ProblemInstance
from dextrivia.costs.hohmann import HohmannCostModel
from dextrivia.snapshots import Snapshot

__all__ = ["build_instance"]


def build_instance(
    snapshot: Snapshot,
    n: int = 10,
    rule: str = "first",
    seed: int | None = None,
    epoch: datetime | None = None,
    cost_model: CostModel | None = None,
) -> ProblemInstance:
    """Build an instance from ``n`` objects of ``snapshot``.

    ``epoch`` defaults to the snapshot's median TLE epoch. There is deliberately
    no hardcoded calendar date anywhere in this package: propagating months away
    from the TLE epochs silently degrades every altitude.
    """
    cost_model = cost_model or HohmannCostModel()
    objects = snapshot.select(n, rule=rule, seed=seed)
    epoch_source = "snapshot-median-tle-epoch" if epoch is None else "caller"
    epoch = epoch or snapshot.median_epoch()
    costs = cost_model.build(objects, epoch)
    return ProblemInstance(
        norad_ids=tuple(o.norad_id for o in objects),
        names=tuple(o.name for o in objects),
        epoch=epoch,
        costs=costs,
        metadata={
            "snapshot": snapshot.path.name,
            "snapshot_fetched_utc": (
                snapshot.fetched_utc.isoformat() if snapshot.fetched_utc else None
            ),
            "snapshot_size": len(snapshot),
            "selection_rule": rule,
            "selection_n": n,
            "selection_seed": seed,
            "cost_model": cost_model.name,
            "epoch_source": epoch_source,
        },
    )
