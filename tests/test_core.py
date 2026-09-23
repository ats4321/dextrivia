"""ProblemInstance contract: shapes, open-path cost accounting, round-tripping."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from dextrivia.core import ProblemInstance, Solution

EPOCH = datetime(2026, 4, 2, tzinfo=UTC)


def instance(costs: np.ndarray, n: int = 3) -> ProblemInstance:
    return ProblemInstance(
        norad_ids=tuple(range(100, 100 + n)),
        names=tuple("IRIDIUM 33 DEB" for _ in range(n)),
        epoch=EPOCH,
        costs=costs,
    )


def test_static_costs_are_shape_n_by_n():
    inst = instance(np.zeros((3, 3)))
    assert inst.n == 3
    assert not inst.time_dependent
    assert inst.leg_costs(0).shape == (3, 3)


def test_time_slotted_costs_have_one_slot_per_leg():
    """An open path over N objects has N-1 legs, hence N-1 cost slots."""
    inst = instance(np.zeros((2, 3, 3)))
    assert inst.time_dependent
    with pytest.raises(IndexError):
        inst.leg_costs(2)


def test_wrong_cost_shape_is_rejected():
    with pytest.raises(ValueError, match="costs must have shape"):
        instance(np.zeros((3, 3, 3)))


def test_naive_epoch_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        ProblemInstance(
            norad_ids=(1, 2),
            names=("a", "b"),
            epoch=datetime(2026, 4, 2),  # noqa: DTZ001
            costs=np.zeros((2, 2)),
        )


def test_duplicate_norad_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        ProblemInstance(norad_ids=(1, 1), names=("a", "b"), epoch=EPOCH, costs=np.zeros((2, 2)))


def test_path_cost_is_open_and_has_no_return_leg():
    costs = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 2.0], [5.0, 2.0, 0.0]])
    inst = instance(costs)
    # 0 -> 1 -> 2 costs 1 + 2. A closed tour would add the 5.0 return leg.
    assert inst.path_cost([0, 1, 2]) == pytest.approx(3.0)


def test_path_cost_uses_the_slot_matching_the_position_in_sequence():
    costs = np.zeros((2, 3, 3))
    costs[0, 0, 1] = 7.0  # leg flown first
    costs[1, 0, 1] = 100.0  # same transfer, flown second
    inst = instance(costs)
    assert inst.path_cost([0, 1, 2]) == pytest.approx(7.0)
    assert inst.path_cost([2, 0, 1]) == pytest.approx(100.0)


@pytest.mark.parametrize("shape", [(4, 4), (3, 4, 4)])
def test_save_load_roundtrip(tmp_path, shape):
    rng = np.random.default_rng(0)
    inst = ProblemInstance(
        norad_ids=(24946, 33773, 33775, 33776),
        names=("IRIDIUM 33", "DEB", "DEB", "DEB"),
        epoch=EPOCH,
        costs=rng.random(shape),
        metadata={"snapshot": "iridium33_20260402.json", "selection_seed": None},
    )
    loaded = ProblemInstance.load(inst.save(tmp_path / "i.npz"))
    assert loaded.norad_ids == inst.norad_ids
    assert loaded.names == inst.names
    assert loaded.epoch == inst.epoch
    assert loaded.metadata == inst.metadata
    np.testing.assert_allclose(loaded.costs, inst.costs)


@pytest.mark.parametrize("name", ["i.npz", "i", "i.instance"])
def test_save_normalises_the_npz_suffix_and_returns_the_written_path(tmp_path, name):
    """np.savez appends .npz itself, so save() must return the path it really wrote."""
    inst = instance(np.zeros((3, 3)))
    written = inst.save(tmp_path / name)
    assert written.suffix == ".npz"
    assert written.exists()
    assert ProblemInstance.load(written).norad_ids == inst.norad_ids


def test_solution_maps_indices_back_to_norad_ids():
    inst = instance(np.zeros((3, 3)))
    solution = Solution(
        sequence=(2, 0, 1),
        total_dv_kms=0.0,
        runtime_s=0.0,
        solver_name="test",
        feasible=True,
    )
    assert solution.norad_order(inst) == (102, 100, 101)
