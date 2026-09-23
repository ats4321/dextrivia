"""Snapshot selection and the never-overwrite guarantee."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dextrivia.snapshots import _new_snapshot_path, _parse_tle_text

TLE_TEXT = """IRIDIUM 33
1 24946U 97051C   26092.50220897  .00000345  00000+0  11402-3 0  9991
2 24946  86.3921  21.7103 0006794 198.8367 161.2579 14.35115739494230
IRIDIUM 33 DEB
1 33773U 97051L   26092.49673196  .00001217  00000+0  35370-3 0  9990
2 33773  86.4063  13.6419 0012229 137.6859 222.5288 14.43534418899452
"""


def test_parsing_recovers_norad_ids_not_just_names():
    fetched = datetime(2026, 4, 2, tzinfo=UTC)
    payload = _parse_tle_text(TLE_TEXT, source="test", group="g", fetched=fetched)
    assert [o["norad_id"] for o in payload["objects"]] == [24946, 33773]
    assert payload["fetched_utc"] == fetched.isoformat()


def test_truncated_tle_text_is_rejected():
    with pytest.raises(ValueError, match="3-line TLE records"):
        _parse_tle_text(
            "\n".join(TLE_TEXT.splitlines()[:5]),
            source="test",
            group="g",
            fetched=datetime(2026, 4, 2, tzinfo=UTC),
        )


def test_fetching_twice_in_a_day_does_not_overwrite(tmp_path):
    fetched = datetime(2026, 4, 2, 18, 30, 0, tzinfo=UTC)
    first = _new_snapshot_path(tmp_path, "iridium-33-debris", fetched)
    assert first.name == "iridium33_20260402.json"
    first.write_text("{}", encoding="utf-8")
    second = _new_snapshot_path(tmp_path, "iridium-33-debris", fetched)
    assert second != first and not second.exists()


def test_selection_rules(snapshot):
    assert snapshot.select(5) == snapshot.objects[:5]
    assert snapshot.select(5, rule="random", seed=7) == snapshot.select(5, rule="random", seed=7)
    assert snapshot.select(5, rule="random", seed=7) != snapshot.select(5, rule="random", seed=8)
    with pytest.raises(ValueError, match="requires a seed"):
        snapshot.select(5, rule="random")
    with pytest.raises(ValueError, match="unknown selection rule"):
        snapshot.select(5, rule="cheapest")
    with pytest.raises(ValueError, match="n must be in"):
        snapshot.select(500)


def test_instance_metadata_records_everything_needed_to_reproduce(snapshot):
    from dextrivia.instances import build_instance

    inst = build_instance(snapshot, n=4, rule="random", seed=3)
    meta = inst.metadata
    assert meta["snapshot"] == snapshot.path.name
    assert (meta["selection_rule"], meta["selection_n"], meta["selection_seed"]) == ("random", 4, 3)
    assert meta["cost_model"] == "hohmann-coplanar"
    assert meta["epoch_source"] == "snapshot-median-tle-epoch"
    assert inst.epoch == snapshot.median_epoch()
