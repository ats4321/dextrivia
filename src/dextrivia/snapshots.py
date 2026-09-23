"""TLE snapshots: fetch, load, select.

A snapshot is an immutable, timestamped JSON file under ``data/snapshots/``.
Fetching always writes a NEW file and never overwrites an existing one, because
a benchmark number is meaningless without the exact TLE set that produced it.
Benchmarks reference a snapshot by filename.

Schema::

    {
      "source": "<url the data came from>",
      "group": "iridium-33-debris",
      "fetched_utc": "2026-04-02T18:11:24+00:00",   // null if unknown
      "fetched_utc_note": "optional: how the timestamp was determined",
      "objects": [{"norad_id": 24946, "name": "...", "line1": "...", "line2": "..."}]
    }
"""

from __future__ import annotations

import json
import random
import re
import statistics
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dextrivia.core import DebrisObject
from dextrivia.propagation import satrec, tle_epoch

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
DEFAULT_GROUP = "iridium-33-debris"
# Celestrak serves /pub/TLE/*.txt as 403 to unknown agents; gp.php + a real UA works.
USER_AGENT = "dextrivia/0.1 (orbital debris sequencing research; +https://celestrak.org/)"

SELECTION_RULES = ("first", "random")

#: ``<stem>_<YYYYMMDD>[suffix].json``. The optional suffix is what
#: ``_new_snapshot_path`` adds when a second snapshot lands on the same day.
SNAPSHOT_NAME_RE = re.compile(r"(?P<stem>[a-z0-9]+)_(?P<date>\d{8})(?P<suffix>.*)")

__all__ = [
    "Snapshot",
    "SELECTION_RULES",
    "fetch_snapshot",
    "default_snapshot_dir",
    "group_stem",
    "parse_snapshot_name",
]


def default_snapshot_dir() -> Path:
    """``<repo>/data/snapshots``, resolved relative to the installed package."""
    return Path(__file__).resolve().parents[2] / "data" / "snapshots"


def group_stem(group: str) -> str:
    """The filename token a Celestrak group maps to: ``iridium-33-debris`` -> ``iridium33``."""
    return group.replace("-debris", "").replace("-", "")


def parse_snapshot_name(path: str | Path) -> tuple[str, str, str] | None:
    """``(stem, YYYYMMDD, suffix)`` from a snapshot filename, or None if it does not match.

    The date comes from the NAME, not from ``fetched_utc``: the committed legacy
    snapshot has a null ``fetched_utc`` because its real download time was never
    recorded, and ordering snapshots must still work.
    """
    match = SNAPSHOT_NAME_RE.fullmatch(Path(path).stem)
    if match is None:
        return None
    return match["stem"], match["date"], match["suffix"]


@dataclass(frozen=True)
class Snapshot:
    """An on-disk TLE snapshot, loaded into memory."""

    path: Path
    source: str
    group: str
    #: None when the snapshot predates `dextrivia fetch` and the real download
    #: time was never recorded. See ``fetched_utc_note`` in the file.
    fetched_utc: datetime | None
    objects: tuple[DebrisObject, ...]

    @classmethod
    def load(cls, path: str | Path) -> Snapshot:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        fetched = raw.get("fetched_utc")
        return cls(
            path=path,
            source=raw["source"],
            group=raw.get("group", ""),
            fetched_utc=datetime.fromisoformat(fetched) if fetched else None,
            objects=tuple(
                DebrisObject(
                    norad_id=int(o["norad_id"]),
                    name=o["name"],
                    line1=o["line1"],
                    line2=o["line2"],
                )
                for o in raw["objects"]
            ),
        )

    def __len__(self) -> int:
        return len(self.objects)

    def epochs(self) -> list[datetime]:
        """Each object's own TLE epoch (they differ by hours to days)."""
        return [tle_epoch(o.line1, o.line2) for o in self.objects]

    def median_epoch(self) -> datetime:
        """Median TLE epoch across the snapshot.

        The default propagation epoch. Propagating to a date far from the TLE
        epochs is the single easiest way to get confidently wrong altitudes, so
        "the middle of the data we actually have" is the honest default.
        """
        stamps = sorted(e.timestamp() for e in self.epochs())
        return datetime.fromtimestamp(statistics.median(stamps), tz=UTC)

    def select(
        self, n: int, rule: str = "first", seed: int | None = None
    ) -> tuple[DebrisObject, ...]:
        """Pick ``n`` objects by an explicit, recorded rule.

        ``first``  the first n objects in snapshot (catalogue) order.
        ``random`` a seeded sample; ``seed`` is required so it is reproducible.
        """
        if not 2 <= n <= len(self.objects):
            raise ValueError(f"n must be in 2..{len(self.objects)}, got {n}")
        if rule == "first":
            return self.objects[:n]
        if rule == "random":
            if seed is None:
                raise ValueError("selection rule 'random' requires a seed")
            return tuple(random.Random(seed).sample(list(self.objects), n))
        raise ValueError(f"unknown selection rule {rule!r}, expected one of {SELECTION_RULES}")


def _parse_tle_text(text: str, source: str, group: str, fetched: datetime) -> dict:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) % 3:
        raise ValueError(f"expected 3-line TLE records, got {len(lines)} lines")
    objects = []
    for i in range(0, len(lines), 3):
        name, line1, line2 = lines[i : i + 3]
        objects.append(
            {
                "norad_id": int(satrec(line1, line2).satnum),
                "name": name,
                "line1": line1,
                "line2": line2,
            }
        )
    return {
        "source": source,
        "group": group,
        "fetched_utc": fetched.isoformat(),
        "objects": objects,
    }


def _new_snapshot_path(out_dir: Path, group: str, fetched: datetime) -> Path:
    """A path that does not exist yet. Snapshots are never overwritten."""
    stem = group_stem(group)
    path = out_dir / f"{stem}_{fetched:%Y%m%d}.json"
    if path.exists():
        path = out_dir / f"{stem}_{fetched:%Y%m%dT%H%M%SZ}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot {path}")
    return path


def fetch_snapshot(group: str = DEFAULT_GROUP, out_dir: Path | None = None) -> Path:
    """Download a TLE group from Celestrak into a new snapshot file."""
    out_dir = Path(out_dir) if out_dir else default_snapshot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    url = CELESTRAK_GP.format(group=group)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    fetched = datetime.now(UTC)
    payload = _parse_tle_text(text, source=url, group=group, fetched=fetched)
    path = _new_snapshot_path(out_dir, group, fetched)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
