"""Assert a `dextrivia bench` run produced a usable results directory.

    uv run python scripts/check_bench_output.py results/some-run

Used by the CI smoke test. The pytest suite covers the harness's bookkeeping in
detail; this covers the one thing it cannot, which is ``dextrivia bench`` still
working as an actual command against the committed instance family.

Deliberately a file rather than an inline ``python -c`` in the workflow: a check
nobody can run locally is a check that rots.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print(__doc__)
        return 2
    out = Path(argv[0])

    rows = list(csv.DictReader((out / "results.csv").open()))
    summary = list(csv.DictReader((out / "summary.csv").open()))
    manifest = json.loads((out / "manifest.json").read_text())

    assert rows, "bench wrote no result rows"
    assert summary, "bench wrote no summary rows"

    missed = [(r["solver"], r["reason"]) for r in rows if r["feasible"] != "True"]
    assert not missed, f"solvers unexpectedly missed on the smallest instance: {missed}"

    # N=4 is inside Held-Karp range, so every gap here is a real optimality gap.
    # If this ever says best-known, the oracle silently stopped running.
    kinds = {r["reference_kind"] for r in rows}
    assert kinds == {"exact"}, f"expected an exact oracle at N=4, got {kinds}"

    assert manifest["git"]["sha"], "manifest lost the git sha"
    assert manifest["instances"], "manifest lost the instance list"
    assert all(len(i["sha256"]) == 64 for i in manifest["instances"]), "instance hash missing"
    assert manifest["versions"]["numpy"], "manifest lost library versions"

    print(f"bench smoke ok: {len(rows)} rows, {len(summary)} summary rows, oracle present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
