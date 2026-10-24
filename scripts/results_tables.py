"""Print the README's results tables from committed results directories.

    uv run python scripts/results_tables.py results/canonical results/canonical-v2 ...

Every number in the README's results tables is printed by this script from a
``summary.csv``. Nothing is typed by hand, so a table and its CSV cannot drift
apart: re-run this and paste.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

#: The solvers that answer "is there headroom, and is the reference proven?".
#: qaoa / qaoa-swap / ortools / exact / brute get their own table or a footnote:
#: they refuse most of these instances, and a column of "refused" says nothing.
MAIN_SOLVERS = (
    "greedy",
    "localsearch",
    "ils",
    "sa-perm",
    "sa-perm-cold",
    "cpsat",
    "highs",
    "sa-qubo",
)
SIMULATED_QUANTUM = ("qaoa", "qaoa-swap")


def _f(value: str | None) -> float | None:
    return None if value in ("", None) else float(value)


def _order(label: str) -> tuple[int, int, int]:
    n, variant = label.split("_", 1)
    delta = 0 if variant == "static" else int(variant[2:-1])
    return (int(n[1:]), variant != "static", delta)


def gap_cell(row: dict[str, str] | None) -> str:
    if row is None or not int(row["feasible_runs"]):
        return "refused"
    mean, sd = _f(row["gap_mean_pct"]), _f(row["gap_std_pct"]) or 0.0
    if mean is None:
        return "-"
    if abs(mean) < 0.005:
        mean = 0.0  # float round-off either side of the reference, not a result
    cell = f"{mean:+.2f}%"
    if int(row["runs"]) > 1 and sd >= 0.005:
        cell += f" ± {sd:.2f}"
    if int(row["runs"]) == 1 and row["solver"] == "cpsat":
        cell += " (1 seed)"
    return f"**{cell}**" if mean >= 0.005 and row["solver"] == "localsearch" else cell


def reference_cells(head: dict[str, str]) -> tuple[str, str]:
    kind = head["reference_kind"]
    bound, gap = _f(head["reference_lower_bound_kms"]), _f(head["reference_bound_gap_pct"])
    if kind == "exact":
        return "*exact*", "Held-Karp"
    shown = "-" if bound is None else f"{bound:.4f}"
    by = head["reference_bound_solver"] or "-"
    if kind == "proven":
        return "**proven**", f"{shown} ({by})"
    if gap is None:
        return "best-known", shown
    gap_text = f"{gap:.2f}%" if gap >= 0.01 else f"{gap:.5f}%"
    return "best-known", f"{shown} ({by}), gap {gap_text}"


def main_table(directory: Path) -> str:
    rows = list(csv.DictReader((directory / "summary.csv").open()))
    by = {(r["instance"], r["solver"]): r for r in rows}
    labels = sorted({r["instance"] for r in rows}, key=_order)
    header = "| instance | reference (km/s) | kind | certified bound | " + " | ".join(
        f"`{s}`" for s in MAIN_SOLVERS
    )
    lines = [header + " |", "|" + "---|" * (4 + len(MAIN_SOLVERS))]
    for label in labels:
        head = by[(label, "greedy")]
        kind, bound = reference_cells(head)
        cells = [gap_cell(by.get((label, s))) for s in MAIN_SOLVERS]
        lines.append(
            f"| `{label}` | {_f(head['reference_dv_kms']):.4f} | {kind} | {bound} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def quantum_table(directories: list[Path]) -> str:
    lines = [
        "| family | instance | solver | gap (mean ± sd) | runtime (mean ± sd) | runtime kind |"
        " feasibility |",
        "|---|---|---|---|---|---|---|",
    ]
    for directory in directories:
        rows = list(csv.DictReader((directory / "summary.csv").open()))
        config = json.loads((directory / "manifest.json").read_text())["solvers"]
        for r in sorted(rows, key=lambda r: (_order(r["instance"]), r["solver"])):
            if r["solver"] not in SIMULATED_QUANTUM or not int(r["feasible_runs"]):
                continue
            rt, rsd = _f(r["runtime_mean_s"]), _f(r["runtime_std_s"])
            feas = _f(r["feasibility_rate_mean"])
            # qaoa-swap's subspace backend cannot leave the feasible set, so its
            # 1.000 is structural, not measured. The bench does not capture that
            # flag (it reads feasibility_is_structural, the solver writes
            # feasibility_measured), so it is taken from the run's own config.
            backend = config.get(r["solver"], {}).get("backend")
            structural = r["solver"] == "qaoa-swap" and backend == "subspace"
            lines.append(
                f"| {directory.name} | `{r['instance']}` | `{r['solver']}` | {gap_cell(r)} "
                f"| {rt:.2f} s ± {rsd:.2f} | {r['runtime_note']} "
                f"| {'-' if feas is None else f'{feas:.3f}'}"
                f"{' (structural, subspace backend)' if structural else ' (measured)'} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    dirs = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not dirs:
        print(__doc__)
        return 2
    for directory in dirs:
        print(f"### `{directory}`\n")
        print(main_table(directory))
        print()
    print("### Simulated quantum solvers, where they run\n")
    print(quantum_table(dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
