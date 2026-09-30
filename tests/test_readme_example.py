"""The README's quoted CLI example must still print the number it quotes.

A fresh snapshot became the ``dextrivia build`` default on 2026-09-28, which
would have silently changed every unpinned example. This runs the README block
as written (build output redirected into ``tmp_path``) and compares.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from dextrivia.cli import main

README = Path(__file__).resolve().parents[1] / "README.md"


def quoted_example() -> tuple[list[list[str]], str]:
    blocks = re.findall(r"```bash\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    block = next(b for b in blocks if "dextrivia solve" in b)
    commands = [
        shlex.split(line)[2:]  # drop "uv run"
        for line in block.splitlines()
        if line.startswith("uv run dextrivia ")
    ]
    quoted = re.search(r"# total dv\s+([0-9.]+) km/s", block)
    assert quoted, "the README example no longer quotes its output"
    return commands, quoted.group(1)


def test_readme_solve_example_prints_the_quoted_total(tmp_path, capsys):
    commands, quoted = quoted_example()
    assert [c[1] for c in commands] == ["build", "solve"]
    assert "--snapshot" in commands[0], "the example must pin its snapshot"

    build, solve = commands
    index = solve.index("--instance") + 1
    instance = tmp_path / Path(solve[index]).name
    solve[index] = str(instance)

    assert main([*build[1:], "--out", str(instance)]) == 0
    capsys.readouterr()
    assert main(solve[1:]) == 0
    printed = re.search(r"total dv\s+([0-9.]+) km/s", capsys.readouterr().out)
    assert printed and printed.group(1) == quoted
