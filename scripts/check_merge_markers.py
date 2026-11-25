"""Fail when a tracked file still carries a merge-conflict marker.

A file with ``<<<<<<<`` or ``>>>>>>>`` is a merge someone stopped halfway; git
still applies the patterns inside it, and nothing else reads it. The check is
part of ``make check`` and ``make test``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class MergeMarker:
    """The leading fragments git writes into a half-merged file."""

    PREFIXES = ("<<<<<<< ", ">>>>>>> ")


def tracked_files() -> list[Path]:
    """Every file git tracks in the current checkout."""
    raw = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(path) for path in raw.decode().split("\0") if path]


def _conflict_lines(path: Path) -> list[str]:
    """``path:line: text`` for each marker line in *path*.

    Binary and unreadable files carry no markers to find.
    """
    try:
        lines = path.read_text().splitlines()
    except (UnicodeDecodeError, OSError):
        return []
    return [
        f"{path}:{line_number}: {line[:80]}"
        for line_number, line in enumerate(lines, 1)
        if line.startswith(MergeMarker.PREFIXES)
    ]


def main() -> int:
    bad = [marker for path in tracked_files() for marker in _conflict_lines(path)]
    if bad:
        print("merge-conflict markers found:")
        for finding in bad:
            print(f"  {finding}")
        return 1
    print("merge markers: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
