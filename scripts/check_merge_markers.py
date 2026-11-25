"""Fail when a tracked file still carries a merge-conflict marker.

A file with ``<<<<<<<`` or ``>>>>>>>`` is a merge someone stopped halfway; git
still applies the patterns inside it, and nothing else reads it. The check is
part of ``make check`` and ``make test``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

MARKERS = ("<<<<<<< ", ">>>>>>> ")


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(path) for path in raw.decode().split("\0") if path]


def main() -> int:
    bad: list[str] = []
    for path in tracked_files():
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.startswith(MARKERS):
                bad.append(f"{path}:{line_number}: {line[:80]}")
    if bad:
        print("merge-conflict markers found:")
        for finding in bad:
            print(f"  {finding}")
        return 1
    print("merge markers: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
