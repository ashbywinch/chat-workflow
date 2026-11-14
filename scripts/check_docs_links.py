#!/usr/bin/env python3
"""Check that every relative markdown link in the docs set resolves.

The docs set is `README.md`, `AGENTS.md`, and every `docs/**/*.md`. Absolute
URLs, in-page anchors, and links containing `=` or spaces are skipped. A link
that escapes the repo (e.g. into `.sisyphus/`) is a failure: the target would
not exist for anyone reading the committed tree.

The repo root is derived from this file's location (`scripts/` sits at the
root), so the check keeps working when test modules move. Run via
`make docs-check`; `tests/unit/test_docs_links.py` calls `find_broken_links()`
so `make test` covers it too.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def doc_files() -> list[Path]:
    return [
        REPO_ROOT / "README.md",
        REPO_ROOT / "AGENTS.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
    ]


def _relative_links(md_file: Path) -> Iterator[tuple[str, Path]]:
    text = md_file.read_text(encoding="utf-8")
    for match in LINK_RE.finditer(text):
        url = match.group(2).strip()
        if url.startswith("http://") or url.startswith("https://") or url.startswith("#"):
            continue
        if "=" in url or " " in url:
            continue
        if "#" in url:
            url = url[: url.index("#")]
        yield match.group(1), (md_file.parent / url).resolve()


def find_broken_links() -> list[str]:
    """Return one message per broken link; empty means every link resolves."""
    failures: list[str] = []
    for md_file in doc_files():
        for link_text, target in _relative_links(md_file):
            rel_source = md_file.relative_to(REPO_ROOT)
            if not target.is_relative_to(REPO_ROOT):
                failures.append(f"{rel_source}: link '{link_text}' -> {target} (outside repo)")
            elif not target.exists():
                failures.append(f"{rel_source}: link '{link_text}' -> {target.relative_to(REPO_ROOT)}")
    return failures


def main() -> int:
    failures = find_broken_links()
    if failures:
        print("Broken relative links:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"docs-check: {len(doc_files())} files, every relative link resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
