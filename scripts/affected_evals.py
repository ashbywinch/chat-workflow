#!/usr/bin/env python3
"""Find eval files affected by source changes, using the code-review-graph DB.

Uses transitive dependency tracking via BFS on the graph's IMPORTS_FROM
edges so that changing a deeply imported module (e.g. ``llm_interaction.py``)
correctly triggers evals that depend on it through intermediate modules.

Usage:
    python scripts/affected_evals.py                      # diff against origin/main
    python scripts/affected_evals.py --git-base HEAD      # no changes -> empty
    python scripts/affected_evals.py --git-base HEAD~5    # last 5 commits
    python scripts/affected_evals.py --list               # show full dep map
    python scripts/affected_evals.py --verbose            # human-readable output
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DB = PROJECT_ROOT / ".code-review-graph" / "graph.db"
EVAL_DIR = PROJECT_ROOT / "tests" / "evals"

# Files whose changes trigger ALL evals (they're imported by everything).
UNIVERSAL_DEPS = {
    "tests/evals/helpers.py",
    "tests/conftest.py",
    "tests/__init__.py",
    "tests/evals/__init__.py",
}


def get_changed_files(git_base: str) -> list[str]:
    """Get list of changed file paths (relative to project root) since *git_base*."""
    result = subprocess.run(
        ["git", "diff", "--name-only", git_base],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"Error: git diff failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return [f for f in result.stdout.strip().split("\n") if f]


def get_all_eval_files() -> list[Path]:
    """Return all ``test_*.py`` files in the evals directory tree."""
    return sorted(EVAL_DIR.rglob("test_*.py"))


def _file_for_qualified(cursor, qualified: str) -> str | None:
    """Return the file path for a qualified node name, or None."""
    cursor.execute(
        "SELECT file_path FROM nodes WHERE qualified_name = ? LIMIT 1",
        (qualified,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def get_transitive_dep_map() -> dict[str, set[str]]:
    """Build ``{eval_relpath: set(source_relpaths)}`` via BFS on the graph.

    For each eval file, follows ``IMPORTS_FROM`` edges transitively so
    that a change to ``llm_interaction.py`` triggers evals that import
    ``atomic_workflow.py`` (which imports ``llm_interaction.py``).
    """
    if not GRAPH_DB.exists():
        return _fallback_ast_deps()

    conn = sqlite3.connect(str(GRAPH_DB))
    cursor = conn.cursor()

    # Find all eval test files known to the graph
    cursor.execute(
        "SELECT DISTINCT file_path FROM nodes "
        "WHERE file_path LIKE '%tests/evals/%test_%' AND kind = 'Test'"
    )
    eval_paths = {row[0] for row in cursor.fetchall()}

    dep_map: dict[str, set[str]] = {}

    for eval_path in sorted(eval_paths):
        rel = os.path.relpath(eval_path, str(PROJECT_ROOT))

        # BFS from every node in this eval file
        cursor.execute(
            "SELECT qualified_name FROM nodes WHERE file_path = ?",
            (eval_path,),
        )
        queue: deque[str] = deque()
        for (qn,) in cursor.fetchall():
            if qn:
                queue.append(qn)

        visited: set[str] = set()
        sources: set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Follow IMPORTS_FROM edges outward
            cursor.execute(
                "SELECT DISTINCT e.target_qualified FROM edges e "
                "WHERE e.source_qualified = ? AND e.kind = 'IMPORTS_FROM'",
                (current,),
            )
            for (target,) in cursor.fetchall():
                if target not in visited:
                    queue.append(target)
                target_file = _file_for_qualified(cursor, target)
                if target_file:
                    target_rel = os.path.relpath(target_file, str(PROJECT_ROOT))
                    sources.add(target_rel)

        if sources:
            dep_map[rel] = sources

    conn.close()
    return dep_map


def _fallback_ast_deps() -> dict[str, set[str]]:
    """Fallback: parse import statements from eval files (no graph DB)."""
    import ast

    dep_map: dict[str, set[str]] = {}
    for ef in get_all_eval_files():
        rel = os.path.relpath(str(ef), str(PROJECT_ROOT))
        try:
            tree = ast.parse(ef.read_text())
            sources: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        sources.add(alias.name.replace(".", "/"))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    sources.add(node.module.replace(".", "/"))
            if sources:
                dep_map[rel] = sources
        except SyntaxError:
            pass
    return dep_map


def find_affected_evals(
    changed_files: list[str],
    dep_map: dict[str, set[str]],
) -> list[str]:
    """Return sorted list of eval file paths affected by *changed_files*.

    An eval is affected if any of its transitive source dependencies
    matches a changed file, or if the eval file itself was changed.
    Universal deps (helpers.py, conftest.py) always trigger all evals.
    """
    if not changed_files:
        return []

    # Collect all eval paths once
    all_evals = sorted(dep_map.keys())

    # Check universal deps first
    for cf in changed_files:
        if cf in UNIVERSAL_DEPS or cf.startswith("tests/evals/"):
            return all_evals

    affected: set[str] = set()

    # If an eval file itself changed, always include it
    for cf in changed_files:
        for ev in all_evals:
            if cf == ev:
                affected.add(ev)

    # Check transitive dependency matches
    for cf in changed_files:
        for eval_path, sources in dep_map.items():
            if eval_path in affected:
                continue  # already included
            for src in sources:
                if cf == src or cf.startswith(src.rstrip(".py").rsplit("/", 1)[0] + "/"):
                    affected.add(eval_path)
                    break

    return sorted(affected)


def show_dependency_map(dep_map: dict[str, set[str]]) -> None:
    """Pretty-print the transitive dependency map."""
    for eval_path in sorted(dep_map):
        sources = dep_map[eval_path]
        print(f"{eval_path}")
        for src in sorted(sources):
            print(f"  <- {src}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Find eval files affected by source changes (via code-review-graph)"
    )
    parser.add_argument(
        "--git-base", default="origin/main",
        help="Git ref to diff against (default: origin/main)",
    )
    parser.add_argument("--verbose", action="store_true", help="Human-readable output")
    parser.add_argument("--list", action="store_true", help="Show full dependency map")
    args = parser.parse_args()

    if args.list:
        dep_map = get_transitive_dep_map()
        show_dependency_map(dep_map)
        return

    changed_files = get_changed_files(args.git_base)
    dep_map = get_transitive_dep_map()

    if args.verbose:
        print(f"Changed files ({len(changed_files)}):")
        for cf in changed_files:
            print(f"  {cf}")
        print()

    affected = find_affected_evals(changed_files, dep_map)

    if args.verbose:
        if affected:
            print(f"Affected evals ({len(affected)}):")
            for af in affected:
                print(f"  {af}")
        else:
            print("No evals affected by current changes.")
    else:
        if affected:
            print(" ".join(affected))


if __name__ == "__main__":
    main()
