#!/usr/bin/env python3
"""Find eval files affected by source changes, using the code-review-graph DB.

Usage:
    python scripts/affected_evals.py                      # diff against origin/main
    python scripts/affected_evals.py --git-base HEAD      # no changes → empty
    python scripts/affected_evals.py --git-base HEAD~5    # last 5 commits
    python scripts/affected_evals.py --list               # show full dependency map
    python scripts/affected_evals.py --verbose            # human-readable output
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DB = PROJECT_ROOT / ".code-review-graph" / "graph.db"
EVAL_DIR = PROJECT_ROOT / "tests" / "evals"

# Files that affect ALL evals when changed
UNIVERSAL_DEPS = {
    "tests/evals/helpers.py",
    "tests/conftest.py",
    "tests/__init__.py",
}


def get_changed_files(git_base: str) -> list[str]:
    """Get list of changed file paths (relative to project root) since git_base."""
    result = subprocess.run(
        ["git", "diff", "--name-only", git_base],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"Error: git diff failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return [f for f in result.stdout.strip().split("\n") if f]


def get_all_eval_files() -> list[Path]:
    """Return all test_*.py files in the evals directory."""
    return sorted(EVAL_DIR.glob("test_*.py"))


def get_eval_imports_from_graph() -> dict[str, set[str]]:
    """Build {eval_file: set(source_modules)} map from the graph DB.

    Queries edges where an eval test node imports from a source module.
    """
    if not GRAPH_DB.exists():
        return {}

    conn = sqlite3.connect(str(GRAPH_DB))
    cursor = conn.cursor()

    # Get all eval test nodes with their file paths
    cursor.execute(
        "SELECT DISTINCT file_path FROM nodes "
        "WHERE file_path LIKE '%tests/evals/test_%' AND kind = 'Test'"
    )
    eval_test_paths = {row[0] for row in cursor.fetchall()}

    # Get all edges where these test nodes import from source modules
    eval_imports: dict[str, set[str]] = {}
    for eval_path in eval_test_paths:
        # Get the qualified names of imports from this eval file
        cursor.execute(
            "SELECT DISTINCT e.target_qualified FROM edges e "
            "JOIN nodes n ON n.qualified_name = e.source_qualified "
            "WHERE n.file_path = ? AND e.kind = 'IMPORTS_FROM'",
            (eval_path,),
        )
        imports = set()
        for (target,) in cursor.fetchall():
            # Convert qualified name to file path pattern
            # e.g., 'chat_workflow.atomic_workflow' -> 'chat_workflow/atomic_workflow'
            imports.add(target.replace(".", "/"))
        if imports:
            eval_imports[eval_path] = imports

    conn.close()
    return eval_imports


def build_dependency_map() -> dict[str, set[str]]:
    """Build a complete {eval_file_path: set(source_file_patterns)} map.

    Uses the graph DB for import relationships. Falls back to AST parsing
    if the graph is unavailable.
    """
    eval_files = get_all_eval_files()
    deps: dict[str, set[str]] = {}

    # Try graph DB first
    graph_deps = get_eval_imports_from_graph()

    if graph_deps:
        # Map absolute paths from graph to our eval file paths
        for abs_path, sources in graph_deps.items():
            rel_path = os.path.relpath(abs_path, str(PROJECT_ROOT))
            deps[rel_path] = sources
        return deps

    # Fallback: AST parsing (if graph DB unavailable)
    import ast

    for ef in eval_files:
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
                deps[rel] = sources
        except SyntaxError:
            pass

    return deps


def find_affected_evals(changed_files: list[str], dep_map: dict[str, set[str]]) -> list[str]:
    """Return list of eval file paths affected by the changed files."""
    if not changed_files:
        return []

    affected: set[str] = set()

    # Check if any universal dependency changed
    for cf in changed_files:
        for ud in UNIVERSAL_DEPS:
            if cf == ud or cf.startswith("tests/"):
                # Any change in tests/ dir affects everything
                return sorted(dep_map.keys())

    for cf in changed_files:
        for eval_path, sources in dep_map.items():
            # Check if any source file pattern matches the changed file
            for src in sources:
                if src in cf or cf.startswith(src):
                    affected.add(eval_path)
                    break

    return sorted(affected)


def show_dependency_map(dep_map: dict[str, set[str]]) -> None:
    """Pretty-print the dependency map."""
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
        dep_map = build_dependency_map()
        show_dependency_map(dep_map)
        return

    changed_files = get_changed_files(args.git_base)
    dep_map = build_dependency_map()

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
        # Machine-readable: space-separated paths
        if affected:
            print(" ".join(affected))


if __name__ == "__main__":
    main()
