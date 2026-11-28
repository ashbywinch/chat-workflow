"""Run lucidlint over every Python file and fail on any finding.

The installed CLI's ``--repo`` mode does not walk this tree, so the gate runs
the per-file mode that does. Warnings fail too: the scan is the standard, not a
suggestion, until the debt is paid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


class LintScope:
    """The tree the gate walks, and how it invokes the scanner."""

    ROOT = Path(__file__).resolve().parent.parent
    BINARY = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / ".venv/bin/lucidlint")
    SOURCE_DIRS = ("chat_workflow", "workflows", "tests", "scripts", "chat_workflow_cli")
    SKIP_PARTS = {"__pycache__", ".venv", ".sisyphus", "test-results", ".pytest_cache"}


def source_files() -> list[Path]:
    """Every Python file in the source dirs, minus the skipped parts."""
    return [
        path
        for directory in LintScope.SOURCE_DIRS
        for path in sorted((LintScope.ROOT / directory).rglob("*.py"))
        if LintScope.SKIP_PARTS.isdisjoint(path.parts)
    ]


def main() -> int:
    findings: list[tuple[str, str, int, str, str]] = []
    for path in source_files():
        result = subprocess.run(
            [LintScope.BINARY, "--file", str(path.relative_to(LintScope.ROOT)), "--json"],
            cwd=LintScope.ROOT,
            capture_output=True,
            text=True,
        )
        # The CLI prints a human summary line before the JSON report when a
        # file has actions; the report itself starts at the first brace.
        start = result.stdout.find("{")
        try:
            report = json.loads(result.stdout[start:]) if start >= 0 else {}
        except json.JSONDecodeError:
            print(f"{path}: lucidlint did not return JSON", file=sys.stderr)
            print(result.stdout[:500], file=sys.stderr)
            print(result.stderr[:500], file=sys.stderr)
            return 2
        findings.extend(
            (
                finding.get("severity", "?"),
                finding.get("kind", "?"),
                finding.get("line", 0),
                str(path.relative_to(LintScope.ROOT)),
                finding.get("message", ""),
            )
            for finding in report.get("actions", report.get("findings", []))
        )

    if not findings:
        print("lucidlint: clean")
        return 0

    print(f"lucidlint: {len(findings)} finding(s)")
    for severity, kind, line, path, message in findings:
        print(f"  {severity:4} {kind:14} {path}:{line}  {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
