"""Enforce the layer boundaries recorded in docs/TECHSPEC.md.

`chat_workflow/` is the core; `workflows/` are consumers of it; the CLI is
the composition root that may see both. The core must never depend on a
consumer — that is the boundary that keeps the library embeddable.

archunitpython matches layer patterns against ABSOLUTE normalized paths.
Relative globs ("chat_workflow/*.py") compile to regexes that match nothing,
every edge resolves to no layer, and the check passes VACUOUSLY. Derive the
patterns from the repo root (skill://archunitpython-glob-rules).
"""

import unittest
from pathlib import Path
from typing import Any

from archunitpython.layers import project_layers

ROOT = str(Path(__file__).resolve().parents[2])  # tests/unit -> repo root


def _format_violations(violations: list[Any]) -> str:
    lines = ["Layer violations found:"]
    for violation in violations:
        dependency = violation.dependency
        lines.append(f"  {dependency.source_label} → {dependency.target_label}")
    return "\n".join(lines)


class TestArchitectureLayers(unittest.TestCase):
    """The core never imports a consumer; the CLI may see both."""

    def test_core_does_not_depend_on_consumers(self):
        la = project_layers()
        la = la.layer("core").defined_by(f"{ROOT}/chat_workflow/*.py")
        la = la.layer("workflows").defined_by(f"{ROOT}/workflows/*.py")
        la = la.layer("cli").defined_by(f"{ROOT}/chat_workflow_cli/*.py")

        la = la.where_layer("core").may_only_depend_on_layers()
        la = la.where_layer("workflows").may_only_depend_on_layers("core")
        la = la.where_layer("cli").may_only_depend_on_layers("core", "workflows")

        violations = la.check()
        self.assertEqual(violations, [], _format_violations(violations))


if __name__ == "__main__":
    unittest.main(verbosity=2)
