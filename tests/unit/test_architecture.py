"""Enforce the layer boundaries recorded in docs/TECHSPEC.md.

`chat_workflow` is the core; `workflows` are consumers of it; the CLI is the
composition root that may see both. The core must never depend on a consumer —
that is the boundary that keeps the library embeddable.

Why this scans imports itself instead of using an archunitpython layer check:
archunitpython resolves absolute imports only from the project root (and its
parent). With the sources in a `packages/<name>/src/` layout, no package import
resolves from the repo root, every cross-package edge looks external, and the
layer check passes VACUOUSLY. Measured: an inverted rule reported 39
component→core edges before the move and 0 after it. This scan maps each
imported top-level module to its layer directly, so the rule cannot go vacuous;
`test_boundaries_are_observed` fails if the expected edges disappear from the
scan.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # tests/unit -> repo root

# Layer -> the package's `src/` root. Production source only: the test suite
# lives at the repo root and is not part of any layer.
SRC_ROOTS = {
    "core": ROOT / "packages" / "chat-workflow" / "src",
    "components": ROOT / "packages" / "chat-workflow-components" / "src",
    "cli": ROOT / "packages" / "chat-workflow-cli" / "src",
}

# Top-level module -> layer.
MODULE_LAYERS = {
    "chat_workflow": "core",
    "workflows": "components",
    "chat_workflow_cli": "cli",
}

# Layer -> the layers it may import. Everything else is a violation.
ALLOWED_DEPENDENCIES = {
    "core": frozenset(),
    "components": frozenset({"core"}),
    "cli": frozenset({"core", "components"}),
}


def _imported_top_level_modules(path: Path) -> set[str]:
    """Top-level modules a source file imports absolutely (relative = intra-layer)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def layer_edges() -> dict[tuple[str, str], list[str]]:
    """Return ``{(source_layer, target_layer): [offender, ...]}`` for real edges."""
    edges: dict[tuple[str, str], list[str]] = {}
    for source_layer, src_root in SRC_ROOTS.items():
        for path in sorted(src_root.rglob("*.py")):
            for module in _imported_top_level_modules(path):
                target_layer = MODULE_LAYERS.get(module)
                if target_layer is None or target_layer == source_layer:
                    continue
                rel_path = path.relative_to(ROOT)
                edges.setdefault((source_layer, target_layer), []).append(f"{rel_path} imports {module}")
    return edges


class TestArchitectureLayers(unittest.TestCase):
    """The core never imports a consumer; the CLI may see both."""

    def test_layers_only_depend_on_allowed_layers(self):
        violations = [
            f"{source_layer} -> {target_layer}: {offender}"
            for (source_layer, target_layer), offenders in layer_edges().items()
            if target_layer not in ALLOWED_DEPENDENCIES[source_layer]
            for offender in offenders
        ]
        self.assertEqual(violations, [], "Layer violations found:\n" + "\n".join(violations))

    def test_boundaries_are_observed(self):
        """Guard against a vacuous scan: the cross-package edges must be found."""
        edges = layer_edges()
        self.assertIn(("components", "core"), edges)
        self.assertIn(("cli", "core"), edges)


if __name__ == "__main__":
    unittest.main(verbosity=2)
