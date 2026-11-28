"""Tests for CLI discovery visibility rules.

Verifies that the framework correctly controls which workflows
are CLI-discoverable based on naming conventions and export rules.
"""

import types
import unittest

from chat_workflow_cli.cli import (
    _discover_class_workflow_methods,
    discover_workflow_functions,
)


def _commands(result) -> dict:
    return {command.name: command.func for command in result}


def _classes(result) -> dict[str, dict]:
    return {group.name: _commands(group.commands) for group in result}


class TestModuleLevelVisibility(unittest.TestCase):
    """Module-level workflow function discovery rules."""

    def test_public_workflow_discovered(self):
        """A public @atomic_workflow_step function is discovered by its name."""
        mod = types.ModuleType("test_mod")

        def public_wf(context: str = ""):
            pass

        public_wf._is_workflow_step = True  # type: ignore[attr-defined]  # the test marks a plain function as a workflow step

        mod.public_wf = public_wf  # type: ignore[attr-defined]  # a module is built by hand for the test

        result = _commands(discover_workflow_functions(mod))
        self.assertIn("public_wf", result)
        self.assertIs(result["public_wf"], public_wf)

    def test_private_function_not_discovered(self):
        """Functions starting with underscore are excluded even with _is_workflow_step."""
        mod = types.ModuleType("test_mod")

        def public_func():
            pass

        public_func._is_workflow_step = True  # type: ignore[attr-defined]  # the test marks a plain function as a workflow step

        def _private_func():
            pass

        _private_func._is_workflow_step = True  # type: ignore[attr-defined]  # the test marks a plain function as a workflow step

        mod.public_func = public_func  # type: ignore[attr-defined]  # a module is built by hand for the test
        mod._private_func = _private_func  # type: ignore[attr-defined]  # a module is built by hand for the test

        result = _commands(discover_workflow_functions(mod))
        self.assertIn("public_func", result)
        self.assertNotIn("_private_func", result)

    def test_generate_reviewed_criteria_discovered(self):
        """The real generate_reviewed_criteria in the example workflow is discoverable."""
        import workflows.evaluation_criteria  # noqa: F811 — the import is the fixture; discovery reads the module

        # discover_workflow_functions doesn't check __all__, just _is_workflow_step + public name
        mod = workflows.evaluation_criteria
        functions = _commands(discover_workflow_functions(mod))
        self.assertIn("generate_reviewed_criteria", functions)


class TestClassMethodVisibility(unittest.TestCase):
    """Classmethod workflow discovery rules via _discover_class_workflow_methods."""

    def test_exported_class_public_method_discovered(self):
        """WorkflowSpec.create is exported and public — should be discovered."""
        import workflows.workflow

        classes = _classes(_discover_class_workflow_methods(workflows.workflow))
        self.assertIn("WorkflowSpec", classes)
        self.assertIn("create", classes["WorkflowSpec"])

    def test_private_classmethod_not_discovered(self):
        """Private classmethods like _generate_diagram are excluded even with _is_workflow_step."""
        import workflows.workflow

        classes = _classes(_discover_class_workflow_methods(workflows.workflow))
        # The class is exported, but private methods inside it should not appear
        workflow_methods = classes.get("WorkflowSpec", {})
        for mname in workflow_methods:
            with self.subTest(method=mname):
                self.assertFalse(
                    mname.startswith("_"),
                    f"Private method {mname} should not be in discovered methods",
                )

    def test_non_exported_class_not_discovered(self):
        """Component is not in __all__, so its methods are not discovered."""
        import workflows.workflow

        classes = _classes(_discover_class_workflow_methods(workflows.workflow))
        self.assertNotIn("Component", classes)

    def test_mock_private_method_not_discovered(self):
        """A mock private method with _is_workflow_step on an exported class is excluded."""
        mod = types.ModuleType("test_mod")
        mod.__all__ = ["MyClass"]  # type: ignore[attr-defined]  # a module is built by hand for the test

        class MyClass:
            pass

        def public_method():
            pass

        public_method._is_workflow_step = True  # type: ignore[attr-defined]  # the test marks a plain function as a workflow step

        def _private_method():
            pass

        _private_method._is_workflow_step = True  # type: ignore[attr-defined]  # the test marks a plain function as a workflow step

        MyClass.public_method = staticmethod(public_method)  # type: ignore[attr-defined]  # the test attaches a step to a hand-built class
        MyClass._private_method = staticmethod(_private_method)  # type: ignore[attr-defined]  # the test attaches a step to a hand-built class
        mod.MyClass = MyClass  # type: ignore[attr-defined]  # a module is built by hand for the test

        classes = _classes(_discover_class_workflow_methods(mod))
        self.assertIn("MyClass", classes)
        self.assertIn("public_method", classes["MyClass"])
        self.assertNotIn("_private_method", classes["MyClass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
