"""Unit tests for the CLI module — individual functions in isolation."""

import types
import unittest
from unittest.mock import patch

from chat_workflow.cli import (
    _build_workflow_sub_app,
    _snake_to_kebab,
    discover_workflow_functions,
)


class TestSnakeToKebab(unittest.TestCase):
    """_snake_to_kebab is a trivial re-format helper."""

    def test_basic(self):
        self.assertEqual(_snake_to_kebab("hello_world"), "hello-world")

    def test_multi_underscore(self):
        self.assertEqual(_snake_to_kebab("generate_reviewed_criteria"), "generate-reviewed-criteria")

    def test_no_underscore(self):
        self.assertEqual(_snake_to_kebab("hello"), "hello")

    def test_empty(self):
        self.assertEqual(_snake_to_kebab(""), "")

    def test_leading_trailing_underscore(self):
        self.assertEqual(_snake_to_kebab("_hello_"), "-hello-")

    def test_double_underscore(self):
        self.assertEqual(_snake_to_kebab("a__b"), "a--b")


class TestDiscoverWorkflowFunctions(unittest.TestCase):
    """discover_workflow_functions filters module members by _is_workflow."""

    def test_finds_workflow_functions(self):
        module = types.ModuleType("test_mod")

        def wf_func():
            pass

        wf_func._is_workflow = True  # type: ignore[attr-defined]

        def plain_func():
            pass

        module.wf_func = wf_func  # type: ignore[attr-defined]
        module.plain_func = plain_func  # type: ignore[attr-defined]

        result = discover_workflow_functions(module)
        self.assertEqual(result, {"wf_func": wf_func})

    def test_empty_when_no_workflow_functions(self):
        module = types.ModuleType("test_mod")

        def plain_func():
            pass

        module.plain_func = plain_func  # type: ignore[attr-defined]

        result = discover_workflow_functions(module)
        self.assertEqual(result, {})

    def test_empty_module(self):
        module = types.ModuleType("empty_mod")
        result = discover_workflow_functions(module)
        self.assertEqual(result, {})


class TestBuildWorkflowSubApp(unittest.TestCase):
    """_build_workflow_sub_app wraps a module's @workflow functions into a Typer sub-app."""

    def test_import_error_returns_none(self):
        with patch("chat_workflow.cli.importlib.import_module", side_effect=ImportError("no such module")):
            result = _build_workflow_sub_app("nonexistent")
            self.assertIsNone(result)

    def test_no_workflow_functions_returns_none(self):
        fake_module = types.ModuleType("fake_workflow")
        with (
            patch("chat_workflow.cli.importlib.import_module", return_value=fake_module),
        ):
            result = _build_workflow_sub_app("empty_workflow")
            self.assertIsNone(result)

    def test_creates_sub_app_with_command(self):
        fake_module = types.ModuleType("fake_workflow")

        def my_workflow(context: str = "", *, tools):
            pass

        my_workflow._is_workflow = True  # type: ignore[attr-defined]

        fake_module.my_workflow = my_workflow  # type: ignore[attr-defined]

        with patch("chat_workflow.cli.importlib.import_module", return_value=fake_module):
            sub_app = _build_workflow_sub_app("test_workflow")

        self.assertIsNotNone(sub_app)
        self.assertEqual(len(sub_app.registered_commands), 1)
        registered = sub_app.registered_commands[0]
        self.assertEqual(registered.name, "my-workflow")


class TestDiscoverWorkflows(unittest.TestCase):
    """discover_workflows finds subdirectories with __init__.py."""

    def test_returns_workflow_dirs(self):
        """When run from the actual project root, the evaluation_criteria workflow should be found."""
        from chat_workflow.cli import discover_workflows

        result = discover_workflows()
        self.assertIn("evaluation_criteria", result)
        path = result["evaluation_criteria"]
        self.assertTrue(path.is_dir())
        self.assertTrue((path / "__init__.py").exists())

    def test_skips_non_package_dirs(self):
        """Subdirectories without __init__.py are not workflows."""
        from chat_workflow.cli import discover_workflows

        result = discover_workflows()
        self.assertNotIn("__pycache__", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
