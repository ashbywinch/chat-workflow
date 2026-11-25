"""Unit tests for the CLI module — individual functions in isolation."""

import types
import unittest

from chat_workflow_cli.cli import (
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

        wf_func._is_workflow = True  # type: ignore[attr-defined]  # the test marks a plain function as a workflow

        def plain_func():
            pass

        module.wf_func = wf_func  # type: ignore[attr-defined]  # a module is built by hand for the test
        module.plain_func = plain_func  # type: ignore[attr-defined]  # a module is built by hand for the test

        result = discover_workflow_functions(module)
        self.assertEqual([(command.name, command.func) for command in result], [("wf_func", wf_func)])

    def test_empty_when_no_workflow_functions(self):
        module = types.ModuleType("test_mod")

        def plain_func():
            pass

        module.plain_func = plain_func  # type: ignore[attr-defined]  # a module is built by hand for the test

        result = discover_workflow_functions(module)
        self.assertEqual(result, [])

    def test_empty_module(self):
        module = types.ModuleType("empty_mod")
        result = discover_workflow_functions(module)
        self.assertEqual(result, [])


class TestBuildWorkflowSubApp(unittest.TestCase):
    """_build_workflow_sub_app wraps a module's @workflow functions into a Typer sub-app."""

    def test_import_error_returns_none(self):
        def failing_importer(name):
            raise ImportError(f"no such module: {name}")

        self.assertIsNone(_build_workflow_sub_app("nonexistent", importer=failing_importer))

    def test_no_workflow_functions_returns_none(self):
        fake_module = types.ModuleType("fake_workflow")
        result = _build_workflow_sub_app("empty_workflow", importer=lambda name: fake_module)
        self.assertIsNone(result)

    def test_creates_sub_app_with_command(self):
        fake_module = types.ModuleType("fake_workflow")

        def my_workflow(context: str = "", *, tools):
            pass

        my_workflow._is_workflow = True  # type: ignore[attr-defined]  # the test marks a plain function as a workflow

        fake_module.my_workflow = my_workflow  # type: ignore[attr-defined]  # a module is built by hand for the test

        sub_app = _build_workflow_sub_app("test_workflow", importer=lambda name: fake_module)

        self.assertIsNotNone(sub_app)
        self.assertEqual(len(sub_app.registered_commands), 1)
        registered = sub_app.registered_commands[0]
        self.assertEqual(registered.name, "my-workflow")


class TestDiscoverWorkflows(unittest.TestCase):
    """discover_workflows finds subdirectories with __init__.py."""

    # lucidlint: ignore fakefs the subject is the checked-in workflows/ tree this repo ships
    def test_returns_workflow_dirs(self):
        """When run from the actual project root, the evaluation_criteria workflow should be found."""
        from chat_workflow_cli.cli import discover_workflows

        modules = {module.name: module for module in discover_workflows()}
        self.assertIn("evaluation_criteria", modules)
        path = modules["evaluation_criteria"].path
        self.assertTrue(path.is_dir())
        self.assertTrue((path / "__init__.py").exists())

    # lucidlint: ignore fakefs the subject is the checked-in workflows/ tree this repo ships
    def test_skips_non_package_dirs(self):
        """Subdirectories without __init__.py are not workflows."""
        from chat_workflow_cli.cli import discover_workflows

        names = {module.name for module in discover_workflows()}
        self.assertNotIn("__pycache__", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
