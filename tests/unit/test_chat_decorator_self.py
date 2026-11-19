#!/usr/bin/env python3
"""Test that @atomic_workflow with Self return type raises TypeError.

This test proves that using typing.Self as a return type in a @atomic_workflow-decorated
classmethod causes a TypeError: issubclass() arg 1 must be a class.

The bug is at chat_workflow/atomic_workflow.py:511 where issubclass()
is called on the return type annotation without checking if it's a special
form like typing.Self.

Mirrors the pattern from workflows/evaluation_criteria/evaluation_criteria.py:43-49
which uses Self as a classmethod return type.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from typing import TypeVar

from pydantic import BaseModel


class TestChatDecoratorSelfTypeError(unittest.TestCase):
    """Prove that Self return type in @atomic_workflow decorator no longer raises at import time."""

    def test_self_return_type_imports_successfully(self):
        """Importing a module with @atomic_workflow + Self return type should succeed (no TypeError)."""
        module_code = """
from typing import Self
from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow


class DummyModel(BaseModel):
    name: str = Field(description="A name")


class TestWorkflow:
    @atomic_workflow
    @classmethod
    def generate(
        cls,
        context: str,
        max_turns: int = 10,
    ) -> Self:
        \"\"\"You are a helpful assistant.\"\"\"
        pass
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "test_self_module.py")
            with open(module_path, "w") as f:
                f.write(module_code)

            spec = importlib.util.spec_from_file_location(
                "test_self_module",
                module_path,
            )
            self.assertIsNotNone(spec, "Failed to create module spec")
            self.assertIsNotNone(spec.loader, "Module spec has no loader")

            mod = importlib.util.module_from_spec(spec)

            # Add project root to sys.path so chat_workflow is importable
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            old_path = sys.path.copy()
            sys.path.insert(0, project_root)
            try:
                # Should NOT raise - Self type resolution is deferred to call time
                spec.loader.exec_module(mod)
                self.assertTrue(
                    hasattr(mod, "TestWorkflow"),
                    "Module should have TestWorkflow class",
                )
                self.assertTrue(
                    hasattr(mod.TestWorkflow, "generate"),
                    "TestWorkflow should have 'generate' method",
                )
            finally:
                sys.path[:] = old_path


class TestChatDecoratorTypeVar(unittest.TestCase):
    """Test that @atomic_workflow with TypeVar return type works correctly."""

    def test_typevar_decoration_succeeds(self):
        """TypeVar bound to BaseModel should decorate without error."""
        module_code = """
from typing import TypeVar
from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow

ModelType = TypeVar("ModelType", bound=BaseModel)


class DummyModel(BaseModel):
    name: str = Field(description="A name")


@atomic_workflow
def generate(context: str, max_turns: int = 10) -> ModelType:
    \"\"\"You are a helpful assistant.\"\"\"
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "test_typevar_success.py")
            with open(module_path, "w") as f:
                f.write(module_code)

            spec = importlib.util.spec_from_file_location(
                "test_typevar_success",
                module_path,
            )
            self.assertIsNotNone(spec, "Failed to create module spec")
            self.assertIsNotNone(spec.loader, "Module spec has no loader")

            mod = importlib.util.module_from_spec(spec)

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            old_path = sys.path.copy()
            sys.path.insert(0, project_root)
            try:
                # Should not raise - TypeVar with BaseModel bound is valid
                spec.loader.exec_module(mod)
            finally:
                sys.path[:] = old_path

    def test_typevar_import_does_not_raise(self):
        """Module-level decorated function imports without error."""
        module_code = """
from typing import TypeVar
from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow

ModelType = TypeVar("ModelType", bound=BaseModel)


class DummyModel(BaseModel):
    name: str = Field(description="A name")


@atomic_workflow
def generate(context: str, max_turns: int = 10) -> ModelType:
    \"\"\"You are a helpful assistant.\"\"\"
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "test_typevar_import.py")
            with open(module_path, "w") as f:
                f.write(module_code)

            spec = importlib.util.spec_from_file_location(
                "test_typevar_import",
                module_path,
            )
            self.assertIsNotNone(spec, "Failed to create module spec")
            self.assertIsNotNone(spec.loader, "Module spec has no loader")

            mod = importlib.util.module_from_spec(spec)

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            old_path = sys.path.copy()
            sys.path.insert(0, project_root)
            try:
                spec.loader.exec_module(mod)
                self.assertTrue(
                    hasattr(mod, "generate"),
                    "Module should have 'generate' function",
                )
            finally:
                sys.path[:] = old_path

    def test_typevar_no_params_requires_session(self):
        """Calling without session raises TypeError."""
        module_code = """
from typing import TypeVar
from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow

ModelType = TypeVar("ModelType", bound=BaseModel)


class DummyModel(BaseModel):
    name: str = Field(description="A name")


@atomic_workflow
def generate(context: str, max_turns: int = 10) -> ModelType:
    \"\"\"You are a helpful assistant.\"\"\"
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "test_typevar_tools.py")
            with open(module_path, "w") as f:
                f.write(module_code)

            spec = importlib.util.spec_from_file_location(
                "test_typevar_tools",
                module_path,
            )
            self.assertIsNotNone(spec, "Failed to create module spec")
            self.assertIsNotNone(spec.loader, "Module spec has no loader")

            mod = importlib.util.module_from_spec(spec)

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            old_path = sys.path.copy()
            sys.path.insert(0, project_root)
            try:
                spec.loader.exec_module(mod)
                with self.assertRaises(TypeError) as ctx:
                    mod.generate(context="test")
                error_msg = str(ctx.exception)
                self.assertIn(
                    "requires 'session' parameter",
                    error_msg,
                    f"Expected 'requires session parameter' in error, got: {error_msg}",
                )
            finally:
                sys.path[:] = old_path

    def test_typevar_bound_is_base_model(self):
        """TypeVar bound is BaseModel."""
        module_code = """
from typing import TypeVar
from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow

ModelType = TypeVar("ModelType", bound=BaseModel)


class DummyModel(BaseModel):
    name: str = Field(description="A name")


@atomic_workflow
def generate(context: str, max_turns: int = 10) -> ModelType:
    \"\"\"You are a helpful assistant.\"\"\"
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, "test_typevar_bound.py")
            with open(module_path, "w") as f:
                f.write(module_code)

            spec = importlib.util.spec_from_file_location(
                "test_typevar_bound",
                module_path,
            )
            self.assertIsNotNone(spec, "Failed to create module spec")
            self.assertIsNotNone(spec.loader, "Module spec has no loader")

            mod = importlib.util.module_from_spec(spec)

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            old_path = sys.path.copy()
            sys.path.insert(0, project_root)
            try:
                spec.loader.exec_module(mod)
                model_type = mod.ModelType
                self.assertIsInstance(model_type, TypeVar)
                self.assertIs(
                    model_type.__bound__,
                    BaseModel,
                    "TypeVar bound should be BaseModel",
                )
            finally:
                sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
