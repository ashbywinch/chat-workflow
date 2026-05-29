#!/usr/bin/env python3
"""End-to-end integration tests for the InteractiveEntity code generation system.

Tests the full flow:
1. ``generate_class()`` produces valid InteractiveEntity subclass code
2. The generated code can be written to a file and imported
3. The imported class can be instantiated with valid data
4. The imported class rejects invalid data via validation
5. ``verify_code()`` ensures the generated code passes linting
6. The ``create_workflow`` workflow is discoverable by the CLI
7. Edge cases: empty fields list, no validation rules
"""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import libcst as cst

from chat_workflow import InteractiveEntity, ValidationError
from chat_workflow.code_generator import generate_class, verify_code
from chat_workflow_cli.cli import discover_workflows


class TestGenerateClassProducesValidCode(unittest.TestCase):
    """Test that generate_class() produces syntactically valid Python code."""

    def test_generate_class_returns_string(self):
        """generate_class should return a string."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertIsInstance(result, str)

    def test_generated_code_is_valid_python(self):
        """Generated code should be syntactically valid Python."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        tree = cst.parse_module(result)
        self.assertIsNotNone(tree)

    def test_generated_code_is_interactive_entity_subclass(self):
        """Generated class should inherit from InteractiveEntity."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("class TestEntity(InteractiveEntity):", result)

    def test_generated_code_includes_required_imports(self):
        """Generated code should include all required imports."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("from __future__ import annotations", result)
        self.assertIn("from pydantic import BaseModel, Field, model_validator", result)
        self.assertIn("from chat_workflow import InteractiveEntity, ValidationError, atomic_workflow", result)

    def test_generated_code_includes_fields(self):
        """Generated code should include the specified fields."""
        fields = [
            {"name": "name", "type": "str", "desc": "The name"},
            {"name": "count", "type": "int", "desc": "The count"},
        ]
        result = generate_class("TestEntity", fields)
        self.assertIn("name: str = Field(..., description=\"The name\")", result)
        self.assertIn("count: int = Field(..., description=\"The count\")", result)

    def test_generated_code_includes_model_validator(self):
        """Generated code should include a model_validator method."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("@model_validator(mode=\"after\")", result)
        self.assertIn("def validate_business_rules(self):", result)

    def test_generated_code_includes_workflow_classmethod(self):
        """Generated code should include a generate_from_chat classmethod."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("@atomic_workflow", result)
        self.assertIn("@classmethod", result)
        self.assertIn("def generate_from_chat(", result)

    def test_generated_code_includes_validation_rules(self):
        """Generated code should include _validation_rules when provided."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class(
            "TestEntity",
            fields,
            validation_rules="name must be non-empty",
        )
        self.assertIn('_validation_rules: str = "name must be non-empty"', result)

    def test_generated_code_without_validation_rules(self):
        """Generated code should work without validation_rules."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertNotIn("_validation_rules", result)


class TestVerifyCodeOnGeneratedCode(unittest.TestCase):
    """Test that verify_code() passes on generated code."""

    def test_verify_code_passes_on_generated_code(self):
        """Generated code should pass ruff linting and formatting."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class("TestEntity", fields)
        result = verify_code(code)
        self.assertIsInstance(result, str)
        # The result should still be valid Python
        tree = cst.parse_module(result)
        self.assertIsNotNone(tree)

    def test_verify_code_passes_with_validation_rules(self):
        """Generated code with validation rules should pass ruff checks."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class("TestEntity", fields, validation_rules="name must be non-empty")
        result = verify_code(code)
        self.assertIsInstance(result, str)
        self.assertIn("name must be non-empty", result)

    def test_verify_code_passes_with_multiple_fields(self):
        """Generated code with multiple fields should pass ruff checks."""
        fields = [
            {"name": "name", "type": "str", "desc": "The name"},
            {"name": "count", "type": "int", "desc": "The count"},
            {"name": "active", "type": "bool", "desc": "Is active"},
        ]
        code = generate_class("TestEntity", fields)
        result = verify_code(code)
        self.assertIsInstance(result, str)

    def test_verify_code_passes_with_empty_fields(self):
        """Generated code with empty fields list should pass ruff checks."""
        code = generate_class("EmptyEntity", [])
        result = verify_code(code)
        self.assertIsInstance(result, str)


class TestDynamicImportAndInstantiation(unittest.TestCase):
    """Test that generated code can be written to a file, imported, and used."""

    def _write_and_import(self, code: str, module_name: str, tmp_dir: Path):
        """Helper: write code to a temp file and import it as a module."""
        # Write the module file
        module_path = tmp_dir / f"{module_name}.py"
        module_path.write_text(code)

        # Add temp dir to sys.path and import
        sys_path_before = list(sys.path)
        try:
            sys.path.insert(0, str(tmp_dir))
            if module_name in sys.modules:
                del sys.modules[module_name]
            mod = importlib.import_module(module_name)
            return mod
        finally:
            sys.path[:] = sys_path_before

    def test_import_generated_class(self):
        """Generated code should be importable as a Python module."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class("TestEntity", fields)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mod = self._write_and_import(code, "test_entity_mod", Path(tmp_dir))
            self.assertIsNotNone(mod)
            self.assertTrue(hasattr(mod, "TestEntity"))

    def test_instantiate_with_valid_data(self):
        """Imported class should instantiate with valid field values."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class("TestEntity", fields)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mod = self._write_and_import(code, "test_entity_inst", Path(tmp_dir))
            entity = mod.TestEntity(name="hello world")
            self.assertEqual(entity.name, "hello world")
            self.assertIsInstance(entity, InteractiveEntity)

    def test_instantiate_with_multiple_fields(self):
        """Imported class should handle multiple fields correctly."""
        fields = [
            {"name": "title", "type": "str", "desc": "The title"},
            {"name": "count", "type": "int", "desc": "The count"},
            {"name": "active", "type": "bool", "desc": "Is active"},
        ]
        code = generate_class("MultiFieldEntity", fields)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mod = self._write_and_import(code, "test_multi_field", Path(tmp_dir))
            entity = mod.MultiFieldEntity(title="test", count=42, active=True)
            self.assertEqual(entity.title, "test")
            self.assertEqual(entity.count, 42)
            self.assertTrue(entity.active)

    def test_validation_rejects_invalid_data(self):
        """Imported class should reject invalid data via model_validator."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class("ValidatedEntity", fields)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mod = self._write_and_import(code, "test_validated", Path(tmp_dir))
            # Empty name should fail the model_validator check
            with self.assertRaises(ValidationError):
                mod.ValidatedEntity(name="")

    def test_validation_rejects_invalid_data_with_rules(self):
        """Imported class with _validation_rules should reject invalid data."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class(
            "RuledEntity",
            fields,
            validation_rules="name must be non-empty",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mod = self._write_and_import(code, "test_ruled", Path(tmp_dir))
            # Empty name should fail both model_validator and _validation_rules
            with self.assertRaises(ValidationError):
                mod.RuledEntity(name="")

    def test_valid_data_passes_validation(self):
        """Valid data should pass all validation checks."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class(
            "PassingEntity",
            fields,
            validation_rules="name must be non-empty",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mod = self._write_and_import(code, "test_passing", Path(tmp_dir))
            entity = mod.PassingEntity(name="valid name")
            self.assertEqual(entity.name, "valid name")

    def test_validation_error_message(self):
        """ValidationError should include context about the violation."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class("ErrorEntity", fields)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mod = self._write_and_import(code, "test_error", Path(tmp_dir))
            with self.assertRaises(ValidationError) as ctx:
                mod.ErrorEntity(name="")
            error_msg = str(ctx.exception).lower()
            self.assertIn("name", error_msg)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for the code generation system."""

    def test_empty_fields_list(self):
        """generate_class should handle an empty fields list."""
        code = generate_class("EmptyEntity", [])
        self.assertIsInstance(code, str)
        self.assertIn("class EmptyEntity(InteractiveEntity):", code)
        # Should still have model_validator and workflow method
        self.assertIn("@model_validator", code)
        self.assertIn("generate_from_chat", code)

    def test_no_validation_rules(self):
        """generate_class should work without validation_rules."""
        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        code = generate_class("NoRulesEntity", fields)
        self.assertNotIn("_validation_rules", code)
        # Should still have a default model_validator
        self.assertIn("@model_validator", code)

    def test_field_without_description(self):
        """generate_class should handle fields without a description."""
        fields = [{"name": "name", "type": "str"}]
        code = generate_class("NoDescEntity", fields)
        self.assertIn("name: str = Field(...)", code)

    def test_optional_field_type(self):
        """generate_class should handle optional field types (str | None)."""
        fields = [{"name": "nickname", "type": "str | None", "desc": "Optional nickname"}]
        code = generate_class("OptionalEntity", fields)
        self.assertIn("nickname: str | None = Field(", code)

    def test_int_field_type(self):
        """generate_class should handle int field type."""
        fields = [{"name": "count", "type": "int", "desc": "The count"}]
        code = generate_class("IntEntity", fields)
        self.assertIn("count: int = Field(", code)

    def test_float_field_type(self):
        """generate_class should handle float field type."""
        fields = [{"name": "price", "type": "float", "desc": "The price"}]
        code = generate_class("FloatEntity", fields)
        self.assertIn("price: float = Field(", code)

    def test_bool_field_type(self):
        """generate_class should handle bool field type."""
        fields = [{"name": "active", "type": "bool", "desc": "Is active"}]
        code = generate_class("BoolEntity", fields)
        self.assertIn("active: bool = Field(", code)

    def test_list_field_type(self):
        """generate_class should handle list field type."""
        fields = [{"name": "tags", "type": "list", "desc": "List of tags"}]
        code = generate_class("ListEntity", fields)
        self.assertIn("tags: list = Field(", code)

    def test_import_empty_fields_class(self):
        """A class with empty fields should still be importable and instantiable."""
        code = generate_class("EmptyFieldEntity", [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            sys_path_before = list(sys.path)
            try:
                tmp_path = Path(tmp_dir)
                sys.path.insert(0, str(tmp_path))
                module_path = tmp_path / "empty_field_mod.py"
                module_path.write_text(code)

                if "empty_field_mod" in sys.modules:
                    del sys.modules["empty_field_mod"]
                mod = importlib.import_module("empty_field_mod")
                self.assertTrue(hasattr(mod, "EmptyFieldEntity"))
                entity = mod.EmptyFieldEntity()
                self.assertIsInstance(entity, InteractiveEntity)
            finally:
                sys.path[:] = sys_path_before


class TestCliDiscovery(unittest.TestCase):
    """Test that the workflow management commands are discoverable by the CLI."""

    def test_workflow_discovered(self):
        """discover_workflows should find the workflow directory."""
        workflows = discover_workflows()
        self.assertIn("workflow", workflows)

    def test_workflow_has_init_py(self):
        """The workflow directory should have an __init__.py."""
        workflows = discover_workflows()
        workflow_path = workflows["workflow"]
        self.assertTrue((workflow_path / "__init__.py").exists())

    def test_workflow_discoverable_by_cli(self):
        """The CLI should be able to build a sub-app for workflow."""
        from chat_workflow_cli.cli import _build_workflow_sub_app
        sub_app = _build_workflow_sub_app("workflow")
        self.assertIsNotNone(sub_app)
        self.assertEqual(len(sub_app.registered_commands), 1)
        self.assertEqual(sub_app.registered_commands[0].name, "create")


class TestFullEndToEndFlow(unittest.TestCase):
    """Test the complete end-to-end flow end-to-end."""

    def test_full_flow_generate_import_instantiate_validate(self):
        """Complete flow: generate -> verify -> write -> import -> instantiate -> validate."""
        # 1. Generate
        fields = [
            {"name": "title", "type": "str", "desc": "The title"},
            {"name": "priority", "type": "int", "desc": "Priority level"},
        ]
        code = generate_class(
            "FullFlowEntity",
            fields,
            validation_rules="title must be non-empty",
        )

        # 2. Verify
        clean_code = verify_code(code)
        self.assertIsInstance(clean_code, str)

        # 3. Write to temp file and import
        with tempfile.TemporaryDirectory() as tmp_dir:
            sys_path_before = list(sys.path)
            try:
                tmp_path = Path(tmp_dir)
                sys.path.insert(0, str(tmp_path))
                module_path = tmp_path / "full_flow_mod.py"
                module_path.write_text(clean_code)

                if "full_flow_mod" in sys.modules:
                    del sys.modules["full_flow_mod"]
                mod = importlib.import_module("full_flow_mod")

                # 4. Instantiate with valid data
                entity = mod.FullFlowEntity(title="my task", priority=5)
                self.assertEqual(entity.title, "my task")
                self.assertEqual(entity.priority, 5)
                self.assertIsInstance(entity, InteractiveEntity)

                # 5. Validate rejects invalid data
                with self.assertRaises(ValidationError):
                    mod.FullFlowEntity(title="", priority=5)

                # 6. Validate rejects invalid data via _validation_rules
                with self.assertRaises(ValidationError):
                    mod.FullFlowEntity(title="", priority=0)

            finally:
                sys.path[:] = sys_path_before


if __name__ == "__main__":
    unittest.main(verbosity=2)