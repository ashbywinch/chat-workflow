#!/usr/bin/env python3
"""Tests for the code generation library setup.

These tests describe the expected behavior of the code generator BEFORE
the chat_workflow/code_generator.py module exists. Tests that verify
the module and its generate_code() function will FAIL initially.

Scenarios tested:
1. libcst can be imported (should PASS — already installed)
2. chat_workflow.code_generator module can be imported (will FAIL)
3. A generate_code() function exists in the module (will FAIL)
4. generate_code() produces syntactically valid Python code (will FAIL)
5. Basic libcst operations work (parsing, generating simple class) (should PASS)
"""

import unittest


class TestLibCSTImport(unittest.TestCase):
    """Test that libcst is available (should pass — already installed)."""

    def test_libcst_import(self):
        """libcst should be importable."""
        import libcst as cst

        self.assertTrue(hasattr(cst, "parse_module"))

    def test_libcst_parse_simple_class(self):
        """libcst should parse a simple class definition."""
        import libcst as cst

        tree = cst.parse_module("class Test: pass")
        self.assertIsNotNone(tree)
        self.assertEqual(tree.code, "class Test: pass")

    def test_libcst_parse_and_generate_class(self):
        """libcst should round-trip parse and generate code."""
        import libcst as cst

        source = "class Foo:\n    pass\n"
        tree = cst.parse_module(source)
        self.assertEqual(tree.code, source)

    def test_libcst_class_def_construction(self):
        """libcst should support constructing a ClassDef programmatically."""
        import libcst as cst

        class_def = cst.ClassDef(
            name=cst.Name("MyClass"),
            body=cst.SimpleStatementSuite([cst.Pass()]),
        )
        module = cst.Module([class_def])
        generated = module.code
        self.assertIn("class MyClass:", generated)
        self.assertIn("pass", generated)

    def test_libcst_syntax_validity(self):
        """libcst-generated code should be syntactically valid when re-parsed."""
        import libcst as cst

        class_def = cst.ClassDef(
            name=cst.Name("ValidClass"),
            body=cst.SimpleStatementSuite([cst.Pass()]),
        )
        module = cst.Module([class_def])
        generated = module.code

        # Re-parse to verify syntactic validity
        reparsed = cst.parse_module(generated)
        self.assertEqual(reparsed.code, generated)


class TestCodeGeneratorModule(unittest.TestCase):
    """Test that the code_generator module exists (will FAIL initially)."""

    def test_code_generator_module_import(self):
        """chat_workflow.code_generator should be importable."""
        from chat_workflow.code_generator import generate_code  # noqa: F401

    def test_generate_code_function_exists(self):
        """generate_code should be a callable function."""
        from chat_workflow.code_generator import generate_code

        self.assertTrue(callable(generate_code))

    def test_generate_code_returns_string(self):
        """generate_code should return a string."""
        from chat_workflow.code_generator import generate_code

        result = generate_code("class Test: pass")
        self.assertIsInstance(result, str)

    def test_generate_code_produces_valid_syntax(self):
        """generate_code output should be syntactically valid Python."""
        import libcst as cst

        from chat_workflow.code_generator import generate_code

        result = generate_code("class Test: pass")
        # Re-parsing should not raise
        tree = cst.parse_module(result)
        self.assertIsNotNone(tree)

    def test_generate_code_accepts_class_def_kwargs(self):
        """generate_code should accept keyword arguments for class construction."""
        from chat_workflow.code_generator import generate_code

        result = generate_code(
            class_name="MyModel",
            fields=[("name", "str"), ("value", "int")],
        )
        self.assertIsInstance(result, str)
        self.assertIn("class MyModel:", result)


class TestFieldGenerator(unittest.TestCase):
    """Test the generate_field() function (will FAIL initially — not yet implemented)."""

    def test_generate_field_import(self):
        """generate_field should be importable from chat_workflow.code_generator."""
        from chat_workflow.code_generator import generate_field  # noqa: F401

    def test_generate_field_is_callable(self):
        """generate_field should be a callable function."""
        from chat_workflow.code_generator import generate_field

        self.assertTrue(callable(generate_field))

    def test_generate_field_returns_string(self):
        """generate_field should return a string."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("name", "str", "User name")
        self.assertIsInstance(result, str)

    def test_generate_field_contains_field_name(self):
        """Generated field code should contain the field name."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("name", "str", "User name")
        self.assertIn("name:", result)

    def test_generate_field_contains_field_decorator(self):
        """Generated field code should contain Field(...)."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("name", "str", "User name")
        self.assertIn("Field(", result)

    def test_generate_field_contains_description(self):
        """Generated field code should include the description."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("name", "str", "User name")
        self.assertIn("User name", result)

    def test_generate_field_valid_syntax(self):
        """Generated field code should be syntactically valid Python."""
        import libcst as cst

        from chat_workflow.code_generator import generate_field

        result = generate_field("name", "str", "User name")
        # Wrap in a dummy class to make it a valid module for parsing
        wrapper = f"class Dummy:\n    {result}"
        tree = cst.parse_module(wrapper)
        self.assertIsNotNone(tree)

    def test_generate_field_str_type(self):
        """generate_field should handle str type."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("name", "str", "A name")
        self.assertIn("str", result)

    def test_generate_field_int_type(self):
        """generate_field should handle int type."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("age", "int", "Age in years")
        self.assertIn("int", result)

    def test_generate_field_float_type(self):
        """generate_field should handle float type."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("price", "float", "Price in dollars")
        self.assertIn("float", result)

    def test_generate_field_list_type(self):
        """generate_field should handle list type."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("tags", "list", "List of tags")
        self.assertIn("list", result)

    def test_generate_field_optional_type(self):
        """generate_field should handle optional fields (type | None)."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("nickname", "str | None", "Optional nickname")
        self.assertIn("None", result)

    def test_generate_field_no_description(self):
        """generate_field should handle missing description gracefully."""
        from chat_workflow.code_generator import generate_field

        result = generate_field("name", "str")
        self.assertIsInstance(result, str)
        self.assertIn("name:", result)
        self.assertIn("Field(", result)


class TestMethodGenerator(unittest.TestCase):
    """Test that generate_workflow_method exists and produces valid output.

    These tests will FAIL initially because generate_workflow_method()
    hasn't been implemented yet in chat_workflow/code_generator.py.

    Scenarios tested:
    1. The function is importable and callable
    2. It returns a string containing @atomic_workflow and @classmethod
    3. The generated code includes generate_from_chat method name
    4. The generated code includes context, max_turns, and session parameters
    5. The generated code is syntactically valid Python
    """

    def test_generate_workflow_method_import(self):
        """generate_workflow_method should be importable (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method  # noqa: F401

    def test_generate_workflow_method_is_callable(self):
        """generate_workflow_method should be a callable function (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        self.assertTrue(callable(generate_workflow_method))

    def test_generate_workflow_method_returns_string(self):
        """generate_workflow_method should return a string (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        self.assertIsInstance(result, str)

    def test_generate_workflow_method_contains_atomic_workflow(self):
        """Generated code should contain @atomic_workflow decorator (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        self.assertIn("@atomic_workflow", result)

    def test_generate_workflow_method_contains_classmethod(self):
        """Generated code should contain @classmethod decorator (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        self.assertIn("@classmethod", result)

    def test_generate_workflow_method_contains_generate_from_chat(self):
        """Generated code should contain generate_from_chat method (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        self.assertIn("generate_from_chat", result)

    def test_generate_workflow_method_contains_context_param(self):
        """Generated code should include context parameter (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        self.assertIn("context", result)

    def test_generate_workflow_method_contains_max_turns_param(self):
        """Generated code should include max_turns parameter (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        self.assertIn("max_turns", result)

    def test_generate_workflow_method_contains_session_param(self):
        """Generated code should include session parameter (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        self.assertIn("session", result)

    def test_generate_workflow_method_produces_valid_syntax(self):
        """Generated code should be syntactically valid Python (will FAIL)."""
        import libcst as cst

        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("TestWorkflow")
        # Re-parsing should not raise
        tree = cst.parse_module(result)
        self.assertIsNotNone(tree)

    def test_generate_workflow_method_accepts_class_name_arg(self):
        """generate_workflow_method should accept a class name argument (will FAIL)."""
        from chat_workflow.code_generator import generate_workflow_method

        result = generate_workflow_method("MyCustomWorkflow")
        self.assertIsInstance(result, str)
        self.assertIn("MyCustomWorkflow", result)

    """Test the generate_model_validator function (will FAIL initially — not yet implemented)."""

    def test_generate_model_validator_import(self):
        """generate_model_validator should be importable from code_generator."""
        from chat_workflow.code_generator import generate_model_validator  # noqa: F401

    def test_generate_model_validator_is_callable(self):
        """generate_model_validator should be a callable function."""
        from chat_workflow.code_generator import generate_model_validator

        self.assertTrue(callable(generate_model_validator))

    def test_generate_model_validator_returns_string(self):
        """generate_model_validator should return a string."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        self.assertIsInstance(result, str)

    def test_generate_model_validator_contains_decorator(self):
        """Generated code should contain the @model_validator decorator."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        self.assertIn("@model_validator(mode=\"after\")", result)

    def test_generate_model_validator_contains_method_signature(self):
        """Generated code should contain the validate_business_rules method."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        self.assertIn("def validate_business_rules(self):", result)

    def test_generate_model_validator_contains_validation_logic(self):
        """Generated code should contain the validation check."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        self.assertIn("if not self.name:", result)

    def test_generate_model_validator_contains_raise(self):
        """Generated code should raise ValidationError with the given message."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        self.assertIn('raise ValidationError("name must be non-empty")', result)

    def test_generate_model_validator_returns_self(self):
        """Generated code should return self."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        self.assertIn("return self", result)

    def test_generate_model_validator_imports_validation_error(self):
        """Generated code should import ValidationError."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        self.assertIn("from chat_workflow import ValidationError", result)

    def test_generate_model_validator_syntactically_valid(self):
        """Generated code should be syntactically valid Python."""
        import libcst as cst

        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("name must be non-empty")
        tree = cst.parse_module(result)
        self.assertIsNotNone(tree)

    def test_generate_model_validator_accepts_custom_field(self):
        """generate_model_validator should accept a custom field name."""
        from chat_workflow.code_generator import generate_model_validator

        result = generate_model_validator("value must be positive", field_name="value")
        self.assertIn("if not self.value:", result)
        self.assertIn('raise ValidationError("value must be positive")', result)


class TestImportReload(unittest.TestCase):
    """Test the import/reload mechanism (will FAIL initially — functions don't exist yet)."""

    def test_import_module_returns_module(self):
        """import_module should return a module object for a valid dotted path."""
        from chat_workflow.code_generator import import_module

        mod = import_module("workflows.evaluation_criteria")
        self.assertIsNotNone(mod)
        self.assertTrue(hasattr(mod, "__name__"))
        self.assertEqual(mod.__name__, "workflows.evaluation_criteria")

    def test_reload_module_returns_module(self):
        """reload_module should return a module object for an already-imported module."""
        from chat_workflow.code_generator import reload_module

        mod = reload_module("workflows.evaluation_criteria")
        self.assertIsNotNone(mod)
        self.assertTrue(hasattr(mod, "__name__"))

    def test_import_module_invalid_path_returns_none(self):
        """import_module should return None for an invalid dotted path."""
        from chat_workflow.code_generator import import_module

        result = import_module("nonexistent.module.path")
        self.assertIsNone(result)

    def test_import_module_invalid_path_does_not_raise(self):
        """import_module should not raise an exception for an invalid path."""
        from chat_workflow.code_generator import import_module

        try:
            import_module("nonexistent.module.path")
        except Exception:
            self.fail("import_module raised an exception for an invalid path")

    def test_import_then_reload_round_trip(self):
        """Module should be importable and then reloadable."""
        from chat_workflow.code_generator import import_module, reload_module

        mod1 = import_module("workflows.evaluation_criteria")
        self.assertIsNotNone(mod1)

        mod2 = reload_module("workflows.evaluation_criteria")
        self.assertIsNotNone(mod2)
        self.assertEqual(mod2.__name__, "workflows.evaluation_criteria")


class TestClassGenerator(unittest.TestCase):
    """Test the generate_class() function (will FAIL initially — not yet implemented).

    generate_class() should produce a complete InteractiveEntity subclass
    with imports, fields, validation rules, model_validator, and workflow
    classmethod.
    """

    def test_generate_class_import(self):
        """generate_class should be importable from chat_workflow.code_generator."""
        from chat_workflow.code_generator import generate_class  # noqa: F401

    def test_generate_class_is_callable(self):
        """generate_class should be a callable function."""
        from chat_workflow.code_generator import generate_class

        self.assertTrue(callable(generate_class))

    def test_generate_class_returns_string(self):
        """generate_class should return a string."""
        from chat_workflow.code_generator import generate_class

        fields = [{"name": "test", "type": "str", "desc": "A test field"}]
        result = generate_class("TestEntity", fields)
        self.assertIsInstance(result, str)

    def test_generate_class_contains_class_definition(self):
        """Generated code should contain 'class TestEntity(InteractiveEntity):'."""
        from chat_workflow.code_generator import generate_class

        fields = [{"name": "test", "type": "str", "desc": "A test field"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("class TestEntity(InteractiveEntity):", result)

    def test_generate_class_is_valid_syntax(self):
        """Generated code should be syntactically valid Python."""
        import libcst as cst

        from chat_workflow.code_generator import generate_class

        fields = [{"name": "test", "type": "str", "desc": "A test field"}]
        result = generate_class("TestEntity", fields)
        tree = cst.parse_module(result)
        self.assertIsNotNone(tree)

    def test_generate_class_includes_imports(self):
        """Generated code should include required imports."""
        from chat_workflow.code_generator import generate_class

        fields = [{"name": "test", "type": "str", "desc": "A test field"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("from __future__ import annotations", result)
        self.assertIn("from pydantic import", result)
        self.assertIn("from chat_workflow import", result)

    def test_generate_class_includes_fields(self):
        """Generated code should include the specified fields with Field descriptors."""
        from chat_workflow.code_generator import generate_class

        fields = [
            {"name": "name", "type": "str", "desc": "The name"},
            {"name": "count", "type": "int", "desc": "The count"},
        ]
        result = generate_class("TestEntity", fields)
        self.assertIn("name: str = Field(..., description=\"The name\")", result)
        self.assertIn("count: int = Field(..., description=\"The count\")", result)

    def test_generate_class_includes_validation_rules(self):
        """Generated code should include _validation_rules class attribute."""
        from chat_workflow.code_generator import generate_class

        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class(
            "TestEntity",
            fields,
            validation_rules="name must be non-empty",
        )
        self.assertIn("_validation_rules", result)
        self.assertIn("name must be non-empty", result)

    def test_generate_class_includes_model_validator(self):
        """Generated code should include a model_validator method."""
        from chat_workflow.code_generator import generate_class

        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("model_validator", result)
        self.assertIn("validate_business_rules", result)

    def test_generate_class_includes_workflow_classmethod(self):
        """Generated code should include a generate_from_chat classmethod."""
        from chat_workflow.code_generator import generate_class

        fields = [{"name": "name", "type": "str", "desc": "The name"}]
        result = generate_class("TestEntity", fields)
        self.assertIn("generate_from_chat", result)
        self.assertIn("atomic_workflow", result)
        self.assertIn("@classmethod", result)


class TestVerifyCode(unittest.TestCase):
    """Test the verify_code() function for ruff linting/formatting verification.

    verify_code() should:
    1. Pass clean code through unchanged
    2. Fix badly formatted code automatically
    3. Raise RuntimeError when code can't be fixed after max_attempts
    """

    def test_verify_code_import(self):
        """verify_code should be importable from chat_workflow.code_generator."""
        from chat_workflow.code_generator import verify_code  # noqa: F401

    def test_verify_code_is_callable(self):
        """verify_code should be a callable function."""
        from chat_workflow.code_generator import verify_code

        self.assertTrue(callable(verify_code))

    def test_verify_code_returns_string(self):
        """verify_code should return a string for clean code."""
        from chat_workflow.code_generator import verify_code

        result = verify_code("x = 1\n")
        self.assertIsInstance(result, str)

    def test_verify_code_clean_code_passes_unchanged(self):
        """Clean, well-formatted code should pass through unchanged."""
        from chat_workflow.code_generator import verify_code

        source = "x = 1\ny = 2\n"
        result = verify_code(source)
        self.assertEqual(result, source)

    def test_verify_code_fixes_bad_formatting(self):
        """Badly formatted code should be auto-fixed by ruff format."""
        from chat_workflow.code_generator import verify_code

        # Missing spaces around operators — ruff format will add them
        bad = "x=1\ny=2\n"
        result = verify_code(bad)
        self.assertIn("x = 1", result)
        self.assertIn("y = 2", result)
        # The fixed version should have spaces around =
        self.assertNotIn("x=1", result)

    def test_verify_code_fixes_lint_issues(self):
        """Code with auto-fixable lint issues should be fixed."""
        from chat_workflow.code_generator import verify_code

        # Unused import (F401) — ruff check --fix can remove it.
        # Code is already well-formatted so format check passes first.
        source = "import os\n\nx = 1\n"
        result = verify_code(source)
        self.assertNotIn("import os", result)
        self.assertIn("x = 1", result)

    def test_verify_code_max_attempts_raises_error(self):
        """verify_code should raise RuntimeError when code can't be fixed."""
        from chat_workflow.code_generator import verify_code

        # Undefined name (F821) — ruff cannot auto-fix this
        bad_code = "x = undefined_variable\n"
        with self.assertRaises(RuntimeError):
            verify_code(bad_code, max_attempts=2)

    def test_verify_code_max_attempts_error_message(self):
        """RuntimeError message should mention max_attempts."""
        from chat_workflow.code_generator import verify_code

        bad_code = "x = undefined_variable\n"
        with self.assertRaises(RuntimeError) as ctx:
            verify_code(bad_code, max_attempts=1)
        self.assertIn("1 attempts", str(ctx.exception))

    def test_verify_code_class_def_round_trip(self):
        """A valid class definition should pass through cleanly."""
        from chat_workflow.code_generator import verify_code

        source = "class Foo:\n    pass\n"
        result = verify_code(source)
        self.assertEqual(result, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)