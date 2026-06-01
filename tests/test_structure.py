"""Tests for ComponentStructure, StructField, and StructValidator models."""
import unittest

from pydantic import ValidationError

from workflows.workflow.structure import (
    ComponentStructure,
    StructField,
    StructValidator,
)


class TestStructField(unittest.TestCase):
    """Construction and field validation tests for StructField."""

    def make_valid(self, **overrides) -> StructField:
        kwargs = dict(
            name="action_owner",
            type_expr="str",
        )
        kwargs.update(overrides)
        return StructField(**kwargs)

    # --- Construction ---

    def test_valid_construction(self):
        f = self.make_valid()
        self.assertEqual(f.name, "action_owner")
        self.assertEqual(f.type_expr, "str")
        self.assertEqual(f.field_def_kwargs, {})

    def test_with_field_def_kwargs(self):
        f = self.make_valid(
            field_def_kwargs={
                "description": "The person responsible",
                "min_length": "1",
            }
        )
        self.assertEqual(f.field_def_kwargs["description"], "The person responsible")
        self.assertEqual(f.field_def_kwargs["min_length"], "1")

    def test_field_def_kwargs_defaults_to_empty_dict(self):
        f = self.make_valid()
        self.assertEqual(f.field_def_kwargs, {})

    # --- Missing required fields ---

    def test_missing_name_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(name="")

    def test_missing_type_expr_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(type_expr="")


class TestStructValidator(unittest.TestCase):
    """Construction and field validation tests for StructValidator."""

    def make_valid(self, **overrides) -> StructValidator:
        kwargs = dict(
            rule="description must not exceed 3 sentences",
            domain_origin="Descriptions should be concise and scannable",
        )
        kwargs.update(overrides)
        return StructValidator(**kwargs)

    # --- Construction ---

    def test_valid_construction(self):
        v = self.make_valid()
        self.assertEqual(v.rule, "description must not exceed 3 sentences")
        self.assertEqual(
            v.domain_origin, "Descriptions should be concise and scannable"
        )

    # --- Missing required fields ---

    def test_missing_rule_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(rule="")

    def test_missing_domain_origin_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(domain_origin="")


class TestComponentStructure(unittest.TestCase):
    """Construction and field validation tests for ComponentStructure."""

    def make_valid(self, **overrides) -> ComponentStructure:
        kwargs = dict(
            description="Structured meeting minutes",
        )
        kwargs.update(overrides)
        return ComponentStructure(**kwargs)

    # --- Construction ---

    def test_valid_construction(self):
        c = self.make_valid()
        self.assertEqual(c.description, "Structured meeting minutes")
        self.assertEqual(c.base_class, "BaseModel")
        self.assertEqual(c.fields, [])
        self.assertEqual(c.model_validators, [])
        self.assertEqual(c.extra_imports, [])

    def test_with_fields(self):
        c = self.make_valid(
            fields=[
                StructField(name="title", type_expr="str"),
                StructField(
                    name="action_items",
                    type_expr="list[ActionItem]",
                    field_def_kwargs={"description": "Action items from the meeting"},
                ),
            ]
        )
        self.assertEqual(len(c.fields), 2)
        self.assertEqual(c.fields[0].name, "title")
        self.assertEqual(c.fields[1].type_expr, "list[ActionItem]")

    def test_with_model_validators(self):
        c = self.make_valid(
            model_validators=[
                StructValidator(
                    rule="at least one action item required",
                    domain_origin="Meetings should produce actionable outcomes",
                ),
            ]
        )
        self.assertEqual(len(c.model_validators), 1)
        self.assertEqual(
            c.model_validators[0].rule, "at least one action item required"
        )

    def test_with_extra_imports(self):
        c = self.make_valid(
            extra_imports=["from datetime import datetime"],
        )
        self.assertEqual(c.extra_imports, ["from datetime import datetime"])

    def test_base_class_can_be_llm_validated(self):
        c = self.make_valid(base_class="LLMValidated")
        self.assertEqual(c.base_class, "LLMValidated")

    def test_fields_can_be_empty(self):
        c = self.make_valid(fields=[])
        self.assertEqual(c.fields, [])

    def test_model_validators_can_be_empty(self):
        c = self.make_valid(model_validators=[])
        self.assertEqual(c.model_validators, [])

    def test_extra_imports_can_be_empty(self):
        c = self.make_valid(extra_imports=[])
        self.assertEqual(c.extra_imports, [])

    # --- Missing required fields ---

    def test_missing_description_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(description="")


if __name__ == "__main__":
    unittest.main(verbosity=2)
