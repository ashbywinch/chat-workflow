"""Tests for ComponentResponsibilities model."""

import unittest

from pydantic import ValidationError

from workflows.workflow.component_responsibilities import (
    ComponentResponsibilities,
)


class TestComponentResponsibilities(unittest.TestCase):
    """Construction and field validation tests."""

    def make_valid(self, **overrides) -> ComponentResponsibilities:
        kwargs = dict(
            name="Invoice",
            purpose="Generate and manage invoices for completed orders",
            scope_description=(
                "Represents the invoice lifecycle: creation, delivery, "
                "payment tracking. Does NOT represent order fulfillment "
                "or inventory management."
            ),
            required_inputs=["Order", "Customer"],
            component_type="artifact_producing",
        )
        kwargs.update(overrides)
        return ComponentResponsibilities(**kwargs)

    # --- Construction ---

    def test_valid_construction(self):
        c = self.make_valid()
        self.assertEqual(c.name, "Invoice")
        self.assertEqual(c.purpose, "Generate and manage invoices for completed orders")
        self.assertIn("invoice lifecycle", c.scope_description)
        self.assertEqual(c.required_inputs, ["Order", "Customer"])
        self.assertEqual(c.component_type, "artifact_producing")
        self.assertEqual(c.incidental_notes, "")

    def test_incidental_notes_defaults_to_empty_string(self):
        c = self.make_valid()
        self.assertEqual(c.incidental_notes, "")

    def test_incidental_notes_can_be_set(self):
        c = self.make_valid(incidental_notes="User mentioned they use QuickBooks")
        self.assertEqual(c.incidental_notes, "User mentioned they use QuickBooks")

    def test_required_inputs_can_be_empty(self):
        c = self.make_valid(required_inputs=[])
        self.assertEqual(c.required_inputs, [])

    def test_required_inputs_with_multiple_values(self):
        c = self.make_valid(required_inputs=["Order", "Customer", "PaymentGateway"])
        self.assertEqual(len(c.required_inputs), 3)

    # --- Missing required fields ---

    def test_missing_name_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(name="")

    def test_missing_purpose_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(purpose="")

    def test_missing_scope_description_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(scope_description="")

    def test_missing_component_type_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(component_type="")

    # --- component_type validation ---

    def test_valid_component_types(self):
        for ct in ("value_stream", "artifact_producing", "planning_service"):
            with self.subTest(component_type=ct):
                c = self.make_valid(component_type=ct)
                self.assertEqual(c.component_type, ct)

    def test_invalid_component_type_raises(self):
        with self.assertRaises(ValidationError):
            self.make_valid(component_type="invalid_type")

    def test_invalid_component_type_message(self):
        with self.assertRaises(ValidationError) as ctx:
            self.make_valid(component_type="bad_type")
        err = str(ctx.exception)
        self.assertIn("artifact_producing", err)
        self.assertIn("planning_service", err)
        self.assertIn("value_stream", err)
        self.assertIn("bad_type", err)

    # --- Edge cases ---

    def test_long_scope_description(self):
        long_desc = "Represents X. " * 100
        c = self.make_valid(scope_description=long_desc.strip())
        self.assertTrue(c.scope_description.startswith("Represents X."))

    def test_incidental_notes_with_empty_string(self):
        c = self.make_valid(incidental_notes="")
        self.assertEqual(c.incidental_notes, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
