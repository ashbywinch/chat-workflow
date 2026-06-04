#!/usr/bin/env python3
"""Failing tests for InteractiveEntity validation system.

These tests describe the expected validation behavior BEFORE implementation.
They will FAIL initially because the validation infrastructure in
InteractiveEntity has not been built yet.

When the validation system is implemented:
  - Field constraints on InteractiveEntity subclasses should raise
    chat_workflow.ValidationError (not pydantic.ValidationError)
  - _validation_rules should be automatically enforced at runtime
  - Model-level validators should integrate with the validation system
"""

import unittest

from pydantic import Field

from chat_workflow import InteractiveEntity, ValidationError


class TestInteractiveEntityFieldValidation(unittest.TestCase):
    """Field-level validation: subclasses use Pydantic Field() constraints.

    Expected behavior:
      - Invalid field values raise chat_workflow.ValidationError
      - Valid field values pass through without error
    """

    def test_min_length_constraint_rejects_empty_string(self):
        """FAILS: Field(min_length=1) raises pydantic.ValidationError,
        not chat_workflow.ValidationError (no conversion mechanism yet)."""

        class NamedEntity(InteractiveEntity):
            name: str = Field(default="", min_length=1)

        with self.assertRaises(ValidationError):
            NamedEntity(name="")

    def test_ge_constraint_rejects_below_minimum(self):
        """FAILS: Field(ge=0) raises pydantic.ValidationError,
        not chat_workflow.ValidationError."""

        class ScoredEntity(InteractiveEntity):
            score: int = Field(default=0, ge=0)

        with self.assertRaises(ValidationError):
            ScoredEntity(score=-5)

    def test_le_constraint_rejects_above_maximum(self):
        """FAILS: Field(le=100) raises pydantic.ValidationError,
        not chat_workflow.ValidationError."""

        class ScoredEntity(InteractiveEntity):
            score: int = Field(default=0, ge=0, le=100)

        with self.assertRaises(ValidationError):
            ScoredEntity(score=200)

    def test_multiple_constraints_reject_invalid(self):
        """FAILS: Multiple Field constraints raise pydantic.ValidationError,
        not chat_workflow.ValidationError."""

        class RangedEntity(InteractiveEntity):
            value: float = Field(default=0.0, ge=-10.0, le=10.0)

        with self.assertRaises(ValidationError):
            RangedEntity(value=-20.0)

        with self.assertRaises(ValidationError):
            RangedEntity(value=20.0)

    def test_valid_field_data_passes_validation(self):
        """Valid data passes through Field constraints without error."""

        class NamedEntity(InteractiveEntity):
            name: str = Field(default="", min_length=1)

        entity = NamedEntity(name="valid name")
        self.assertEqual(entity.name, "valid name")

    def test_valid_numeric_data_passes_validation(self):
        """Valid numeric data passes through Field constraints."""

        class ScoredEntity(InteractiveEntity):
            score: int = Field(default=0, ge=0, le=100)

        entity = ScoredEntity(score=50)
        self.assertEqual(entity.score, 50)

        entity = ScoredEntity(score=0)
        self.assertEqual(entity.score, 0)

        entity = ScoredEntity(score=100)
        self.assertEqual(entity.score, 100)


class TestInteractiveEntityValidationRules(unittest.TestCase):
    """_validation_rules usage: natural-language rules for prompt injection
    and runtime validation.

    Expected behavior:
      - _validation_rules is accessible on the class (already works)
      - _validation_rules is automatically enforced at instantiation
        (not yet implemented — will FAIL)
    """

    def test_validation_rules_accessible_on_class(self):
        """_validation_rules is a class attribute accessible for prompt injection."""

        class TaskEntity(InteractiveEntity):
            _validation_rules: str = "title must be non-empty and at most 100 characters"
            title: str = ""

        entity = TaskEntity()
        self.assertEqual(
            entity._validation_rules,
            "title must be non-empty and at most 100 characters",
        )
        self.assertTrue(hasattr(TaskEntity, "_validation_rules"))

    def test_validation_rules_defaults_to_empty_string(self):
        """Subclasses without _validation_rules default to empty string."""

        class SimpleEntity(InteractiveEntity):
            name: str = ""

        entity = SimpleEntity()
        self.assertEqual(entity._validation_rules, "")

    def test_validation_rules_auto_enforced_on_instantiation(self):
        """FAILS: _validation_rules should be automatically enforced at
        instantiation time, but no such mechanism exists yet.

        When implemented, the validation system should read _validation_rules
        and apply the described constraints at runtime.
        """

        class TaskEntity(InteractiveEntity):
            _validation_rules: str = "title must be non-empty"
            title: str = ""

        # Empty title violates _validation_rules — should raise ValidationError
        with self.assertRaises(ValidationError):
            TaskEntity(title="")

    def test_validation_rules_numeric_rules_enforced(self):
        """FAILS: Numeric _validation_rules should be automatically enforced.

        When implemented, the validation system should parse rules
        like 'priority must be between 1 and 5' and enforce them.
        """

        class PriorityEntity(InteractiveEntity):
            _validation_rules: str = "priority must be between 1 and 5 (inclusive)"
            priority: int = 3

        # Out-of-range value violates _validation_rules
        with self.assertRaises(ValidationError):
            PriorityEntity(priority=10)

    def test_validation_rules_valid_data_passes(self):
        """Data that satisfies _validation_rules passes without error.

        This will PASS once auto-enforcement is implemented.
        For now it passes trivially because no enforcement exists.
        """

        class TaskEntity(InteractiveEntity):
            _validation_rules: str = "title must be non-empty"
            title: str = ""

        # This should work — _validation_rules doesn't auto-enforce yet
        entity = TaskEntity(title="valid title")
        self.assertEqual(entity.title, "valid title")


class TestInteractiveEntityModelValidation(unittest.TestCase):
    """Model-level validation: @model_validator for cross-field business rules.

    Expected behavior:
      - Business rules from _validation_rules are enforced by model_validator
      - chat_workflow.ValidationError is raised for violations
      - The validation system should discover and run model_validators
    """

    def test_model_validator_enforces_business_rules(self):
        """FAILS: Subclass with a @model_validator incorporating
        _validation_rules should raise chat_workflow.ValidationError.

        When implemented, the validation system should enforce rules
        defined in _validation_rules as model-level validators.
        """

        class TaskEntity(InteractiveEntity):
            _validation_rules: str = "if priority > 5, description is required"
            title: str = ""
            priority: int = 1
            description: str = ""

            # The validation system will generate this or
            # expect it to be defined by the subclass

        # High priority without description violates business rules
        # This will fail because no model_validator exists yet
        with self.assertRaises(ValidationError):
            TaskEntity(title="urgent", priority=8, description="")

    def test_model_validator_cross_field_validation(self):
        """FAILS: Cross-field validation rules enforced via model_validator.

        When implemented, rules like 'end_date must be after start_date'
        should be enforceable via _validation_rules.
        """

        class DatedEntity(InteractiveEntity):
            _validation_rules: str = "end_date must be after start_date"
            start: int = 0
            end: int = 0

        # End before start violates rules
        with self.assertRaises(ValidationError):
            DatedEntity(start=10, end=5)

    def test_valid_model_data_passes(self):
        """Data satisfying all business rules passes validation."""

        class TaskEntity(InteractiveEntity):
            _validation_rules: str = "title must be non-empty"
            title: str = ""

        entity = TaskEntity(title="my task")
        self.assertEqual(entity.title, "my task")


class TestInteractiveEntityValidationSystem(unittest.TestCase):
    """System-level tests for the validation infrastructure.

    Expected behavior:
      - All validation (field-level, model-level) raises
        chat_workflow.ValidationError consistently
      - The validation system integrates with Pydantic's validation pipeline
      - Subclasses can override or extend validation behavior
    """

    def test_validation_system_uses_chat_workflow_validation_error(self):
        """FAILS: The validation system should convert pydantic.ValidationError
        to chat_workflow.ValidationError for consistent error handling.

        Currently, Field constraint violations raise pydantic.ValidationError
        which is NOT a subclass of chat_workflow.ValidationError.
        """

        class NamedEntity(InteractiveEntity):
            name: str = Field(default="", min_length=1)

        try:
            NamedEntity(name="")
            self.fail("Expected ValidationError")
        except ValidationError:
            pass  # Expected — this IS a chat_workflow.ValidationError
        except Exception as e:
            # Currently this path is taken — pydantic.ValidationError
            self.fail(f"Expected chat_workflow.ValidationError, got {type(e).__name__}: {e}")

    def test_validation_rules_inherited_by_subclasses(self):
        """FAILS: _validation_rules should be inherited by subclasses.

        When the validation system is implemented, a subclass
        should inherit and enforce its parent's _validation_rules.
        """

        class BaseTask(InteractiveEntity):
            _validation_rules: str = "title is required"
            title: str = ""

        class ExtendedTask(BaseTask):
            description: str = ""

        # ExtendedTask should enforce BaseTask's _validation_rules
        with self.assertRaises(ValidationError):
            ExtendedTask(title="", description="has no title")

    def test_validation_error_message_contains_context(self):
        """FAILS: ValidationError should include context about which rule
        was violated and which field was involved.

        When implemented, error messages from _validation_rules enforcement
        should reference the specific rule that was violated.
        """

        class TaskEntity(InteractiveEntity):
            _validation_rules: str = "title must be non-empty"
            title: str = ""

        try:
            TaskEntity(title="")
            self.fail("Expected ValidationError")
        except ValidationError as e:
            error_msg = str(e).lower()
            # The error message should reference the violated rule
            self.assertIn("title", error_msg)
            self.assertIn("non-empty", error_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
