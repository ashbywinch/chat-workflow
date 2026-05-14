"""InteractiveEntity base class for self-modifying workflows.

InteractiveEntity is a Pydantic BaseModel subclass that serves as the
foundation for workflows that can be created interactively through
conversation. Subclasses define fields, validation rules, and workflow
methods that are generated dynamically.
"""

from __future__ import annotations

import re
from typing import Any

import pydantic
from pydantic import BaseModel

from chat_workflow.exceptions import ValidationError

# Regex patterns for parsing _validation_rules
_NON_EMPTY_RE = re.compile(r"(\w+)\s+must be non-empty")
_REQUIRED_RE = re.compile(r"(\w+)\s+is required")
_BETWEEN_RE = re.compile(r"(\w+)\s+must be between\s+(\d+)\s+and\s+(\d+)\s*(?:\(inclusive\))?")
_AFTER_RE = re.compile(r"(\w+)\s+must be after\s+(\w+)")
_CONDITIONAL_RE = re.compile(r"if\s+(\w+)\s*>\s*(\d+)\s*,\s*(\w+)\s+is required")


class InteractiveEntity(BaseModel):
    """Base class for interactively-created workflow entities.

    Subclasses define their own fields and validation rules. The
    ``_validation_rules`` class attribute stores natural-language
    validation rules used both for prompt injection (guiding the LLM)
    and runtime validation (enforced by Pydantic).

    Attributes:
        _validation_rules: Natural-language description of validation
            rules for this entity's fields and business logic.
    """

    _validation_rules: str = ""

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except pydantic.ValidationError as e:
            raise ValidationError(str(e)) from e

    def model_post_init(self, __context: Any) -> None:
        """Enforce _validation_rules after initialization."""
        self._enforce_validation_rules()

    def _get_validation_rules(self) -> str:
        return str(self._validation_rules)

    def _resolve_field_name(self, name_from_rule: str) -> str | None:
        model_fields = type(self).model_fields
        if name_from_rule in model_fields:
            return name_from_rule
        for field_name in model_fields:
            if field_name in name_from_rule or name_from_rule in field_name:
                return field_name
        return None

    def _enforce_validation_rules(self) -> None:
        """Parse and enforce all _validation_rules."""
        rules_text = self._get_validation_rules()
        if not rules_text:
            return

        for rule in rules_text.split("\n"):
            rule = rule.strip()
            if not rule:
                continue
            self._enforce_single_rule(rule)

    def _enforce_single_rule(self, rule: str) -> None:
        """Enforce a single validation rule.

        Raises:
            ValidationError: If the rule is violated.
        """
        # if X > N, Y is required
        m = _CONDITIONAL_RE.fullmatch(rule)
        if m:
            field_x = self._resolve_field_name(m.group(1))
            threshold = int(m.group(2))
            field_y = self._resolve_field_name(m.group(3))
            if field_x is not None and field_y is not None:
                val_x = getattr(self, field_x, 0)
                val_y = getattr(self, field_y, "")
                if val_x > threshold and not val_y:
                    raise ValidationError(
                        f"Validation rule violated: '{rule}'"
                    )
            return

        # X must be between A and B
        m = _BETWEEN_RE.fullmatch(rule)
        if m:
            field_name = self._resolve_field_name(m.group(1))
            lower = int(m.group(2))
            upper = int(m.group(3))
            if field_name is not None:
                val = getattr(self, field_name, 0)
                if not (lower <= val <= upper):
                    raise ValidationError(
                        f"Validation rule violated: '{rule}'"
                    )
            return

        # X must be after Y
        m = _AFTER_RE.fullmatch(rule)
        if m:
            field_x = self._resolve_field_name(m.group(1))
            field_y = self._resolve_field_name(m.group(2))
            if field_x is not None and field_y is not None:
                val_x = getattr(self, field_x, 0)
                val_y = getattr(self, field_y, 0)
                if not (val_x > val_y):
                    raise ValidationError(
                        f"Validation rule violated: '{rule}'"
                    )
            return

        # X must be non-empty
        m = _NON_EMPTY_RE.fullmatch(rule)
        if m:
            field_name = self._resolve_field_name(m.group(1))
            if field_name is not None:
                val = getattr(self, field_name, "")
                if not val:
                    raise ValidationError(
                        f"Validation rule violated: '{rule}'"
                    )
            return

        # X is required
        m = _REQUIRED_RE.fullmatch(rule)
        if m:
            field_name = self._resolve_field_name(m.group(1))
            if field_name is not None:
                val = getattr(self, field_name, "")
                if not val:
                    raise ValidationError(
                        f"Validation rule violated: '{rule}'"
                    )
            return