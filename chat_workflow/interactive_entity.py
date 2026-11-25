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

# Regex patterns for parsing _validation_rules. Named groups name what each
# capture is, so the enforcement reads the rule rather than a position.
_NON_EMPTY_RE = re.compile(r"(?P<field>\w+)\s+must be non-empty")
_REQUIRED_RE = re.compile(r"(?P<field>\w+)\s+is required")
_BETWEEN_RE = re.compile(
    r"(?P<field>\w+)\s+must be between\s+(?P<lower>\d+)\s+and\s+(?P<upper>\d+)"
    r"\s*(?:\(inclusive\))?"
)
_AFTER_RE = re.compile(r"(?P<field>\w+)\s+must be after\s+(?P<other>\w+)")
_CONDITIONAL_RE = re.compile(
    r"if\s+(?P<field>\w+)\s*>\s*(?P<threshold>\d+)\s*,\s*(?P<required>\w+)\s+is required"
)


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
        handlers = (
            self._enforce_conditional,
            self._enforce_between,
            self._enforce_after,
            self._enforce_presence,
        )
        for handler in handlers:
            if handler(rule):
                return

    def _enforce_conditional(self, rule: str) -> bool:
        """Enforce ``if X > N, Y is required``."""
        m = _CONDITIONAL_RE.fullmatch(rule)
        if not m:
            return False
        field_x = self._resolve_field_name(m.group("field"))
        threshold = int(m.group("threshold"))
        field_y = self._resolve_field_name(m.group("required"))
        if (
            field_x is not None
            and field_y is not None
            and getattr(self, field_x, 0) > threshold
            and not getattr(self, field_y, "")
        ):
            self._violated(rule)
        return True

    def _enforce_between(self, rule: str) -> bool:
        """Enforce ``X must be between A and B``."""
        m = _BETWEEN_RE.fullmatch(rule)
        if not m:
            return False
        field_name = self._resolve_field_name(m.group("field"))
        lower = int(m.group("lower"))
        upper = int(m.group("upper"))
        if field_name is not None:
            val = getattr(self, field_name, 0)
            if not (lower <= val <= upper):
                self._violated(rule)
        return True

    def _enforce_after(self, rule: str) -> bool:
        """Enforce ``X must be after Y``."""
        m = _AFTER_RE.fullmatch(rule)
        if not m:
            return False
        field_x = self._resolve_field_name(m.group("field"))
        field_y = self._resolve_field_name(m.group("other"))
        if (
            field_x is not None
            and field_y is not None
            and not (getattr(self, field_x, 0) > getattr(self, field_y, 0))
        ):
            self._violated(rule)
        return True

    def _enforce_presence(self, rule: str) -> bool:
        """Enforce ``X must be non-empty`` and ``X is required``."""
        for pattern in (_NON_EMPTY_RE, _REQUIRED_RE):
            m = pattern.fullmatch(rule)
            if m:
                self._require_present(self._resolve_field_name(m.group("field")), rule)
                return True
        return False

    @staticmethod
    def _violated(rule: str) -> None:
        raise ValidationError(f"Validation rule violated: '{rule}'")

    def _require_present(self, field_name: str | None, rule: str) -> None:
        """Raise when a resolved field holds an empty value."""
        if field_name is not None and not getattr(self, field_name, ""):
            self._violated(rule)
