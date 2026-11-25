"""Unit tests for ComponentSourceCode programmatic validation rules.

Tests the @model_validator(mode="after") on ComponentSourceCode that checks:
- Valid Python syntax
- Exactly one BaseModel subclass
- @atomic_workflow_step decorator present

Also tests that new _validation_rules entries are collected.
"""

# lucidlint: ignore-file fakefs validating a component runs `ruff` as a subprocess over the candidate
# module (reading this repo's pyproject.toml) and resolves each generated import with
# importlib.util.find_spec — a faked filesystem would defeat exactly what these tests assert

import unittest

from pydantic import ValidationError as PydanticValidationError

from tests.sample_code import (
    AUXILIARY_RETURN_CODE,
    CREATE_UPDATE_CODE,
    INVALID_SYNTAX_CODE,
    MULTI_ARTIFACT_CODE,
    MULTIPLE_BASEMODEL_CODE,
    MULTIPLE_RETURN_TYPES_CODE,
    NESTED_BASEMODEL_CODE,
    NO_BASEMODEL_CODE,
    NO_WORKFLOW_CODE,
    NOOP_VALIDATOR_CODE,
    SECOND_ARTIFACT_CODE,
    VALID_COMPONENT_CODE,
)
from workflows.workflow.component_source_code import ComponentSourceCode


class _DeterministicOnly(ComponentSourceCode):
    """The real model with the LLM judge switched off at its documented seam.

    These tests assert the deterministic rules; a judge call would need an API
    key and would answer the wrong question.
    """

    _skip_llm_validation = True


class TestSourceLines(unittest.TestCase):
    def test_an_element_with_embedded_newlines_is_split_into_lines(self):
        from workflows.workflow.component_source_code import _join_source_lines

        joined = _join_source_lines(["a = 1", "b = 2\nc = 3"])

        self.assertEqual(joined, "a = 1\nb = 2\nc = 3")

class TestComponentSourceCodeProgrammaticValidation(unittest.TestCase):
    """Tests for the programmatic @model_validator on ComponentSourceCode."""

    def test_valid_code_passes_validation(self):
        """Valid code with one BaseModel and @atomic_workflow_step should pass."""
        component = _DeterministicOnly(code=VALID_COMPONENT_CODE)
        self.assertIn("class MinutesDraft", component.code)

    def test_ruff_violation_is_reported_with_rule_and_line(self):
        """A generated ruff error must name the rule and line, not just fail."""
        from chat_workflow.exceptions import ValidationError

        # E501 is not auto-fixed by `_ruff_fix`, so it reaches validation.
        code = VALID_COMPONENT_CODE + "\n# " + "x" * 130 + "\n"
        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        message = str(ctx.exception)
        self.assertIn("Generated code must pass `ruff check`", message)
        self.assertRegex(message, r"generated_component\.py:\d+:.*E501")
        self.assertIn("x" * 20, message)
        self.assertRegex(message, r"line \d+: '# x")

    def test_invalid_syntax_raises_validation_error(self):
        """Invalid Python syntax should raise ValidationError."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=INVALID_SYNTAX_CODE)

        self.assertIn("not valid Python", str(ctx.exception))
        self.assertIn("Offending line", str(ctx.exception))
        self.assertIn("plain ASCII Python", str(ctx.exception))

    def test_no_basemodel_raises_validation_error(self):
        """Code with no BaseModel subclass should raise ValidationError."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=NO_BASEMODEL_CODE)

        self.assertIn("at least one BaseModel", str(ctx.exception))

    def test_multiple_basemodels_raises_validation_error(self):
        """Code with multiple independent BaseModel subclasses should raise."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=MULTIPLE_BASEMODEL_CODE)

        self.assertIn("multiple BaseModel", str(ctx.exception))

    def test_no_atomic_workflow_step_raises_validation_error(self):
        """Code without @atomic_workflow_step decorator should raise ValidationError."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=NO_WORKFLOW_CODE)

        self.assertIn("at least one public method", str(ctx.exception))


class TestSessionPromptPlacement(unittest.TestCase):
    """The framework supplies the session wrapper; generated steps follow it."""

    def test_protocol_text_in_a_step_docstring_is_rejected(self):
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            "When you have enough, show the draft once and ask if it is right.",
            "Return JSON with intent and result.",
        )
        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("response protocol", str(ctx.exception))

    def test_a_generated_session_component_is_rejected(self):
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE + "\n\nclass SessionComponent:\n    pass\n"
        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("must not define a SessionComponent", str(ctx.exception))

    def test_a_generated_greeting_step_is_rejected(self):
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace("def create(", "def greet(")
        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("greeting or closing", str(ctx.exception))


class TestNestedPlainClass(unittest.TestCase):
    """A field naming a plain class must fail before the import does."""

    def test_a_plain_nested_class_is_rejected(self):
        code = '''
from typing import ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class Item:
    description = ""


class Draft(LLMValidated):
    items: list[Item] = Field(..., description="Items")

    _validation_rules: ClassVar[list[str]] = ["Items are specific"]

    @atomic_workflow_step
    @classmethod
    def create(cls) -> "Draft":
        """Collect the items."""
        ...
'''

        with self.assertRaises(PydanticValidationError) as ctx:
            ComponentSourceCode(code=code)

        self.assertIn("Item", str(ctx.exception))
        self.assertIn("LLMValidated", str(ctx.exception))


class TestAllOptionalNestedModel(unittest.TestCase):
    """A nested model with no required field accepts an empty object."""

    def test_an_all_optional_nested_model_is_rejected(self):
        code = '''
from typing import ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class Item(LLMValidated):
    """One item."""

    label: str | None = Field(default=None, description="Label")


class Draft(LLMValidated):
    """The artifact."""

    items: list[Item] = Field(..., description="Items")

    _validation_rules: ClassVar[list[str]] = ["Items are specific"]

    @atomic_workflow_step
    @classmethod
    def create(cls) -> "Draft":
        """Collect the items."""
        ...
'''

        with self.assertRaises(PydanticValidationError) as ctx:
            ComponentSourceCode(code=code)

        self.assertIn("Item", str(ctx.exception))
        self.assertIn("empty object", str(ctx.exception))


class TestPrimaryArtifactRule(unittest.TestCase):
    """One primary artifact; nested BaseModels are allowed when used as fields."""

    def test_rule_runs_in_model_validation_without_judge(self):
        """The check is part of the component's validation, not the test harness."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=SECOND_ARTIFACT_CODE)

        self.assertIn("outside the primary artifact", str(ctx.exception))
        self.assertIn("Invoice", str(ctx.exception))

    def test_nested_basemodel_passes_in_model_validation(self):
        """Nested models are accepted by the component's own validation."""
        component = _DeterministicOnly(code=NESTED_BASEMODEL_CODE)

        self.assertIn("class ActionItem", component.code)

    def test_nested_basemodel_is_allowed(self):
        """A BaseModel used as a field of the primary model is not an artifact."""
        component = _DeterministicOnly(code=NESTED_BASEMODEL_CODE)

        self.assertIn("class ActionItem", component.code)
        self.assertIn("class MinutesDraft", component.code)

    def test_second_independent_artifact_is_rejected(self):
        """A BaseModel not reachable from the primary model is a second artifact."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=SECOND_ARTIFACT_CODE)

        self.assertIn("outside the primary artifact", str(ctx.exception))
        self.assertIn("Invoice", str(ctx.exception))

    def test_multiple_return_types_are_rejected(self):
        """Independent returned models mean two artifacts."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=MULTIPLE_RETURN_TYPES_CODE)

        self.assertIn("independent BaseModel types", str(ctx.exception))

    def test_create_and_update_on_one_artifact_are_allowed(self):
        """A component's methods may create and update the same artifact."""
        component = _DeterministicOnly(code=CREATE_UPDATE_CODE)

        self.assertIn("def create", component.code)
        self.assertIn("def update", component.code)

    def test_auxiliary_return_type_is_not_a_second_artifact(self):
        """A helper type a method returns is not the component's artifact."""
        component = _DeterministicOnly(code=AUXILIARY_RETURN_CODE)

        self.assertIn("class RevisionSummary", component.code)
        self.assertIn("class MeetingMinutes", component.code)

    def test_multi_artifact_code_is_rejected(self):
        """Two independent artifacts with no single returned primary are rejected."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=MULTI_ARTIFACT_CODE)

        self.assertIn("multiple BaseModel subclasses", str(ctx.exception))


class TestDeterministicConstraints(unittest.TestCase):
    """Schema-supported constraints are filled in or rejected deterministically."""

    def test_required_str_gets_min_length(self):
        """A required str field without min_length is rejected for the model to fix."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            '..., description="When the meeting took place", min_length=1',
            '..., description="When the meeting took place"',
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("Required string field(s) must declare min_length=1", str(ctx.exception))

    def test_field_without_ellipsis_gets_min_length(self):
        """A required Field(description=...) with no default is still required."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            '..., description="When the meeting took place", min_length=1',
            'description="When the meeting took place"',
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("Required string field(s) must declare min_length=1", str(ctx.exception))

    def test_explicit_min_length_zero_is_preserved(self):
        """min_length=0 means empty is allowed and must not be rewritten."""
        code = VALID_COMPONENT_CODE.replace("min_length=1", "min_length=0")

        component = _DeterministicOnly(code=code)

        self.assertNotIn("min_length=1", component.code)
        self.assertIn("min_length=0", component.code)

    def test_workflow_docstring_repeating_class_docstring_is_rejected(self):
        """A shared phrase between the class and workflow docstrings is duplication."""
        from chat_workflow.exceptions import ValidationError

        code = '''from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow_step


class MinutesDraft(BaseModel):
    """Structured meeting minutes that capture what happened, decisions made, and action items assigned."""

    meeting_date: str = Field(..., description="When the meeting took place", min_length=1)

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MinutesDraft:
        """You help someone create structured meeting minutes that capture what happened,
        decisions made, and action items assigned. Ask about the date and decisions."""
        ...
'''

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("repeats the class docstring", str(ctx.exception))

    def test_workflow_docstring_repeating_field_description_is_rejected(self):
        """A shared phrase with a field description is duplication too."""
        from chat_workflow.exceptions import ValidationError

        code = '''from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow_step


class MinutesDraft(BaseModel):
    """Structured meeting minutes."""

    key_decisions: list[str] = Field(
        ..., description="Important decisions agreed upon during the meeting", min_length=1
    )

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MinutesDraft:
        """You help someone capture important decisions agreed upon during the meeting."""
        ...
'''

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("description of field 'key_decisions'", str(ctx.exception))

    def test_shared_phrase_across_a_sentence_boundary_is_allowed(self):
        """Punctuation is a boundary: two sentences are not a repeated phrase."""
        code = VALID_COMPONENT_CODE.replace(
            'meeting_date: str = Field(\n'
            '        ..., description="When the meeting took place", min_length=1\n'
            '    )',
            'owner: str = Field(\n'
            '        ...,\n'
            "        description=\"It may be the user. 'Owner' only when the "
            'artifact is the owner\'s own work.",\n'
            '        min_length=1,\n'
            '    )',
        ).replace(
            '"""You help someone turn sketchy notes into minutes. Start from what',
            '"""It may be the user owner. Ask who is accountable. Start from what',
        )

        component = _DeterministicOnly(code=code)

        self.assertIn("It may be the user owner", component.code)

    def test_unused_import_is_rejected_by_ruff(self):
        """Ruff findings are fed back, not auto-fixed."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            "from typing import ClassVar",
            "import os\nfrom typing import ClassVar",
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("F401", str(ctx.exception))

    def test_unfixable_ruff_violation_is_rejected(self):
        """Ruff violations it cannot fix (long lines) still fail validation."""
        from chat_workflow.exceptions import ValidationError

        long_line = "very " * 40
        code = VALID_COMPONENT_CODE.replace(
            '"""You help someone turn sketchy notes into minutes. Start from what',
            f'"""You help someone turn sketchy notes into minutes {long_line} Start from what',
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("ruff", str(ctx.exception))
        self.assertIn("E501", str(ctx.exception))

    def test_legacy_framework_import_is_rejected(self):
        """The old path is rejected, not silently rewritten."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            "from chat_workflow import atomic_workflow_step",
            "from chat_workflow.workflow import atomic_workflow_step",
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("cannot be resolved", str(ctx.exception))

    def test_an_invented_dependency_is_still_rejected(self):
        """Rewriting is for known aliases only; a made-up package still fails."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            "from chat_workflow import atomic_workflow_step",
            "from invented_package_xyz import atomic_workflow_step",
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("invented_package_xyz", str(ctx.exception))


    def test_noop_model_validator_is_rejected(self):
        """A validator that only does `pass` enforces nothing."""
        from chat_workflow.exceptions import ValidationError

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=NOOP_VALIDATOR_CODE)

        self.assertIn("no-op 'pass'", str(ctx.exception))

    def test_a_tuple_description_is_rejected(self):
        """A tuple description is rejected, not joined."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            'description="When the meeting took place"',
            'description=("When the meeting took place", "in local time")',
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("Field description= must be one string", str(ctx.exception))

    def test_a_long_condition_is_wrapped_by_the_formatter(self):
        """A joined condition is the model's most common ruff failure."""
        from workflows.workflow.component_source_code import (
            _ruff_format,
            _ruff_violations,
        )

        code = (
            "def f(self, date):\n"
            "    if self.next_meeting_date and self.meeting_date and "
            "date.fromisoformat(self.next_meeting_date) < "
            "date.fromisoformat(self.meeting_date):\n"
            "        return 1\n"
            "    return 0\n"
        )
        self.assertTrue(any("E501" in line for line in _ruff_violations(code)))

        formatted = _ruff_format(code)

        self.assertFalse(any("E501" in line for line in _ruff_violations(formatted)))

    def test_module_is_exchanged_as_lines(self):
        """The model sends one string per line, so no newline escapes exist."""
        import json

        payload = json.dumps({"code": VALID_COMPONENT_CODE.splitlines()})

        component = ComponentSourceCode.model_validate_json(payload)

        from workflows.workflow.component_source_code import _ruff_format

        self.assertEqual(component.code, _ruff_format(VALID_COMPONENT_CODE))

    def test_schema_asks_for_an_array_of_lines(self):
        schema = ComponentSourceCode.model_json_schema()["properties"]["code"]

        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["items"]["type"], "string")

    def test_wrapped_line_with_a_non_ascii_character_still_fits(self):
        """A non-ASCII character costs six characters once escaped."""
        description = "Summary of the meeting — " + "decision " * 20
        code = VALID_COMPONENT_CODE.replace(
            '"A short description."', f'"{description.strip()}"'
        )

        component = _DeterministicOnly(code=code)

        compile(component.code, "generated", "exec")
        self.assertTrue(
            all(len(line) <= 120 for line in component.code.splitlines()),
            "every wrapped line must fit the limit",
        )

    def test_field_validator_taking_self_is_rejected(self):
        """A field validator must take cls; self is rejected, not renamed."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            "from pydantic import BaseModel, Field",
            "from pydantic import BaseModel, Field, field_validator",
        ).replace(
            "    @atomic_workflow_step\n",
            '    @field_validator("meeting_date")\n'
            "    def _check_date(self, value: str) -> str:\n"
            "        return value.strip()\n\n"
            "    @atomic_workflow_step\n",
            1,
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("takes `self`", str(ctx.exception))

    def test_model_validator_with_field_name_is_rejected(self):
        """`@model_validator('field')` parses but fails at import time."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace(
            "from pydantic import BaseModel, Field",
            "from pydantic import BaseModel, Field, model_validator",
        ).replace(
            "    @atomic_workflow_step\n",
            '    @model_validator("meeting_date")\n'
            "    def _check_date(self) -> MinutesDraft:\n"
            "        return self\n\n"
            "    @atomic_workflow_step\n",
            1,
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("@field_validator('meeting_date')", str(ctx.exception))

    def test_long_field_call_is_wrapped(self):
        """A long single-line Field(...) is split at its arguments."""
        long_description = (
            "The date the meeting took place, written at the top of the first "
            "page of the minutes"
        )
        code = VALID_COMPONENT_CODE.replace(
            '    meeting_date: str = Field(\n'
            '        ..., description="When the meeting took place", min_length=1\n'
            '    )',
            f'    meeting_date: str = Field(..., description="{long_description}", min_length=1)',
        )

        component = _DeterministicOnly(code=code)

        self.assertIn("Field(\n", component.code)
        self.assertNotIn(
            f'description="{long_description}", min_length=1)',
            component.code,
        )

    def test_long_validation_rule_is_rejected(self):
        """A long line ruff cannot format is rejected with E501."""
        from chat_workflow.exceptions import ValidationError

        long_rule = (
            "The raw notes should be sufficiently clear that a person can "
            "reconstruct the meeting's key points, decisions, and action items "
            "later the same day without confusion."
        )
        code = VALID_COMPONENT_CODE.replace(
            '"Each decision records the reason behind it, not only what was decided.",',
            f'"{long_rule}",',
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("E501", str(ctx.exception))

    def test_long_string_inside_multiline_field_is_rejected(self):
        """A long string ruff cannot wrap is rejected with E501."""
        from chat_workflow.exceptions import ValidationError

        long_description = (
            "The date the meeting took place, written at the top of the first "
            "page of the minutes and kept with the record of the meeting"
        )
        code = VALID_COMPONENT_CODE.replace(
            'meeting_date: str = Field(\n'
            '        ..., description="When the meeting took place", min_length=1\n'
            '    )',
            'meeting_date: str = Field(\n'
            '        ...,\n'
            f'        description="{long_description}",\n'
            '        min_length=1,\n'
            '    )',
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("E501", str(ctx.exception))

    def test_private_validation_rules_are_rejected(self):
        """An annotated list[str] is a Pydantic private attribute, not a rule."""
        from chat_workflow.exceptions import ValidationError

        code = VALID_COMPONENT_CODE.replace("from typing import ClassVar\n", "").replace(
            "_validation_rules: ClassVar[list[str]]", "_validation_rules: list[str]"
        )

        with self.assertRaises((ValidationError, PydanticValidationError)) as ctx:
            _DeterministicOnly(code=code)

        self.assertIn("ClassVar[list[str]]", str(ctx.exception))


class TestComponentSourceCodeRulesCollection(unittest.TestCase):
    """Tests that new _validation_rules entries are collected."""

    def test_validation_rules_include_new_entries(self):
        """Judge rules are the judgment calls; deterministic rules are not here."""
        rules = ComponentSourceCode.collect_all_rules()

        self.assertGreaterEqual(len(rules), 6)

        self.assertTrue(
            any("cross-cutting" in r for r in rules),
            "Should have High Cohesion rule about cross-cutting concerns",
        )
        self.assertTrue(
            any("domain-quality" in r for r in rules),
            "Should reserve generated-model validation rules for domain quality",
        )

        self.assertFalse(
            any("description=" in r for r in rules),
            "Field descriptions are deterministic, not judged",
        )
        self.assertFalse(
            any("min_length" in r for r in rules),
            "Length constraints are deterministic, not judged",
        )
        self.assertFalse(
            any("second BaseModel" in r for r in rules),
            "The primary-artifact rule is deterministic, not judged",
        )
        self.assertFalse(
            any("one rule" in r.lower() for r in rules),
            "Meta-rules about this rule list do not belong in generated-code rules",
        )
        self.assertFalse(
            any("every word" in r.lower() for r in rules),
            "Unenforceable conciseness rule was removed",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRuffFeedback(unittest.TestCase):
    """A ruff violation is an output error the model sees and corrects."""

    def test_a_ruff_violation_is_fed_back_and_corrected(self):
        from pydantic_ai.models.function import FunctionModel

        from chat_workflow import AgentAnswer, Conversation, Success
        from tests.conftest import ScriptedModel

        invalid = VALID_COMPONENT_CODE.replace(
            "from typing import ClassVar",
            "import os\nfrom typing import ClassVar",
        )
        script = ScriptedModel(
            [
                {
                    "intent": "success",
                    "message": "Done.",
                    "result": {"code": invalid.splitlines()},
                },
                {
                    "intent": "success",
                    "message": "Done.",
                    "result": {"code": VALID_COMPONENT_CODE.splitlines()},
                },
            ]
        )
        original = ComponentSourceCode._skip_llm_validation
        ComponentSourceCode._skip_llm_validation = True
        self.addCleanup(setattr, ComponentSourceCode, "_skip_llm_validation", original)

        conversation = Conversation(
            model="scripted",
            instructions="Generate the module.",
            output_type=AgentAnswer[ComponentSourceCode],
            model_client=FunctionModel(script),
        )

        turn = conversation.send("Generate the module.")

        self.assertIsInstance(turn.output, Success)
        self.assertNotIn("import os", turn.output.result.code)
        self.assertEqual(len(script.calls), 2)
