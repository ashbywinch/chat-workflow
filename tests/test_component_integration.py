"""Integration tests for Component.create with ComponentSpec."""

import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override
from unittest.mock import MagicMock

from pyfakefs.fake_filesystem_unittest import TestCase

from chat_workflow import Session, SessionLog
from tests.sample_code import VALID_COMPONENT_CODE
from workflows.workflow.component import Component
from workflows.workflow.component_source_code import ComponentSourceCode
from workflows.workflow.component_spec import (
    ComponentSpec,
)
from workflows.workflow.domain_spec import (
    ComponentDomainField,
    ComponentDomainSpec,
)
from workflows.workflow.interaction_context import (
    ComponentInteractionContext,
)
from workflows.workflow.structure import (
    ComponentStructure,
    StructField,
    ValidationRule,
)


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


class _StepRecorder:
    """A step fake that records its keyword arguments and returns a fixture."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


class _VerifierRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, code: str) -> str:
        self.calls.append(code)
        return code


class _DomainFake:
    def __init__(self, result: Any) -> None:
        self.explore = _StepRecorder(result)


class _StructureFake:
    def __init__(self, result: Any) -> None:
        self.design = _StepRecorder(result)


class _InteractionFake:
    def __init__(self, result: Any) -> None:
        self.gather = _StepRecorder(result)


class _SourceFake:
    def __init__(self, result: Any) -> None:
        self.generate = _StepRecorder(result)


@dataclass
class _Pipeline:
    domain: _DomainFake
    structure: _StructureFake
    interaction: _InteractionFake
    source: _SourceFake
    verifier: _VerifierRecorder


class _NoRuleJudge(TestCase):
    """Construct real records without the LLM judge, using the documented skip flag."""

    _MODELS = (ComponentSpec, ComponentDomainSpec, Component, ComponentSourceCode)

    @override
    def setUp(self) -> None:
        self.setUpPyfakefs()
        self._saved_skip = {model: model._skip_llm_validation for model in self._MODELS}
        for model in self._MODELS:
            model._skip_llm_validation = True

    @override
    def tearDown(self) -> None:
        for model, value in self._saved_skip.items():
            model._skip_llm_validation = value


class TestComponentCreateWithResponsibilities(_NoRuleJudge):
    """Component.create() with ComponentSpec (Phase 1 skeleton)."""

    def _make_session(self) -> tuple[Session, MagicMock]:
        io = MagicMock()
        return Session(io=io, state=SessionLog(), config=FakeConfig()), io

    def _make_responsibilities(self, **overrides) -> ComponentSpec:
        kwargs = dict(
            name="MeetingMinutes",
            purpose="Capture and structure meeting discussions into actionable minutes",
            scope_description=(
                "Represents the full meeting minutes lifecycle: capturing "
                "discussions, decisions, action items. Does NOT represent "
                "scheduling or calendar management."
            ),
            required_inputs=["MeetingAgenda", "AttendeeList"],
            component_type="artifact_producing",
            owner_role="Owner",
        )
        kwargs.update(overrides)
        return ComponentSpec.model_validate(kwargs)

    def _make_domain_spec(self) -> ComponentDomainSpec:
        return ComponentDomainSpec(
            name="MeetingMinutes",
            description=(
                "Structured meeting minutes that capture what happened, decisions made, and action items assigned"
            ),
            fields=[
                ComponentDomainField(
                    name="meeting_date",
                    domain_description="When the meeting took place",
                    field_type_hint="date",
                ),
                ComponentDomainField(
                    name="attendees",
                    domain_description="People who attended the meeting",
                    field_type_hint="list of person names",
                ),
            ],
            what_good_looks_like=[
                "Attendees can immediately understand decisions made",
                "Someone who missed the meeting can understand what happened without asking",
            ],
            expert_role="Meeting Minutes Administrator",
        )

    def _make_structure(self) -> ComponentStructure:
        return ComponentStructure(
            description=(
                "Structured meeting minutes that capture what happened, decisions made, and action items assigned"
            ),
            base_class="BaseModel",
            fields=[
                StructField(
                    name="meeting_date",
                    type_expr="str",
                    field_def_kwargs={"description": "When the meeting took place"},
                ),
                StructField(
                    name="attendees",
                    type_expr="list[str]",
                    field_def_kwargs={"description": "People who attended the meeting"},
                ),
            ],
            model_validators=[
                ValidationRule(
                    rule="description must not exceed 3 sentences",
                    domain_origin=("Attendees can immediately understand decisions made"),
                ),
            ],
        )

    def _make_interaction_context(self) -> ComponentInteractionContext:
        return ComponentInteractionContext(
            must_prioritize=["Always ask about decisions early"],
            auto_suggest=["Suggest action item owners based on topic"],
            tone_preference="Professional but friendly",
            user_pain_points=["Users often forget to list attendees"],
        )

    def _make_generated_component(self) -> ComponentSourceCode:
        return ComponentSourceCode(code=VALID_COMPONENT_CODE)

    def _pipeline(self) -> _Pipeline:
        return _Pipeline(
            domain=_DomainFake(self._make_domain_spec()),
            structure=_StructureFake(self._make_structure()),
            interaction=_InteractionFake(self._make_interaction_context()),
            source=_SourceFake(self._make_generated_component()),
            verifier=_VerifierRecorder(),
        )

    def _run(self, pipeline: _Pipeline, *, output_dir: Path | None = Path("/out"), cwd=None):
        requirements = self._make_responsibilities()
        session, io = self._make_session()
        result = Component.create(
            requirements=requirements,
            session=session,
            output_dir=output_dir,
            cwd=cwd,
            domain_spec=pipeline.domain,
            structure=pipeline.structure,
            interaction=pipeline.interaction,
            source=pipeline.source,
            verifier=pipeline.verifier,
        )
        return result, requirements, session, io, pipeline

    def test_create_returns_component_with_domain_spec_fields(self):
        """Component.create should return a Component with fields from domain spec."""
        result, _, _, _, _ = self._run(self._pipeline())

        self.assertIsInstance(result, Component)
        self.assertEqual(result.name, "MeetingMinutes")
        self.assertEqual(
            result.purpose,
            "Structured meeting minutes that capture what happened, decisions made, and action items assigned",
        )
        self.assertEqual(result.expert_role, "Meeting Minutes Administrator")
        self.assertEqual(result.component_type, "artifact_producing")
        self.assertEqual(result.model_class, "MeetingMinutes")
        self.assertIsInstance(result.code_path, Path)
        self.assertIn("meetingminutes", str(result.code_path).lower())

    def test_explore_called_with_responsibilities(self):
        """Component.create should pass responsibilities to explore()."""
        _, requirements, session, _, pipeline = self._run(self._pipeline())

        call = pipeline.domain.explore.calls[0]
        self.assertEqual(call["responsibilities"], requirements)
        self.assertIs(call["session"], session)

    def test_default_output_dir(self):
        """When no output_dir given, defaults to cwd/workflows/{name}/."""
        result, _, _, _, _ = self._run(self._pipeline(), output_dir=None, cwd=Path("/work"))

        self.assertIsInstance(result, Component)
        self.assertIn("meetingminutes", str(result.code_path).lower())

    def test_explore_echoes_message(self):
        """Component.create should echo an exploring message."""
        _, _, _, io, _ = self._run(self._pipeline())

        io.echo.assert_any_call("Exploring domain: MeetingMinutes...")

    def test_design_called_with_domain_spec(self):
        """Component.create should pass domain_spec from Phase 1 to Phase 2."""
        _, _, session, _, pipeline = self._run(self._pipeline())

        call = pipeline.structure.design.calls[0]
        self.assertEqual(call["domain_spec"], pipeline.domain.explore.result)
        self.assertIs(call["session"], session)

    def test_design_echoes_message(self):
        """Component.create should echo a designing message."""
        _, _, _, io, _ = self._run(self._pipeline())

        io.echo.assert_any_call("Designing structure: MeetingMinutes...")

    def test_gather_called_with_domain_spec_and_structure(self):
        """Phase 3: gather() should receive domain_spec and structure."""
        _, _, session, _, pipeline = self._run(self._pipeline())

        call = pipeline.interaction.gather.calls[0]
        self.assertEqual(call["domain_spec"], pipeline.domain.explore.result)
        self.assertEqual(call["structure"], pipeline.structure.design.result)
        self.assertIs(call["session"], session)

    def test_gather_echoes_message(self):
        """Phase 3: Component.create should echo a gathering message."""
        _, _, _, io, _ = self._run(self._pipeline())

        io.echo.assert_any_call("Gathering interaction context: MeetingMinutes...")

    def test_generate_called_with_design_spec(self):
        """Phase 4: generate() should receive a ComponentDesignSpec."""
        _, _, session, _, pipeline = self._run(self._pipeline())

        call = pipeline.source.generate.calls[0]
        design_spec = call["design_spec"]
        self.assertEqual(design_spec.domain_spec, pipeline.domain.explore.result)
        self.assertEqual(design_spec.structure, pipeline.structure.design.result)
        self.assertEqual(
            design_spec.interaction_context,
            pipeline.interaction.gather.result,
        )
        self.assertIs(call["session"], session)

    def test_generate_echoes_message(self):
        """Phase 4: Component.create should echo a generating message."""
        _, _, _, io, _ = self._run(self._pipeline())

        io.echo.assert_any_call("Generating component code: MeetingMinutes...")

    def test_code_verified_and_written(self):
        """Code should be verified and written to disk."""
        result, _, _, _, pipeline = self._run(self._pipeline())

        self.assertEqual(pipeline.verifier.calls, [pipeline.source.generate.result.code])
        self.assertTrue(result.code_path.exists())
        self.assertEqual(result.code_path.read_text(), pipeline.source.generate.result.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
