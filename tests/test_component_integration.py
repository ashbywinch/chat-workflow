"""Integration tests for Component.create with ComponentResponsibilities."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from workflows.workflow.component import Component
from workflows.workflow.component_responsibilities import (
    ComponentResponsibilities,
)
from workflows.workflow.component_source_code import ComponentSourceCode
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
    StructValidator,
)


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


class TestComponentCreateWithResponsibilities(unittest.TestCase):
    """Component.create() with ComponentResponsibilities (Phase 1 skeleton)."""

    def _make_session(self) -> Session:
        return Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    def _make_responsibilities(self, **overrides) -> ComponentResponsibilities:
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
        )
        kwargs.update(overrides)
        return ComponentResponsibilities(**kwargs)

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
                "Someone who missed the meeting can catch up in two minutes",
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
                StructValidator(
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
        return ComponentSourceCode(
            code=(
                "from __future__ import annotations\n"
                "from pydantic import BaseModel\n"
                "\n"
                "class MeetingMinutes(BaseModel):\n"
                "    meeting_date: str\n"
                "    attendees: list[str]\n"
            ),
        )

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_create_returns_component_with_domain_spec_fields(
        self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code
    ):
        """Component.create should return a Component with fields from domain spec."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            result = Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

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

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_explore_called_with_responsibilities(
        self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code
    ):
        """Component.create should pass responsibilities to explore()."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            mock_explore.assert_called_once_with(
                responsibilities=responsibilities,
                session=session,
            )

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_default_output_dir(self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code):
        """When no output_dir given, defaults to cwd/workflows/{name}/."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()
        session = self._make_session()

        with patch.object(Path, "cwd", return_value=Path("/tmp")):
            result = Component.create(
                requirements=responsibilities,
                session=session,
            )

            self.assertIsInstance(result, Component)
            self.assertIn("meetingminutes", str(result.code_path).lower())

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_explore_echoes_message(self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code):
        """Component.create should echo an exploring message."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            session.io.echo.assert_any_call("Exploring domain: MeetingMinutes...")

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_design_called_with_domain_spec(
        self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code
    ):
        """Component.create should pass domain_spec from Phase 1 to Phase 2."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            mock_design.assert_called_once_with(
                domain_spec=domain_spec,
                session=session,
            )

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_design_echoes_message(self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code):
        """Component.create should echo a designing message."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            session.io.echo.assert_any_call("Designing structure: MeetingMinutes...")

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_gather_called_with_domain_spec_and_structure(
        self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code
    ):
        """Phase 3: gather() should receive domain_spec and structure."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            mock_gather.assert_called_once_with(
                domain_spec=domain_spec,
                structure=structure,
                session=session,
            )

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_gather_echoes_message(self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code):
        """Phase 3: Component.create should echo a gathering message."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            session.io.echo.assert_any_call("Gathering interaction context: MeetingMinutes...")

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_generate_called_with_design_spec(
        self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code
    ):
        """Phase 4: generate() should receive a ComponentDesignSpec."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args.kwargs
            self.assertIn("design_spec", call_kwargs)
            self.assertEqual(call_kwargs["design_spec"].domain_spec, domain_spec)
            self.assertEqual(call_kwargs["design_spec"].structure, structure)
            self.assertEqual(
                call_kwargs["design_spec"].interaction_context,
                interaction_context,
            )
            self.assertEqual(call_kwargs["session"], session)

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_generate_echoes_message(self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code):
        """Phase 4: Component.create should echo a generating message."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            session.io.echo.assert_any_call("Generating component code: MeetingMinutes...")

    @patch("workflows.workflow.component.verify_code")
    @patch.object(ComponentSourceCode, "generate")
    @patch.object(ComponentInteractionContext, "gather")
    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_code_verified_and_written(self, mock_explore, mock_design, mock_gather, mock_generate, mock_verify_code):
        """Code should be verified and written to disk."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure
        interaction_context = self._make_interaction_context()
        mock_gather.return_value = interaction_context
        generated = self._make_generated_component()
        mock_generate.return_value = generated
        mock_verify_code.side_effect = lambda code: code

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            mock_verify_code.assert_called_once_with(generated.code)
            expected_path = Path(tmpdir) / "meetingminutes.py"
            self.assertTrue(expected_path.exists())
            self.assertEqual(expected_path.read_text(), generated.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
