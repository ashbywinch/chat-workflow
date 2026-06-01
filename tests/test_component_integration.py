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
from workflows.workflow.domain_spec import (
    ComponentDomainField,
    ComponentDomainSpec,
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
                "Structured meeting minutes that capture what happened, "
                "decisions made, and action items assigned"
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
                "Structured meeting minutes that capture what happened, "
                "decisions made, and action items assigned"
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
                    domain_origin=(
                        "Attendees can immediately understand decisions made"
                    ),
                ),
            ],
        )

    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_create_returns_component_with_domain_spec_fields(
        self, mock_explore, mock_design
    ):
        """Component.create should return a Component with fields from domain spec."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure

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
                "Structured meeting minutes that capture what happened, "
                "decisions made, and action items assigned",
            )
            self.assertEqual(result.expert_role, "Meeting Minutes Administrator")
            self.assertEqual(result.component_type, "artifact_producing")
            self.assertEqual(result.model_class, "MeetingMinutes")
            self.assertIsInstance(result.code_path, Path)
            self.assertIn("meetingminutes", str(result.code_path).lower())

    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_explore_called_with_responsibilities(self, mock_explore, mock_design):
        """Component.create should pass responsibilities to explore()."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure

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

    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_default_output_dir(self, mock_explore, mock_design):
        """When no output_dir given, defaults to cwd/workflows/{name}/."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure

        responsibilities = self._make_responsibilities()
        session = self._make_session()

        with patch.object(Path, "cwd", return_value=Path("/tmp")):
            result = Component.create(
                requirements=responsibilities,
                session=session,
            )

            self.assertIsInstance(result, Component)
            self.assertIn("meetingminutes", str(result.code_path).lower())

    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_explore_echoes_message(self, mock_explore, mock_design):
        """Component.create should echo an exploring message."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            session.io.echo.assert_any_call(
                "Exploring domain: MeetingMinutes..."
            )

    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_design_called_with_domain_spec(self, mock_explore, mock_design):
        """Component.create should pass domain_spec from Phase 1 to Phase 2."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure

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

    @patch.object(ComponentStructure, "design")
    @patch.object(ComponentDomainSpec, "explore")
    def test_design_echoes_message(self, mock_explore, mock_design):
        """Component.create should echo a designing message."""
        domain_spec = self._make_domain_spec()
        mock_explore.return_value = domain_spec
        structure = self._make_structure()
        mock_design.return_value = structure

        responsibilities = self._make_responsibilities()

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            Component.create(
                requirements=responsibilities,
                session=session,
                output_dir=Path(tmpdir),
            )

            session.io.echo.assert_any_call(
                "Designing structure: MeetingMinutes..."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
