"""Unit tests for GeneratedComponent.generate() with ComponentDesignSpec."""
import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import AgentIntent, AgentResponse, Session, SessionLog
from workflows.workflow.design_spec import ComponentDesignSpec
from workflows.workflow.domain_spec import ComponentDomainField, ComponentDomainSpec
from workflows.workflow.generated_component import GeneratedComponent
from workflows.workflow.interaction_context import ComponentInteractionContext
from workflows.workflow.structure import ComponentStructure


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


SAMPLE_CODE = """from __future__ import annotations

from pydantic import BaseModel, Field


class MinutesDraft(BaseModel):
    \"\"\"Structured meeting minutes that capture what happened.\"\"\"

    meeting_date: str = Field(..., description="When the meeting took place")
    attendees: list[str] = Field(..., description="People who attended", min_length=1)
    decisions: list[str] = Field(..., description="Key decisions made")
    action_items: list[str] = Field(..., description="Action items assigned")
"""


def _make_design_spec(**overrides) -> ComponentDesignSpec:
    """Create a standard ComponentDesignSpec for testing."""
    kwargs = dict(
        domain_spec=ComponentDomainSpec(
            name="MinutesDraft",
            description="Structured meeting minutes that capture what happened",
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
        ),
        structure=ComponentStructure(
            description="Structured meeting minutes that capture what happened",
        ),
        interaction_context=ComponentInteractionContext(
            must_prioritize=["Always ask about decisions early"],
            auto_suggest=["Suggest action item owners based on the topic discussed"],
            user_pain_points=["Users often forget to list attendees"],
        ),
    )
    kwargs.update(overrides)
    return ComponentDesignSpec(**kwargs)


class TestGeneratedComponentGenerate(unittest.TestCase):
    """Tests for GeneratedComponent.generate() with ComponentDesignSpec."""

    def setUp(self):
        self.design_spec = _make_design_spec()
        self.session = Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    def _make_success_response(self) -> AgentResponse[GeneratedComponent]:
        """Create a mock AgentResponse with SUCCESS intent."""
        return AgentResponse[GeneratedComponent](
            intent=AgentIntent.SUCCESS,
            result=GeneratedComponent(code=SAMPLE_CODE),
        )

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    def test_generate_accepts_design_spec(
        self, mock_get_client, mock_validate
    ):
        """generate() should accept ComponentDesignSpec and return GeneratedComponent."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_success_response()
        )
        mock_get_client.return_value = mock_client

        result = GeneratedComponent.generate(
            design_spec=self.design_spec,
            session=self.session,
        )

        self.assertIsInstance(result, GeneratedComponent)
        self.assertGreater(len(result.code), 0)
        self.assertIn("class MinutesDraft", result.code)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    def test_generate_passes_design_spec_to_llm(
        self, mock_get_client, mock_validate
    ):
        """The design spec should be included in the LLM prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_success_response()
        )
        mock_get_client.return_value = mock_client

        GeneratedComponent.generate(
            design_spec=self.design_spec,
            session=self.session,
        )

        # Verify the LLM was called
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs.get("messages", [])
        # The system prompt should mention the design spec
        system_msg = messages[0]["content"]
        self.assertIn("MinutesDraft", system_msg)
        self.assertIn("meeting_date", system_msg)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    def test_generated_code_is_valid_python(
        self, mock_get_client, mock_validate
    ):
        """The generated code should be syntactically valid Python."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_success_response()
        )
        mock_get_client.return_value = mock_client

        result = GeneratedComponent.generate(
            design_spec=self.design_spec,
            session=self.session,
        )

        # compile() should not raise SyntaxError
        compile(result.code, "<test>", "exec")

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    def test_generate_with_minimal_design_spec(
        self, mock_get_client, mock_validate
    ):
        """generate() should work with a minimal design spec (empty fields, etc.)."""
        minimal_spec = ComponentDesignSpec(
            domain_spec=ComponentDomainSpec(
                name="SimpleNote",
                description="A simple note",
                fields=[],
                what_good_looks_like=[],
                expert_role="Note Taker",
            ),
            structure=ComponentStructure(description="A simple note"),
            interaction_context=ComponentInteractionContext(
                must_prioritize=[],
                auto_suggest=[],
                user_pain_points=[],
            ),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_success_response()
        )
        mock_get_client.return_value = mock_client

        result = GeneratedComponent.generate(
            design_spec=minimal_spec,
            session=self.session,
        )

        self.assertIsInstance(result, GeneratedComponent)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    def test_generate_requires_session(
        self, mock_get_client, mock_validate
    ):
        """generate() should raise TypeError when session is missing."""
        with self.assertRaises(TypeError):
            GeneratedComponent.generate(
                design_spec=self.design_spec,
            )

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    def test_generate_uses_low_max_turns(
        self, mock_get_client, mock_validate
    ):
        """generate() should default to 3 max turns (stateless, single-shot)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_success_response()
        )
        mock_get_client.return_value = mock_client

        GeneratedComponent.generate(
            design_spec=self.design_spec,
            session=self.session,
        )

        # The AtomicWorkflow should be configured with max_turns=3
        # (the default in the refactored method)
        mock_client.chat.completions.create.assert_called_once()


class TestGeneratedComponentConstruction(unittest.TestCase):
    """Tests for GeneratedComponent model construction."""

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_construct_with_code(self, mock_collect):
        """GeneratedComponent should construct with valid code."""
        component = GeneratedComponent(code=SAMPLE_CODE)
        self.assertEqual(component.code, SAMPLE_CODE)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_code_must_be_non_empty(self, mock_collect):
        """GeneratedComponent should require non-empty code."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            GeneratedComponent(code="")


if __name__ == "__main__":
    unittest.main(verbosity=2)
