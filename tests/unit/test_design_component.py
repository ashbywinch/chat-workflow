"""Tests for ComponentSourceCode.generate atomic workflow step."""

import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import AgentIntent, AgentResponse, Session, SessionLog
from chat_workflow.atomic_workflow_step import AtomicWorkflowStep
from workflows.workflow.component_source_code import ComponentSourceCode
from workflows.workflow.models import ComponentSpec


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False
    model_supports_tools = False
    api_base = None
    api_key_env = None


class TestDesignComponent(unittest.TestCase):
    def _make_session(self) -> Session:
        return Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    def test_has_workflow_attribute(self):
        self.assertTrue(
            getattr(ComponentSourceCode.generate, "_is_workflow_step", False),
        )

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            ComponentSourceCode.generate(
                requirements=ComponentSpec(
                    name="Test",
                    purpose="Test component",
                    required_inputs=["Input"],
                    scope_description="description", 
                    component_type="artifact_producing",
                    owner_role="Owner",
                ),
            )
        self.assertIn("session", str(ctx.exception))

    @patch("chat_workflow.llm_interaction.get_client")
    @patch.object(AtomicWorkflowStep, "_call_llm")
    @patch("chat_workflow.mixins.LLMValidated.validate_llm_rules", return_value=None)
    def test_returns_generated_component(self, mock_validate, mock_call_llm, mock_get_client):
        mock_get_client.side_effect = RuntimeError("no api key")
        expected_code = (
            "from __future__ import annotations\n"
            "\n"
            "from pydantic import BaseModel, Field\n"
            "\n"
            "from chat_workflow import atomic_workflow_step\n\n\n"
            "class Order(BaseModel):\n"
            '    name: str = Field(..., min_length=1, description="Order name")\n'
            "\n"
            "    @atomic_workflow_step\n"
            "    @classmethod\n"
            "    def create(cls, context: str, max_turns: int = 10) -> Order:\n"
            '        """Create order."""\n'
            "        ...\n"
        )
        expected = ComponentSourceCode.model_construct(code=expected_code)
        mock_call_llm.return_value = AgentResponse[ComponentSourceCode].model_construct(
            intent=AgentIntent.SUCCESS,
            result=expected,
        )

        session = self._make_session()
        result = ComponentSourceCode.generate(
            requirements=ComponentSpec(
                name="Order",
                purpose="Manage orders",
                required_inputs=["Customer details"],
                scope_description="description", 
                component_type="artifact_producing",
                owner_role="Owner",
            ),
            session=session,
        )

        self.assertIsInstance(result, ComponentSourceCode)
        self.assertIn("class Order(BaseModel)", result.code)
        self.assertIn("BaseModel", result.code)

    @patch("chat_workflow.llm_interaction.get_client")
    @patch.object(AtomicWorkflowStep, "_call_llm")
    @patch("chat_workflow.mixins.LLMValidated.validate_llm_rules", return_value=None)
    def test_code_contains_expected_imports(self, mock_validate, mock_call_llm, mock_get_client):
        mock_get_client.side_effect = RuntimeError("no api key")
        expected_code = (
            "from __future__ import annotations\n"
            "\n"
            "from pydantic import BaseModel, Field\n"
            "\n"
            "from chat_workflow import atomic_workflow_step\n\n\n"
            "class TestComponent(BaseModel):\n"
            '    name: str = Field(..., min_length=1, description="Test name")\n'
            "\n"
            "    @atomic_workflow_step\n"
            "    @classmethod\n"
            "    def create(cls, context: str, max_turns: int = 10) -> TestComponent:\n"
            '        """Create test component."""\n'
            "        ...\n"
        )
        mock_call_llm.return_value = AgentResponse[ComponentSourceCode].model_construct(
            intent=AgentIntent.SUCCESS,
            result=ComponentSourceCode.model_construct(code=expected_code),
        )

        session = self._make_session()
        result = ComponentSourceCode.generate(
            requirements=ComponentSpec(
                name="TestComponent",
                purpose="Test",
                required_inputs=[],
                scope_description="description",
                component_type="artifact_producing",
                owner_role="Owner",
            ),
            session=session,
        )

        self.assertIn("from __future__ import annotations", result.code)
        self.assertIn("from pydantic import BaseModel", result.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
