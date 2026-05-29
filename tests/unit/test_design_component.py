"""Tests for Component.design_component atomic workflow."""
import unittest
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from chat_workflow.atomic_workflow import AtomicWorkflow
from chat_workflow.models import AgentIntent, AgentResponse
from workflows.workflow.component import Component
from workflows.workflow.models import ComponentRequirement, GeneratedComponent


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


class TestDesignComponent(unittest.TestCase):
    def _make_session(self) -> Session:
        return Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    def test_has_workflow_attribute(self):
        self.assertTrue(
            getattr(Component.design_component, "_is_workflow", False),
        )

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            Component.design_component(
                requirements=ComponentRequirement(
                    name="Test",
                    purpose="Test component",
                    required_inputs=["Input"],
                    expected_outputs=["Output"],
                    component_type="artifact_producing",
                ),
            )
        self.assertIn("session", str(ctx.exception))

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_returns_generated_component(self, mock_call_llm):
        expected_code = (
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, Field\n\n"
            "class Order(BaseModel):\n"
            '    name: str = Field(..., description="Order name")\n'
        )
        expected = GeneratedComponent(code=expected_code)
        mock_call_llm.return_value = AgentResponse[GeneratedComponent](
            intent=AgentIntent.SUCCESS,
            result=expected,
        )

        session = self._make_session()
        result = Component.design_component(
            requirements=ComponentRequirement(
                name="Order",
                purpose="Manage orders",
                required_inputs=["Customer details"],
                expected_outputs=["Order confirmation"],
                component_type="artifact_producing",
            ),
            session=session,
        )

        self.assertIsInstance(result, GeneratedComponent)
        self.assertIn("class Order(BaseModel)", result.code)
        self.assertIn("BaseModel", result.code)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_code_contains_expected_imports(self, mock_call_llm):
        expected_code = (
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, Field\n\n"
            "class TestComponent(BaseModel):\n"
            '    name: str = Field(..., description="Test name")\n'
        )
        mock_call_llm.return_value = AgentResponse[GeneratedComponent](
            intent=AgentIntent.SUCCESS,
            result=GeneratedComponent(code=expected_code),
        )

        session = self._make_session()
        result = Component.design_component(
            requirements=ComponentRequirement(
                name="TestComponent",
                purpose="Test",
                required_inputs=[],
                expected_outputs=[],
                component_type="artifact_producing",
            ),
            session=session,
        )

        self.assertIn("from __future__ import annotations", result.code)
        self.assertIn("from pydantic import BaseModel", result.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)