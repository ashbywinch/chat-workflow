"""Tests for Component.create composite workflow."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from chat_workflow import Session, SessionLog
from workflows.workflow import GeneratedComponent
from workflows.workflow.component import Component
from workflows.workflow.models import ComponentRequirement


class FakeConfig:
    model = "test-model"
    provider = "test-provider"
    max_retries = 3
    request_timeout_seconds = 30
    debug = False


class TestComponentCreate(unittest.TestCase):
    def _make_session(self) -> Session:
        return Session(
            io=MagicMock(),
            state=SessionLog(),
            config=FakeConfig(),
        )

    def test_has_workflow_attribute(self):
        self.assertTrue(getattr(Component.create, "_is_workflow", False))

    def test_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            Component.create(
                requirements=ComponentRequirement(
                    name="Test",
                    purpose="Test",
                    required_inputs=[],
                    expected_outputs=[],
                    component_type="artifact_producing",
                ),
            )
        self.assertIn("session", str(ctx.exception))

    @patch.object(GeneratedComponent, "generate")
    @patch("workflows.workflow.component.verify_code")
    def test_writes_file_and_returns_component(self, mock_verify, mock_design):
        """Component.create should write code to disk and return Component."""
        valid_code = (
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, Field\n"
            "from chat_workflow import atomic_workflow\n\n"
            "class Order(BaseModel):\n"
            '    name: str = Field(..., min_length=1, description="Order name")\n'
            "\n"
            "    @atomic_workflow\n"
            "    @classmethod\n"
            "    def create(cls, context: str, max_turns: int = 10):\n"
            '        """Create order."""\n'
            "        ...\n"
        )
        mock_design.return_value = GeneratedComponent.model_construct(code=valid_code)
        mock_verify.return_value = valid_code

        with TemporaryDirectory() as tmpdir:
            session = self._make_session()
            result = Component.create(
                requirements=ComponentRequirement(
                    name="Order",
                    purpose="Manage orders",
                    required_inputs=["Details"],
                    expected_outputs=["Confirmation"],
                    component_type="artifact_producing",
                ),
                session=session,
                output_dir=Path(tmpdir),
            )

            self.assertIsInstance(result, Component)
            self.assertEqual(result.name, "Order")
            self.assertTrue(result.code_path.exists())
            self.assertIn("order", result.code_path.name)

    @patch.object(GeneratedComponent, "generate")
    @patch("workflows.workflow.component.verify_code")
    def test_verify_code_failure_raises_error(self, mock_verify, mock_design):
        """When verify_code fails, error should propagate."""
        valid_code = (
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, Field\n"
            "from chat_workflow import atomic_workflow\n\n"
            "class Bad(BaseModel):\n"
            '    name: str = Field(..., min_length=1, description="Name")\n'
            "\n"
            "    @atomic_workflow\n"
            "    @classmethod\n"
            "    def create(cls, context: str, max_turns: int = 10):\n"
            '        """Create."""\n'
            "        ...\n"
        )
        mock_design.return_value = GeneratedComponent.model_construct(code=valid_code)
        mock_verify.side_effect = RuntimeError("Code quality check failed")

        session = self._make_session()
        with self.assertRaises(RuntimeError):
            Component.create(
                requirements=ComponentRequirement(
                    name="Bad",
                    purpose="Bad component",
                    required_inputs=[],
                    expected_outputs=[],
                    component_type="artifact_producing",
                ),
                session=session,
                output_dir=Path("/tmp"),
            )

    @patch.object(GeneratedComponent, "generate")
    @patch("workflows.workflow.component.verify_code")
    def test_default_output_dir(self, mock_verify, mock_design):
        """When no output_dir given, defaults to cwd/workflows/{name}/."""
        valid_code = (
            "from __future__ import annotations\n"
            "from pydantic import BaseModel, Field\n"
            "from chat_workflow import atomic_workflow\n\n"
            "class X(BaseModel):\n"
            '    name: str = Field(..., min_length=1, description="Name")\n'
            "\n"
            "    @atomic_workflow\n"
            "    @classmethod\n"
            "    def create(cls, context: str, max_turns: int = 10):\n"
            '        """Create."""\n'
            "        ...\n"
        )
        mock_design.return_value = GeneratedComponent.model_construct(code=valid_code)
        mock_verify.return_value = valid_code

        session = self._make_session()
        with patch.object(Path, "cwd", return_value=Path("/tmp")):
            result = Component.create(
                requirements=ComponentRequirement(
                    name="TestComponent",
                    purpose="Test",
                    required_inputs=[],
                    expected_outputs=[],
                    component_type="artifact_producing",
                ),
                session=session,
            )

            self.assertIsInstance(result, Component)
            self.assertIn("testcomponent", str(result.code_path).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
