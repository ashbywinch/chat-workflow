"""Tests for the create-workflow composite workflow."""

import unittest
from unittest.mock import Mock, patch

from chat_workflow import Session, SessionLog
from chat_workflow.config import Config
from workflows.workflow.flows import (
    _sanitize_name,
    _to_class_name,
    create,
)


class TestSanitizeName(unittest.TestCase):
    """Test the _sanitize_name helper function."""

    def test_basic_name(self):
        self.assertEqual(_sanitize_name("my_workflow"), "my_workflow")

    def test_spaces_to_underscores(self):
        self.assertEqual(_sanitize_name("my workflow"), "my_workflow")

    def test_removes_special_chars(self):
        self.assertEqual(_sanitize_name("my-workflow!"), "myworkflow")

    def test_starts_with_number(self):
        self.assertEqual(_sanitize_name("123workflow"), "workflow_123workflow")

    def test_empty_string(self):
        self.assertEqual(_sanitize_name(""), "")

    def test_uppercase_to_lowercase(self):
        self.assertEqual(_sanitize_name("MyWorkflow"), "myworkflow")


class TestToClassName(unittest.TestCase):
    """Test the _to_class_name helper function."""

    def test_snake_to_pascal(self):
        self.assertEqual(_to_class_name("my_workflow"), "MyWorkflow")

    def test_single_word(self):
        self.assertEqual(_to_class_name("workflow"), "Workflow")

    def test_multiple_underscores(self):
        self.assertEqual(_to_class_name("my_test_workflow"), "MyTestWorkflow")


class TestCreateWorkflow(unittest.TestCase):
    """Test the create composite workflow."""

    def _make_mock_session(self, prompts: list[str]) -> Session:
        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(side_effect=prompts)
        return Session(
            io=mock_io,
            state=SessionLog(),
            config=Mock(spec=Config),
        )

    def test_create_is_workflow(self):
        self.assertTrue(getattr(create, "_is_workflow", False))

    def test_create_requires_session(self):
        with self.assertRaises(TypeError) as ctx:
            create()
        self.assertIn("session", str(ctx.exception))

    @patch("workflows.workflow.flows.Path")
    @patch("workflows.workflow.flows.verify_code")
    @patch("workflows.workflow.flows.generate_class")
    @patch("workflows.workflow.flows.import_module")
    @patch("workflows.workflow.flows.reload_module")
    def test_creates_workflow_with_single_field(
        self, mock_reload, mock_import, mock_generate, mock_verify, mock_path
    ):
        mock_generate.return_value = "class TestWorkflow:\n    pass\n"
        mock_verify.return_value = "class TestWorkflow:\n    pass\n"
        mock_import.return_value = Mock()
        mock_workflow_dir = Mock()
        mock_workflow_dir.mkdir = Mock()
        mock_workflow_dir.__truediv__ = Mock(return_value=mock_workflow_dir)
        mock_workflow_dir.write_text = Mock()
        mock_path.return_value.resolve.return_value.parent.parent = mock_workflow_dir

        prompts = [
            "test workflow",
            "name",
            "str",
            "The name",
            "",
            "",
            "finish",
        ]
        session = self._make_mock_session(prompts)

        create(session=session)

        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        self.assertEqual(call_args.kwargs["name"], "TestWorkflow")
        self.assertEqual(len(call_args.kwargs["fields"]), 1)
        self.assertEqual(call_args.kwargs["fields"][0]["name"], "name")

    @patch("workflows.workflow.flows.Path")
    @patch("workflows.workflow.flows.verify_code")
    @patch("workflows.workflow.flows.generate_class")
    @patch("workflows.workflow.flows.import_module")
    @patch("workflows.workflow.flows.reload_module")
    def test_creates_workflow_with_multiple_fields(
        self, mock_reload, mock_import, mock_generate, mock_verify, mock_path
    ):
        mock_generate.return_value = "class TestWorkflow:\n    pass\n"
        mock_verify.return_value = "class TestWorkflow:\n    pass\n"
        mock_import.return_value = Mock()
        mock_workflow_dir = Mock()
        mock_workflow_dir.mkdir = Mock()
        mock_workflow_dir.__truediv__ = Mock(return_value=mock_workflow_dir)
        mock_workflow_dir.write_text = Mock()
        mock_path.return_value.resolve.return_value.parent.parent = mock_workflow_dir

        prompts = [
            "test workflow",
            "name",
            "str",
            "The name",
            "count",
            "int",
            "The count",
            "",
            "",
            "finish",
        ]
        session = self._make_mock_session(prompts)

        create(session=session)

        call_args = mock_generate.call_args
        self.assertEqual(len(call_args.kwargs["fields"]), 2)

    @patch("workflows.workflow.flows.verify_code")
    @patch("workflows.workflow.flows.generate_class")
    def test_handles_verification_failure(self, mock_generate, mock_verify):
        mock_generate.return_value = "bad code"
        mock_verify.side_effect = RuntimeError("Code failed verification")

        prompts = [
            "test workflow",
            "name",
            "str",
            "The name",
            "",
            "",
            "n",
        ]
        session = self._make_mock_session(prompts)

        create(session=session)

        mock_verify.assert_called()

    @patch("workflows.workflow.flows.Path")
    @patch("workflows.workflow.flows.verify_code")
    @patch("workflows.workflow.flows.generate_class")
    @patch("workflows.workflow.flows.import_module")
    @patch("workflows.workflow.flows.reload_module")
    def test_prompts_for_another_workflow(
        self, mock_reload, mock_import, mock_generate, mock_verify, mock_path
    ):
        mock_generate.return_value = "class TestWorkflow:\n    pass\n"
        mock_verify.return_value = "class TestWorkflow:\n    pass\n"
        mock_import.return_value = Mock()
        mock_workflow_dir = Mock()
        mock_workflow_dir.mkdir = Mock()
        mock_workflow_dir.__truediv__ = Mock(return_value=mock_workflow_dir)
        mock_workflow_dir.write_text = Mock()
        mock_path.return_value.resolve.return_value.parent.parent = mock_workflow_dir

        prompts = [
            "first workflow",
            "name",
            "str",
            "The name",
            "",
            "",
            "create another",
            "second workflow",
            "title",
            "str",
            "The title",
            "",
            "",
            "finish",
        ]
        session = self._make_mock_session(prompts)

        create(session=session)

        self.assertEqual(mock_generate.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
