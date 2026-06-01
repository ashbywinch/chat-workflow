"""Tests for annotations and mixins."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, ClassVar
from unittest.mock import patch

from pydantic import Field

from chat_workflow.annotations import Blob, Validation
from chat_workflow.exceptions import ValidationError
from chat_workflow.mixins import BlobSyncMixin, LLMValidated, get_blob_fields


class TestBlob(unittest.TestCase):
    def test_default_extension(self):
        b = Blob()
        self.assertEqual(b.extension, ".txt")

    def test_custom_extension(self):
        b = Blob(".mmd")
        self.assertEqual(b.extension, ".mmd")


class TestValidation(unittest.TestCase):
    def test_rule_storage(self):
        v = Validation("Must have at least 3 participants")
        self.assertEqual(v.rule, "Must have at least 3 participants")


class TestGetBlobFields(unittest.TestCase):
    def test_discovers_blob_fields(self):
        class TestModel(BlobSyncMixin):
            name: str = Field(...)
            diagram: Annotated[str, Blob(".mmd")] = Field(...)

        blobs = get_blob_fields(TestModel)
        self.assertIn("diagram", blobs)
        self.assertEqual(blobs["diagram"], ".mmd")
        self.assertNotIn("name", blobs)

    def test_no_blob_fields_returns_empty(self):
        class TestModel(BlobSyncMixin):
            name: str = Field(...)

        blobs = get_blob_fields(TestModel)
        self.assertEqual(blobs, {})


class TestBlobSyncMixin(unittest.TestCase):
    def test_materialize_blobs_writes_files(self):
        class TestModel(BlobSyncMixin):
            diagram: Annotated[str, Blob(".mmd")] = Field(...)
            name: str = Field(...)

        with TemporaryDirectory() as tmpdir:
            model = TestModel(diagram="sequenceDiagram\na->>b: hello", name="test")
            result = model.materialize_blobs(Path(tmpdir))

            expected_path = Path(tmpdir) / "diagram.mmd"
            self.assertTrue(expected_path.exists())
            self.assertEqual(expected_path.read_text(), "sequenceDiagram\na->>b: hello")
            self.assertIs(result, model)

    def test_get_blob_path(self):
        class TestModel(BlobSyncMixin):
            diagram: Annotated[str, Blob(".mmd")] = Field(...)

        with TemporaryDirectory() as tmpdir:
            model = TestModel(diagram="test")
            model.materialize_blobs(Path(tmpdir))

            path = model.get_blob_path("diagram")
            self.assertIsNotNone(path)
            self.assertEqual(path.name, "diagram.mmd")

            self.assertIsNone(model.get_blob_path("nonexistent"))

    def test_no_blob_fields_does_nothing(self):
        class TestModel(BlobSyncMixin):
            name: str = Field(...)

        with TemporaryDirectory() as tmpdir:
            model = TestModel(name="test")
            model.materialize_blobs(Path(tmpdir))
            self.assertEqual(len(model._blob_paths), 0)

    def test_multiple_blob_fields(self):
        class TestModel(BlobSyncMixin):
            diagram: Annotated[str, Blob(".mmd")] = Field(...)
            data: Annotated[str, Blob(".json")] = Field(...)

        with TemporaryDirectory() as tmpdir:
            model = TestModel(diagram="digraph {}", data='{"key": "val"}')
            model.materialize_blobs(Path(tmpdir))

            self.assertTrue((Path(tmpdir) / "diagram.mmd").exists())
            self.assertTrue((Path(tmpdir) / "data.json").exists())
            self.assertEqual((Path(tmpdir) / "data.json").read_text(), '{"key": "val"}')

    def test_creates_output_dir(self):
        class TestModel(BlobSyncMixin):
            diagram: Annotated[str, Blob(".mmd")] = Field(...)

        with TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            model = TestModel(diagram="test")
            model.materialize_blobs(nested)
            self.assertTrue(nested.exists())
            self.assertTrue((nested / "diagram.mmd").exists())


class TestLLMValidated(unittest.TestCase):
    def test_collect_per_field_rules(self):
        class TestModel(LLMValidated):
            name: str = Field(...)
            diagram: Annotated[str, Validation("Must have participants")] = Field(...)

        rules = TestModel.collect_per_field_rules()
        self.assertIn("diagram", rules)
        self.assertIn("Must have participants", rules["diagram"])
        self.assertNotIn("name", rules)

    def test_collect_all_rules_includes_classvar(self):
        class TestModel(LLMValidated):
            name: str = Field(...)
            diagram: Annotated[str, Validation("Must have participants")] = Field(...)
            _validation_rules: ClassVar[list[str]] = ["All components must be named"]

        rules = TestModel.collect_all_rules()
        self.assertIn("[diagram] Must have participants", rules)
        self.assertIn("All components must be named", rules)

    def test_collect_all_rules_empty_when_no_rules(self):
        class TestModel(LLMValidated):
            name: str = Field(...)

        rules = TestModel.collect_all_rules()
        self.assertEqual(rules, [])

    def test_description_injection(self):
        class TestModel(LLMValidated):
            diagram: Annotated[str, Validation("Must have participants")] = Field(..., description="A diagram")

        desc = TestModel.model_fields["diagram"].description
        self.assertIsNotNone(desc)
        self.assertIn("Must have participants", desc)
        self.assertIn("A diagram", desc)

    def test_description_injection_no_existing_description(self):
        class TestModel(LLMValidated):
            diagram: Annotated[str, Validation("Must have participants")] = Field(...)

        desc = TestModel.model_fields["diagram"].description
        self.assertIsNotNone(desc)
        self.assertEqual(desc, "- Must have participants")

    def test_multiple_rules_on_one_field(self):
        class TestModel(LLMValidated):
            name: Annotated[
                str,
                Validation("Must be non-empty"),
                Validation("Must be unique"),
            ] = Field(...)

        rules = TestModel.collect_per_field_rules()
        self.assertIn("name", rules)
        self.assertEqual(rules["name"], ["Must be non-empty", "Must be unique"])

        desc = TestModel.model_fields["name"].description
        self.assertIn("Must be non-empty", desc)
        self.assertIn("Must be unique", desc)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    def test_model_construct_without_api_key(self, mock_collect):
        """Should not raise when no API key is configured (silently skips LLM validation)."""

        class TestModel(LLMValidated):
            name: str = Field(...)
            diagram: Annotated[str, Validation("Must have participants")] = Field(...)

        # No API key set — validate_llm_rules should be silently skipped
        model = TestModel(name="test", diagram="test")
        self.assertEqual(model.name, "test")

    @patch("chat_workflow.llm_interaction.get_client")
    def test_validate_llm_rules_called_when_mocked(self, mock_get_client):
        """LLM validation runs when get_client is mocked to return a passing result."""

        class TestModel(LLMValidated):
            name: str = Field(...)
            diagram: Annotated[str, Validation("Must have participants")] = Field(...)

        model = TestModel(name="test", diagram="A->B")
        self.assertEqual(model.name, "test")
        self.assertEqual(model.diagram, "A->B")

    @patch("chat_workflow.llm_interaction.get_client")
    def test_validation_error_on_violation(self, mock_get_client):
        """Should raise ValidationError when LLM reports violations."""
        mock_instance = unittest.mock.MagicMock()
        # Return a response that has no 'choices' attribute (so we hit
        # the instructor-client branch that checks for raw 'valid' attr).
        mock_response = unittest.mock.MagicMock(spec_set=["valid", "violations"])
        mock_response.valid = False
        mock_response.violations = ["Rule 1 violated", "Rule 2 violated"]
        mock_instance.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_instance

        class TestModel(LLMValidated):
            name: Annotated[str, Validation("Must be at least 3 chars")] = Field(...)

        with self.assertRaises(ValidationError) as ctx:
            TestModel(name="ab")

        self.assertIn("Rule 1 violated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
