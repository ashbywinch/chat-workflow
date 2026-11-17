#!/usr/bin/env python3
"""Tests for InstructorRetryException handling and LLM self-correction via model_validator."""

import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from chat_workflow.atomic_workflow_step import AtomicWorkflowStep
from chat_workflow.exceptions import MaxRetriesExceededError
from chat_workflow.models import AgentIntent, AgentResponse
from tests.conftest import make_atomic_workflow_step_config, make_valid_criteria
from workflows.evaluation_criteria import EvaluationCriteria


class TestAgentResponseModelValidator(unittest.TestCase):
    """Tests for the model_validator(mode='before') on AgentResponse."""

    def test_bare_list_raises_validation_error(self):
        """Bare list input raises ValidationError with actionable message."""
        with self.assertRaises(ValidationError) as ctx:
            AgentResponse[EvaluationCriteria].model_validate(["item1", "item2"])
        error_msg = str(ctx.exception)
        self.assertIn("bare list", error_msg.lower())

    def test_inner_fields_without_wrapper_raises_validation_error(self):
        """Dict with inner/domain fields but no intent/result raises ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            AgentResponse[EvaluationCriteria].model_validate(
                {"consumer": "test consumer", "format": "json", "source": "test"}
            )
        error_msg = str(ctx.exception)
        self.assertIn("AgentResponse", error_msg)
        self.assertIn("consumer", error_msg)

    def test_valid_input_passes_through_validator(self):
        """Valid AgentResponse input passes through the before-validator successfully."""
        response = AgentResponse[EvaluationCriteria].model_validate(
            {"intent": "success", "result": make_valid_criteria().model_dump()}
        )
        self.assertEqual(response.intent, AgentIntent.SUCCESS)
        self.assertIsNotNone(response.result)

    def test_after_validator_still_works(self):
        """The existing mode='after' validator still enforces intent consistency."""
        with self.assertRaises(ValidationError) as ctx:
            AgentResponse[EvaluationCriteria].model_validate(
                {"intent": "success", "result": None}
            )
        self.assertIn("field named 'result'", str(ctx.exception))


class TestCallLlmInstructorRetryHandling(unittest.TestCase):
    """Tests for InstructorRetryException -> MaxRetriesExceededError in _call_llm."""

    @patch("chat_workflow.llm_interaction.get_client")
    def test_instructor_retry_converted_to_max_retries(self, mock_get_client):
        """InstructorRetryException in _call_llm raises MaxRetriesExceededError."""
        from instructor.core.exceptions import InstructorRetryException

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = InstructorRetryException(
            "test error", n_attempts=3, total_usage=0
        )
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(config=make_atomic_workflow_step_config(max_retries=3))

        with self.assertRaises(MaxRetriesExceededError) as ctx:
            orchestrator._call_llm()

        self.assertIn("3", str(ctx.exception))
        self.assertIn("Maximum retries", str(ctx.exception))

    @patch("chat_workflow.llm_interaction.get_client")
    def test_generic_exceptions_still_propagate(self, mock_get_client):
        """Generic exceptions still propagate through _call_llm unchanged."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Unexpected error")
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(config=make_atomic_workflow_step_config())

        with self.assertRaises(RuntimeError):
            orchestrator._call_llm()

    @patch("chat_workflow.llm_interaction.get_client")
    def test_validation_failure_is_fed_back_before_retry(self, mock_get_client):
        """The rejected response and the reason are both appended for the retry."""
        from types import SimpleNamespace

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow import ValidationError

        rejected_content = '{"intent": "success", "result": {"code": "nested if"}}'
        rejected = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            last_completion=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=rejected_content)
                    )
                ]
            ),
            failed_attempts=[FailedAttempt(1, ValidationError("meeting_date needs min_length"))],
        )
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            rejected,
            AgentResponse[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                result=make_valid_criteria(),
            ),
        ]
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1, validation_corrections=1)
        )

        result = orchestrator._call_llm()

        self.assertEqual(result.intent, AgentIntent.SUCCESS)
        self.assertEqual(orchestrator.messages[-2]["role"], "assistant")
        self.assertEqual(orchestrator.messages[-2]["content"], rejected_content)
        correction = orchestrator.messages[-1]["content"]
        self.assertIn("meeting_date needs min_length", correction)
        self.assertIn("[framework]", correction)

    @patch("chat_workflow.llm_interaction.get_client")
    def test_correction_budget_is_bounded(self, mock_get_client):
        """Exhausting the correction budget raises MaxRetriesExceededError."""
        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow import ValidationError

        def rejected():
            return InstructorRetryException(
                "rejected",
                n_attempts=1,
                total_usage=0,
                failed_attempts=[FailedAttempt(1, ValidationError("still wrong"))],
            )

        def always_reject(**_):
            raise rejected()

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = always_reject
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1, validation_corrections=1)
        )

        with self.assertRaises(MaxRetriesExceededError) as ctx:
            orchestrator._call_llm()

        self.assertIn("still wrong", str(ctx.exception))
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("chat_workflow.llm_interaction.get_client")
    def test_mangled_newline_escape_is_repaired_without_a_retry(self, mock_get_client):
        """A `\\1` where the model wrote `\\n` is repaired, not rejected."""
        from json import JSONDecodeError
        from types import SimpleNamespace

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt
        from pydantic import BaseModel

        class Note(BaseModel):
            text: str

        # The provider returns `\1` in place of `\n` inside a long string.
        content = '{"intent": "success", "result": {"text": "line one\\1line two"}}'
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        rejected = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            last_completion=completion,
            failed_attempts=[
                FailedAttempt(
                    1,
                    JSONDecodeError("Invalid \\escape", content, 1),
                    completion,
                )
            ],
        )
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = rejected
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(
                max_retries=1, validation_corrections=1
            )
        )
        # Point the step at the tiny model under test.
        orchestrator.response_model = AgentResponse[Note]

        result = orchestrator._call_llm()

        assert result.result is not None
        self.assertEqual(result.result.text, "line one\nline two")
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        # No correction was fed back: the repaired response was accepted.
        self.assertEqual([m["role"] for m in orchestrator.messages], ["system"])
        kinds = [event.kind for event in orchestrator.events]
        self.assertIn("repaired_escape", kinds)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    @patch("chat_workflow.atomic_workflow_step.time.sleep", return_value=None)
    def test_a_transient_provider_error_resends_the_same_request(
        self, mock_sleep, mock_get_client, mock_collect
    ):
        """A 500 is the provider's problem; the turn must not advance."""
        from chat_workflow.atomic_workflow_step import _is_transient_provider_error

        class InternalServerError(Exception):
            status_code = 500

        given = []

        def sometimes_fail(**kwargs):
            given.append(list(kwargs["messages"]))
            if len(given) == 1:
                raise InternalServerError("Error code: 500 - upstream failed")
            return AgentResponse[EvaluationCriteria](
                intent=AgentIntent.SUCCESS, result=make_valid_criteria()
            )

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = sometimes_fail
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1, provider_retries=3)
        )

        result = orchestrator._call_llm()

        self.assertEqual(result.intent, AgentIntent.SUCCESS)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertEqual(given[0], given[1], "the same messages are resent")
        self.assertTrue(_is_transient_provider_error(InternalServerError("x")))
        kinds = [event.kind for event in orchestrator.events]
        self.assertIn("provider_retry", kinds)

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    @patch("chat_workflow.atomic_workflow_step.time.sleep", return_value=None)
    def test_transient_retries_are_bounded(
        self, mock_sleep, mock_get_client, mock_collect
    ):
        class ServiceUnavailableError(Exception):
            status_code = 503

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = ServiceUnavailableError("down")
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1, provider_retries=2)
        )

        with self.assertRaises(ServiceUnavailableError):
            orchestrator._call_llm()

        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertEqual(orchestrator.messages, [{"role": "system", "content": "Test prompt"}])

    def test_a_response_from_another_api_is_retried_not_corrected(self):
        """A chat-request object is not an answer; resending can reach a healthy model."""
        from types import SimpleNamespace

        from instructor.core.exceptions import InstructorRetryException

        from chat_workflow.atomic_workflow_step import _returned_no_envelope

        def rejected(content: str) -> InstructorRetryException:
            return InstructorRetryException(
                "rejected",
                n_attempts=1,
                total_usage=0,
                last_completion=SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=content))
                    ]
                ),
            )

        # The provider echoed a chat request at us.
        self.assertTrue(
            _returned_no_envelope(
                rejected('{"model": "gpt-4o-mini", "messages": []}'),
                frozenset({"name", "description"}),
            )
        )
        # The result's own fields without the envelope is a correctable mistake.
        self.assertFalse(
            _returned_no_envelope(
                rejected('{"name": "Meeting Minutes"}'),
                frozenset({"name", "description"}),
            )
        )
        # A real envelope, even a wrong intent, is content.
        self.assertFalse(
            _returned_no_envelope(
                rejected('{"intent": "continue", "message": "hi"}'),
                frozenset({"name"}),
            )
        )
        self.assertFalse(
            _returned_no_envelope(rejected("plain text"), frozenset({"name"}))
        )

    def test_a_model_rule_failure_names_the_rule_and_drops_the_noise(self):
        """Pydantic's str() buries the rule; the model needs the rule itself."""
        from pydantic import BaseModel, Field, ValidationError, model_validator

        from chat_workflow.atomic_workflow_step import _describe_model_errors

        class MinutesDraft(BaseModel):
            decisions: list[str] = Field(...)

            @model_validator(mode="after")
            def _check(self):
                if not self.decisions:
                    raise ValueError("Every decision needs a stated rationale")
                return self

        with self.assertRaises(ValidationError) as ctx:
            MinutesDraft(decisions=[])

        described = _describe_model_errors(ctx.exception)

        self.assertIn("MinutesDraft rejected this result", described)
        self.assertIn("Every decision needs a stated rationale", described)
        self.assertNotIn("input_value", described)
        self.assertNotIn("errors.pydantic.dev", described)
        self.assertNotIn("Value error,", described)

    def test_a_field_rule_failure_names_the_field(self):
        from pydantic import BaseModel, Field, ValidationError, field_validator

        from chat_workflow.atomic_workflow_step import _describe_model_errors

        class Draft(BaseModel):
            meeting_date: str = Field(...)

            @field_validator("meeting_date")
            @classmethod
            def _check(cls, value: str) -> str:
                raise ValueError("meeting_date must be an ISO date")

        with self.assertRaises(ValidationError) as ctx:
            Draft(meeting_date="x")

        described = _describe_model_errors(ctx.exception)

        self.assertIn("- meeting_date: meeting_date must be an ISO date", described)

    def test_the_envelope_name_is_not_shown_as_the_model(self):
        from pydantic import BaseModel, ValidationError

        from chat_workflow.atomic_workflow_step import _describe_model_errors
        from chat_workflow.models import AgentResponse

        class Inner(BaseModel):
            value: int

        with self.assertRaises(ValidationError) as ctx:
            AgentResponse[Inner].model_validate({"intent": "success", "result": {}})

        described = _describe_model_errors(ctx.exception)

        self.assertIn("Inner rejected this result", described)
        self.assertNotIn("AgentResponse", described)

    def test_a_validation_error_is_not_transient(self):
        from chat_workflow.atomic_workflow_step import _is_transient_provider_error
        from chat_workflow.exceptions import MaxRetriesExceededError, ValidationError

        self.assertFalse(_is_transient_provider_error(ValidationError("bad field")))
        self.assertFalse(_is_transient_provider_error(MaxRetriesExceededError(3, "nope")))
        self.assertFalse(_is_transient_provider_error(RuntimeError("plain failure")))

    def test_a_wrapped_provider_error_is_still_transient(self):
        from chat_workflow.atomic_workflow_step import _is_transient_provider_error

        class BadGatewayError(Exception):
            status_code = 502

        try:
            try:
                raise BadGatewayError("bad gateway")
            except BadGatewayError as inner:
                raise RuntimeError("client gave up") from inner
        except RuntimeError as wrapped:
            self.assertTrue(_is_transient_provider_error(wrapped))

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    def test_an_opening_turn_gets_a_user_message(self, mock_get_client, mock_collect):
        """Some providers answer a system-only request with an empty completion."""
        given = []

        def capture(**kwargs):
            given.append(list(kwargs["messages"]))
            return AgentResponse[EvaluationCriteria](
                intent=AgentIntent.SUCCESS, result=make_valid_criteria()
            )

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = capture
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1)
        )
        orchestrator._call_llm()

        roles = [message["role"] for message in given[0]]
        self.assertEqual(roles[-1], "user")
        self.assertIn("Start the conversation", given[0][-1]["content"])
        # The history the model sees next turn is unchanged.
        self.assertNotIn("Start the conversation", [m["content"] for m in orchestrator.messages])

    @patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
    @patch("chat_workflow.llm_interaction.get_client")
    @patch("chat_workflow.atomic_workflow_step.time.sleep", return_value=None)
    def test_a_completion_with_no_content_is_replaced(
        self, mock_sleep, mock_get_client, mock_collect
    ):
        """Some upstreams spend the response on reasoning and emit nothing."""
        from types import SimpleNamespace

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow.exceptions import ValidationError

        empty = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
        )
        rejected = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            last_completion=empty,
            failed_attempts=[FailedAttempt(1, ValidationError("empty"), empty)],
        )
        given = []

        def respond(**kwargs):
            given.append(1)
            if len(given) == 1:
                raise rejected
            return AgentResponse[EvaluationCriteria](
                intent=AgentIntent.SUCCESS, result=make_valid_criteria()
            )

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = respond
        mock_get_client.return_value = mock_client

        orchestrator = AtomicWorkflowStep(
            config=make_atomic_workflow_step_config(max_retries=1, provider_retries=3)
        )

        result = orchestrator._call_llm()

        self.assertEqual(result.intent, AgentIntent.SUCCESS)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        kinds = [event.kind for event in orchestrator.events]
        self.assertIn("provider_retry", kinds)

    def test_a_truncated_response_is_resent_unchanged(self):
        """Nothing to correct when the provider stopped mid-response."""
        from chat_workflow.atomic_workflow_step import _is_transient_provider_error

        error = RuntimeError(
            "The output is incomplete due to a max_tokens length limit."
        )

        self.assertTrue(_is_transient_provider_error(error))

    def test_repair_keeps_a_regex_backslash(self):
        """A stray backslash before a letter is a literal, not a newline."""
        from chat_workflow.atomic_workflow_step import _repair_json_escapes

        repaired = _repair_json_escapes('{"code": "re.sub(r\'\\\\d+\', ...)"}')

        self.assertIn(r"\\d", repaired)

    def test_repair_keeps_the_indentation_after_a_mangled_newline(self):
        """`\\n` that arrives as a bare backslash keeps the line's indentation."""
        from chat_workflow.atomic_workflow_step import _repair_json_escapes

        repaired = _repair_json_escapes('{"code": "class A:\\    x: int"}')

        self.assertIn("\\n    x: int", repaired)

    def test_repair_leaves_valid_json_alone(self):
        """Without a mangled escape the normal correction path still runs."""
        from types import SimpleNamespace

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow.atomic_workflow_step import _recover_from_mangled_escape

        content = '{"intent": "success", "result": {"text": "ok"}}'
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        exc = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            last_completion=completion,
            failed_attempts=[
                FailedAttempt(1, RuntimeError("some other failure"), completion)
            ],
        )

        response, failure = _recover_from_mangled_escape(
            exc, AgentResponse[EvaluationCriteria]
        )
        self.assertIsNone(response)
        self.assertIsNone(failure)

    def test_repair_gives_up_when_the_result_still_does_not_validate(self):
        """A repaired response that still fails validation reports the fields."""
        from types import SimpleNamespace

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow.atomic_workflow_step import _recover_from_mangled_escape

        content = '{"intent": "success", "result": {"text": "line one\\1line two"}}'
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        exc = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            last_completion=completion,
            failed_attempts=[
                FailedAttempt(1, RuntimeError("some other failure"), completion)
            ],
        )

        # EvaluationCriteria has no `text` field, so the repaired JSON is still
        # not a valid response: the field error is what the model must see.
        response, failure = _recover_from_mangled_escape(
            exc, AgentResponse[EvaluationCriteria]
        )
        self.assertIsNone(response)
        assert failure is not None
        # The model's own rule, not the unreadable JSON-escape complaint.
        self.assertIn("budget", failure.reason)
        self.assertNotIn("escape", failure.reason)

    def test_invalid_escape_reason_has_remediation(self):
        from json import JSONDecodeError

        from instructor.core.exceptions import InstructorRetryException
        from instructor.core.retry import FailedAttempt

        from chat_workflow.atomic_workflow_step import _validation_failure

        exc = InstructorRetryException(
            "rejected",
            n_attempts=1,
            total_usage=0,
            failed_attempts=[
                FailedAttempt(
                    1,
                    JSONDecodeError(
                        "Invalid JSON: invalid escape at line 5",
                        '{"x": "\\ "}',
                        5,
                    ),
                )
            ],
        )

        failure = _validation_failure(exc)
        assert failure is not None
        reason = failure.message()

        self.assertIn("Use `\\n`", reason)
        self.assertIn("backslash", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
