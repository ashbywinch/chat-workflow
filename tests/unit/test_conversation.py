"""The one model path: typed output, the transcript, and the retry classes.

Every test scripts the model with a FunctionModel: no network, no provider,
and the exact responses a real gateway has been seen to produce (a blank
completion, a token-limit stop, a wrong envelope, a good answer).
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelResponse,
    RetryFeedbackPart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from chat_workflow import Conversation, FinalResponse, MaxRetriesExceededError, Success
from chat_workflow.models import AgentIntent
from tests.conftest import make_conversation, make_valid_criteria
from workflows.evaluation_criteria import EvaluationCriteria

VALID = {"intent": "success", "result": make_valid_criteria().model_dump()}


def _retry_parts(conversation: Conversation) -> list[Any]:
    """Every correction the model was given.

    The answer is a tool call, so a rejection comes back as a retried
    ``ToolReturnPart``; a business rule and a field error share that channel.
    A text answer's retry is still a ``RetryFeedbackPart``.
    """
    return [
        part
        for message in conversation.messages
        for part in getattr(message, "parts", ())
        if isinstance(part, (RetryFeedbackPart, RetryPromptPart))
        or (isinstance(part, ToolReturnPart) and getattr(part, "outcome", "success") != "success")
    ]


class TestTypedOutput(unittest.TestCase):
    def test_the_output_is_validated_to_the_requested_type(self):
        conversation = make_conversation([VALID])

        turn = conversation.send("hello")

        self.assertIsInstance(turn.output, Success)
        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(turn.output.result, make_valid_criteria())
        self.assertEqual(turn.usage.requests, 1)

    def test_the_history_accumulates_across_turns(self):
        conversation = make_conversation(
            [
                {"intent": "continue", "message": "What is the budget?"},
                VALID,
            ]
        )

        conversation.send("I need criteria")
        second = conversation.send("About fifty pounds")

        self.assertEqual(second.output.intent, AgentIntent.SUCCESS)

        # Each turn is a request, the answer's tool call and its return; the
        # second request carries the first turn's exchange.
        self.assertEqual(len(conversation.messages), 6)
        carried = [type(part).__name__ for message in conversation.script.calls[1] for part in message.parts]
        self.assertIn("ToolCallPart", carried)

    def test_an_empty_prompt_opens_the_conversation(self):
        conversation = make_conversation([VALID])

        conversation.send("")

        sent = conversation.script.calls[0]
        texts = [part.content for message in sent for part in getattr(message, "parts", ())]
        self.assertIn("Start the conversation.", texts)

    def test_an_empty_prompt_uses_the_initial_user_turn(self):
        """A step seeded with its opening context must not add a second prompt."""
        from pydantic_ai.messages import ModelRequest

        conversation = make_conversation(
            [VALID],
            messages=[ModelRequest.user_text_prompt("Context from earlier")],
        )

        conversation.send("")

        sent = conversation.script.calls[0]
        texts = [part.content for message in sent for part in getattr(message, "parts", ())]
        self.assertNotIn("Start the conversation.", texts)
        self.assertIn("Context from earlier", texts)


class TestUnusableResponses(unittest.TestCase):
    """A response with nothing usable is replaced without a prompt.

    The model never learns: no retry part is added, and the transcript holds
    only the answer that worked.
    """

    def test_a_blank_response_is_resent(self):
        conversation = make_conversation([ModelResponse(parts=[]), VALID])

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(len(conversation.script.calls), 2)
        self.assertEqual(len(turn.resends), 1)
        self.assertEqual(_retry_parts(conversation), [])

    def test_a_thinking_only_response_is_resent(self):
        from pydantic_ai.messages import ThinkingPart

        conversation = make_conversation(
            [ModelResponse(parts=[ThinkingPart("I am still weighing this")]), VALID]
        )

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(len(turn.resends), 1)
        self.assertIn("thinking only", turn.resends[0].cause)

    def test_a_token_limit_stop_with_no_output_is_resent(self):
        conversation = make_conversation(
            [ModelResponse(parts=[], finish_reason="length"), VALID]
        )

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(turn.resends[0].finish_reason, "length")

    def test_a_truncated_answer_is_resent_not_corrected(self):
        """A token-limit stop cuts the JSON mid-object. There is no answer to
        correct, so the same request is sampled again — the output retry would
        only ask the model to fix a fragment it never finished."""
        truncated = ModelResponse(
            parts=[TextPart('{"response": {"intent": "success", "result": {"crit',
            )],
            finish_reason="length",
        )
        conversation = make_conversation([truncated, VALID])

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(len(conversation.script.calls), 2)
        self.assertEqual(turn.resends[0].cause, "token limit reached before the answer finished")
        self.assertEqual(_retry_parts(conversation), [])

    def test_a_call_whose_arguments_are_not_json_is_resent(self):
        """An endpoint that treats the tool schema as a hint returns
        `{"response":Got it — a…}`. Nothing there is an answer to correct."""
        malformed = ModelResponse(parts=[ToolCallPart("answer", '{"response":Got it — a…}')])
        conversation = make_conversation([malformed, VALID])

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(turn.resends[0].cause, "tool arguments were not JSON")

    def test_a_stall_of_whitespace_is_resent(self):
        """The gateway's model can emit the opening token and then newlines to
        the token limit (measured: 5,392 lines). Nothing visible is an answer,
        however long the text is."""
        stall = ModelResponse(parts=[TextPart('{"response":' + "\r\n   " * 1200)])
        conversation = make_conversation([stall, VALID])

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(turn.resends[0].cause, "whitespace instead of an answer")
        self.assertEqual(_retry_parts(conversation), [])

    def test_the_resend_is_bounded(self):
        conversation = make_conversation(
            [ModelResponse(parts=[])],
            unusable_resends=1,
            output_retries=0,
        )

        with self.assertRaises(MaxRetriesExceededError):
            conversation.send("hello")

        self.assertEqual(len(conversation.script.calls), 2)
        self.assertEqual(len(conversation.script.calls[0]), 1)

    def test_no_resends_can_be_configured(self):
        conversation = make_conversation(
            [ModelResponse(parts=[]), VALID], unusable_resends=0, output_retries=0
        )

        with self.assertRaises(MaxRetriesExceededError):
            conversation.send("hello")

        self.assertEqual(len(conversation.script.calls), 1)


class TestRejectedOutput(unittest.TestCase):
    """The output channel: the rejection is fed back, visibly."""

    def test_a_rejected_envelope_is_fed_back_and_corrected(self):
        conversation = make_conversation([{"intent": "success", "result": None}, VALID])

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(len(conversation.script.calls), 2)
        retries = _retry_parts(conversation)
        self.assertEqual(len(retries), 1)
        # The second request carried the rejection, so the model saw it.
        self.assertEqual(len(conversation.script.calls[1]), 3)

    def test_a_field_error_names_the_field(self):
        invalid = {"intent": "success", "result": {"context": "x", "criteria": []}}
        conversation = make_conversation([invalid, VALID])

        conversation.send("hello")

        feedback = _retry_parts(conversation)[0]
        rendered = json.dumps(feedback.content) if not isinstance(feedback.content, str) else feedback.content
        self.assertIn("criteria", rendered)

    def test_the_output_budget_is_bounded(self):
        conversation = make_conversation(
            [{"intent": "success", "result": None}], output_retries=1
        )

        with self.assertRaises(MaxRetriesExceededError) as ctx:
            conversation.send("hello")

        self.assertIn("1", str(ctx.exception))
        self.assertEqual(len(conversation.script.calls), 2)

    def test_zero_output_retries_fails_the_first_bad_answer(self):
        """A judgement eval treats the output as content: no correction turn."""
        conversation = make_conversation(
            [{"intent": "success", "result": None}], output_retries=0
        )

        with self.assertRaises(MaxRetriesExceededError):
            conversation.send("hello")

        self.assertEqual(len(conversation.script.calls), 1)
        self.assertEqual(_retry_parts(conversation), [])


class TestBusinessRules(unittest.TestCase):
    """A caller's rule on the answer model rides the one output-retry channel."""

    def test_a_rule_from_the_answer_model_is_told_to_the_model(self):
        """A rule is only retried because ``ValidationError`` is a ``ValueError``.

        Pydantic captures a ``ValueError`` from a validator and hands its
        sentence to the same retry a missing field uses. Without that base the
        exception escapes validation and the workflow crashes on an answer the
        model could have corrected.
        """
        budgetless = {
            "intent": "success",
            "result": {
                "context": "A birthday present",
                "criteria": [
                    {"name": "price", "description": "Keeps the gift affordable"},
                    {"name": "thought", "description": "Shows the person was considered"},
                ],
            },
        }
        conversation = make_conversation([budgetless, VALID])

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(len(conversation.script.calls), 2)
        feedback = _retry_parts(conversation)[0]
        rendered = feedback.content if isinstance(feedback.content, str) else json.dumps(feedback.content)
        self.assertIn("budget", rendered.lower())


class TestInstructions(unittest.TestCase):
    def test_the_instructions_are_sent_once(self):
        """The agent carries the step's instructions; a run adds only the
        final-turn note. Passing both rendered them twice on every request."""
        conversation = make_conversation([VALID])
        conversation.send("hello", final_turn=True)

        parts = [
            part
            for message in conversation.script.calls[0]
            for part in message.parts
            if isinstance(part, SystemPromptPart)
        ]
        self.assertEqual(len(parts), 1)
        self.assertIn("Final turn", parts[0].content)
        self.assertEqual(parts[0].content.count("Test instructions"), 1)


class TestToolFailures(unittest.TestCase):
    def test_a_tool_s_own_failure_is_passed_through_not_rewrapped(self):
        from pydantic_ai import ToolFailed

        def refuses(q: str) -> str:
            """Refuse, saying why."""
            raise ToolFailed("the archive is closed")

        conversation = make_conversation(
            [ModelResponse(parts=[ToolCallPart("refuses", {"q": "x"})]), VALID],
            tools=[refuses],
        )

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        returned = [
            part
            for message in conversation.messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "refuses"
        ][0]
        self.assertEqual(str(returned.content), "the archive is closed")


class TestProviderLog(unittest.TestCase):
    def test_a_failure_names_the_endpoint_that_served_it(self):
        """The endpoints behind one model differ; a failure is diagnosable
        only when the message says who answered."""
        from chat_workflow.conversation import _ProviderLog

        log = _ProviderLog()
        log.provider = "DeepInfra"
        log.model = "deepseek/deepseek-v4-flash"

        self.assertEqual(log.suffix(), " (served by DeepInfra, deepseek/deepseek-v4-flash)")
        self.assertEqual(_ProviderLog().suffix(), "")

    def test_a_model_without_a_client_records_nothing(self):

        conversation = make_conversation([VALID])

        self.assertEqual(conversation._provider_log.suffix(), "")


class TestStepTools(unittest.TestCase):
    def test_a_step_can_call_its_own_tool(self):
        seen: list[str] = []

        def lookup(q: str) -> str:
            """Look a thing up."""
            seen.append(q)
            return "found"

        conversation = make_conversation(
            [ModelResponse(parts=[ToolCallPart("lookup", {"q": "budget"})]), VALID],
            tools=[lookup],
        )

        turn = conversation.send("hello")

        self.assertEqual(seen, ["budget"])
        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        parts = [type(part).__name__ for message in conversation.messages for part in message.parts]
        self.assertIn("ToolCallPart", parts)
        self.assertIn("ToolReturnPart", parts)

    def test_a_failing_tool_is_a_result_not_an_exception(self):
        def broken(q: str) -> str:
            """Break on purpose."""
            raise RuntimeError("down")

        conversation = make_conversation(
            [ModelResponse(parts=[ToolCallPart("broken", {"q": "x"})]), VALID],
            tools=[broken],
        )

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)


class TestTheLastTurn(unittest.TestCase):
    def test_the_last_turn_cannot_ask_a_question(self):
        """No `continue` shape is offered, so the model has nothing to ask
        with: the same answer that was accepted mid-conversation is rejected."""
        conversation = make_conversation(
            [{"intent": "continue", "message": "One more question?"}, VALID],
            final_output_type=FinalResponse[EvaluationCriteria],
        )

        turn = conversation.send("hello", final_turn=True)

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(len(conversation.script.calls), 2)


class TestTheUserView(unittest.TestCase):
    def test_it_shows_the_question_not_the_answer_envelope(self):
        from chat_workflow.transcript import render

        conversation = make_conversation(
            [
                {"intent": "continue", "message": "What is the budget?"},
                VALID,
            ]
        )
        conversation.send("I need criteria")
        conversation.send("About fifty pounds")

        text = render(conversation.messages, audience="user")

        self.assertIn("What is the budget?", text)
        self.assertNotIn('"intent"', text)


class TestGatewayUpstreamFailures(unittest.TestCase):
    """The gateway reports its own upstream failing as a 400, not a 5xx."""

    def test_the_gateways_upstream_failure_is_resent(self):
        error = ModelHTTPError(
            400,
            "dynamic/fallback2",
            body=[{"code": 2005, "message": "Failed to get response from provider"}],
        )
        conversation = make_conversation([error, VALID])

        turn = conversation.send("hello")

        self.assertEqual(turn.output.intent, AgentIntent.SUCCESS)
        self.assertEqual(len(conversation.script.calls), 2)

    def test_a_rejected_request_is_not_resent(self):
        error = ModelHTTPError(400, "dynamic/fallback2", body=[{"code": 4000, "message": "bad schema"}])
        conversation = make_conversation([error, VALID])

        with self.assertRaises(ModelHTTPError):
            conversation.send("hello")

        self.assertEqual(len(conversation.script.calls), 1)


class TestRequestSettings(unittest.TestCase):
    """The budgets and limits reach the request as separate settings."""

    def test_the_channels_are_separate_settings(self):
        conversation = make_conversation(
            [VALID],
            transport_retries=5,
            max_tokens=4096,
            timeout_seconds=90.0,
            temperature=0.0,
            extra_headers={"cf-aig-skip-cache": "true"},
            thinking=True,
        )

        settings = conversation.model_settings()

        self.assertEqual(settings["max_retries"], 5)
        self.assertEqual(settings["max_tokens"], 4096)
        self.assertEqual(settings["timeout"], 90.0)
        self.assertEqual(settings["temperature"], 0.0)
        self.assertEqual(settings["extra_headers"], {"cf-aig-skip-cache": "true"})
        self.assertTrue(settings["thinking"])

    def test_defaults_leave_temperature_to_the_provider(self):
        conversation = make_conversation([VALID])

        settings = conversation.model_settings()

        self.assertNotIn("temperature", settings)
        self.assertEqual(settings["max_retries"], 2)

    def test_the_request_limit_bounds_one_turn(self):
        conversation = make_conversation([VALID], max_requests=1)

        conversation.send("hello")

        self.assertEqual(len(conversation.script.calls), 1)
