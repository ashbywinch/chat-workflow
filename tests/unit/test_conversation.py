"""The one model path: typed output, the transcript, and the retry classes.

Every test scripts the model with a FunctionModel: no network, no provider,
and the exact responses a real gateway has been seen to produce (a blank
completion, a token-limit stop, a wrong envelope, a good answer).
"""

from __future__ import annotations

import json
import unittest

from pydantic_ai.messages import ModelResponse, RetryFeedbackPart

from chat_workflow import AgentResponse, Conversation, MaxRetriesExceededError
from chat_workflow.models import AgentIntent
from tests.conftest import make_conversation, make_valid_criteria

VALID = {"intent": "success", "result": make_valid_criteria().model_dump()}


def _retry_parts(conversation: Conversation) -> list[RetryFeedbackPart]:
    return [
        part
        for message in conversation.messages
        for part in getattr(message, "parts", ())
        if isinstance(part, RetryFeedbackPart)
    ]


class TestTypedOutput(unittest.TestCase):
    def test_the_output_is_validated_to_the_requested_type(self):
        conversation = make_conversation([VALID])

        turn = conversation.send("hello")

        self.assertIsInstance(turn.output, AgentResponse)
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

        self.assertEqual(len(conversation.messages), 4)
        # The second request carried the first exchange: its request and
        # response, then the new prompt.
        sent = conversation.script.calls[1]
        self.assertEqual(len(sent), 3)

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
