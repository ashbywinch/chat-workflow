"""Unit tests for the eval harness itself — no LLM calls.

The harness must never turn a workflow failure or a missing typed result into a
judge verdict. These tests use fake model methods and fake judges.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast, override
from unittest.mock import patch

from chat_workflow import Config, Session, SessionLog
from chat_workflow.session import UserIO
from tests.evals.helpers import (
    AgentIO,
    JudgeResult,
    JudgeVerdict,
    capture_on_failure,
    format_transcript,
    run_multi_turn_eval,
)


class TestEvalHarness(unittest.TestCase):
    @override
    def setUp(self):
        self.config = SimpleNamespace(
            model="test-model",
            provider="test-provider",
            max_retries=0,
            request_timeout_seconds=1,
            model_supports_tools=False,
            api_base=None,
            api_key_env=None,
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"CHAT_WORKFLOW_TRANSCRIPT_DIR": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _judge(self, passed: bool):
        def judge(rules, transcript, config):
            label = next(iter(rules))
            return JudgeResult(
                verdicts=[
                    JudgeVerdict(
                        rule=label,
                        passed=passed,
                        explanation=None if passed else "deliberate failure",
                    )
                ]
            )

        return judge

    def _run(self, model_method, *, judge=None, test_name="harness"):
        return run_multi_turn_eval(
            model_method=model_method,
            method_kwargs={},
            user_persona="You are a test user.",
            judge_rules={"rule": "description"},
            judge=judge or self._judge(True),
            config=cast(Config, self.config),
            test_name=test_name,
        )

    def test_workflow_exception_is_a_workflow_failure(self):
        def model(**kwargs):
            raise RuntimeError("boom")

        with self.assertRaises(AssertionError) as ctx:
            self._run(model, test_name="workflow_raises")

        self.assertIn("[workflow failed]", str(ctx.exception))
        self.assertTrue(any(name.startswith("workflow_raises-") for name in os.listdir(self.tmp.name)))

    def test_none_result_is_a_no_result_failure(self):
        def model(**kwargs):
            return None

        with self.assertRaises(AssertionError) as ctx:
            self._run(model, test_name="no_result")

        self.assertIn("[no typed result]", str(ctx.exception))

    def test_judge_failure_is_a_judge_failure(self):
        def model(**kwargs):
            kwargs["session"].state.messages.append({"role": "assistant", "content": "hello"})
            return ["ok"]

        with self.assertRaises(AssertionError) as ctx:
            self._run(model, judge=self._judge(False), test_name="judge_fails")

        self.assertIn("[judge failed]", str(ctx.exception))
        exception_files = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith("judge_fails-") and name.endswith("-exception.txt")
        ]
        self.assertTrue(exception_files)
        details = Path(self.tmp.name, exception_files[0]).read_text()
        self.assertIn("=== RESULT ===", details)

    def test_empty_transcript_is_a_no_conversation_failure(self):
        def model(**kwargs):
            return ["ok"]

        def judge(rules, transcript, config):
            raise AssertionError("judge must not run without a conversation")

        with self.assertRaises(AssertionError) as ctx:
            self._run(model, judge=judge, test_name="no_conversation")

        self.assertIn("[no conversation]", str(ctx.exception))
        self.assertNotIn("judge must not run", str(ctx.exception))

    def test_judge_is_not_called_when_workflow_fails(self):
        def model(**kwargs):
            raise RuntimeError("boom")

        def judge(rules, transcript, config):
            raise AssertionError("judge must not run after a workflow failure")

        with self.assertRaises(AssertionError) as ctx:
            self._run(model, judge=judge, test_name="workflow_raises_judge")

        self.assertIn("[workflow failed]", str(ctx.exception))
        self.assertNotIn("judge must not run", str(ctx.exception))

    def test_success_returns_the_result(self):
        def model(**kwargs):
            kwargs["session"].state.messages.append({"role": "assistant", "content": "hello"})
            return ["ok"]

        result = self._run(model, test_name="success")

        self.assertEqual(result, ["ok"])


class TestAgentIOUserBot(unittest.TestCase):
    """The user bot must not return an empty turn to the agent."""

    @staticmethod
    def _response(text: str):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    @patch("tests.evals.helpers.litellm.completion")
    def test_empty_response_is_retried(self, mock_completion):
        mock_completion.side_effect = [
            self._response(""),
            self._response("Yes, that looks right."),
        ]
        io = AgentIO("You are a test user.", cast(Config, SimpleNamespace(model="test")))

        text = io.prompt("You")

        self.assertEqual(text, "Yes, that looks right.")
        self.assertEqual(mock_completion.call_count, 2)

    @patch("tests.evals.helpers.litellm.completion")
    def test_three_empty_responses_raise(self, mock_completion):
        mock_completion.side_effect = [self._response("") for _ in range(3)]
        io = AgentIO("You are a test user.", cast(Config, SimpleNamespace(model="test")))

        with self.assertRaises(RuntimeError):
            io.prompt("You")

    @patch("tests.evals.helpers.litellm.completion")
    def test_length_finish_fails_fast(self, mock_completion):
        """A max_tokens cutoff is not retried: the same request cannot fix it."""
        mock_completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=""),
                    finish_reason="length",
                )
            ]
        )
        io = AgentIO("You are a test user.", cast(Config, SimpleNamespace(model="test")))

        with self.assertRaises(RuntimeError) as ctx:
            io.prompt("Incidental notes for Draft Minutes")

        self.assertIn("max_tokens", str(ctx.exception))
        self.assertIn("Incidental notes for Draft Minutes", str(ctx.exception))
        self.assertEqual(mock_completion.call_count, 1)


class TestSessionTranscript(unittest.TestCase):
    """Composite workflow step IO must be in the transcript the judge reads."""

    def test_composite_io_is_recorded_in_transcript(self):
        class FakeIO(UserIO):
            @override
            def echo(self, message: str) -> None:
                pass

            @override
            def prompt(self, label: str, **kwargs: object) -> str:
                return "None"

        session = Session(io=FakeIO(), state=SessionLog())
        session.io.echo("The main parts are Raw Notes and Draft Minutes.")
        answer = session.io.prompt("Incidental notes for Draft Minutes")
        self.assertEqual(answer, "None")

        transcript = format_transcript(session)
        self.assertIn("The main parts are", transcript)
        self.assertIn("Incidental notes for Draft Minutes", transcript)
        self.assertIn("None", transcript)


class TestStructuredTranscript(unittest.TestCase):
    """The canonical event log has separate judge and human projections."""

    class _FakeIO(UserIO):
        @override
        def echo(self, message: str) -> None:
            pass

        @override
        def prompt(self, label: str, **kwargs: object) -> str:
            return "answer"

    def _session(self) -> Session:
        session = Session(io=self._FakeIO(), state=SessionLog())
        session.state.record("user", "hello")
        session.state.record("framework", "rejected envelope", kind="rejected_response")
        session.state.record("assistant", "hi")
        session.state.record("validation", "bad rule", turn=1, recovered=True)
        return session

    def test_judge_transcript_is_conversation_only(self):
        session = self._session()

        text = format_transcript(session, result={"x": 1}, for_judge=True)

        self.assertIn("[user]\nhello", text)
        self.assertIn("[assistant]\nhi", text)
        self.assertIn("[structured-result]", text)
        self.assertNotIn("rejected envelope", text)
        self.assertNotIn('"x"', text)
        self.assertNotIn("[validation", text)

    def test_human_transcript_keeps_harness_and_result(self):
        session = self._session()

        text = format_transcript(session, result={"x": 1})

        self.assertIn("[framework]\nrejected envelope", text)
        self.assertIn("[validation events]", text)
        self.assertIn('"x": 1', text)

    def test_judge_transcript_includes_given_context(self):
        """The judge must see inputs from earlier phases, or 'you said' looks invented."""
        session = self._session()

        text = format_transcript(
            session,
            result={"x": 1},
            for_judge=True,
            given={"name": "Meeting Minutes", "owner_role": "Meeting Coordinator"},
        )

        self.assertIn("[given]", text)
        self.assertIn("Meeting Coordinator", text)
        self.assertLess(text.index("[given]"), text.index("[user]"))

    def test_rejected_pretty_printed_envelope_is_framework(self):
        session = Session(io=self._FakeIO(), state=SessionLog())
        session.state.messages = [
            {"role": "assistant", "content": '{\n  "intent": "success"\n}'}
        ]

        judge_text = format_transcript(session, for_judge=True)
        human_text = format_transcript(session)

        self.assertNotIn("intent", judge_text)
        self.assertIn("[framework]", human_text)

    def test_record_json_contains_events_result_and_verdicts(self):
        session = self._session()
        verdict = JudgeResult(
            verdicts=[
                JudgeVerdict(rule="r", passed=False, explanation="bad", evidence="hi", fix="say hello")
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.dict(os.environ, {"CHAT_WORKFLOW_TRANSCRIPT_DIR": tmp}),
                self.assertRaises(AssertionError),
                capture_on_failure(
                    session,
                    label="record_test",
                    result={"x": 1},
                    judge_result=verdict,
                ),
            ):
                raise AssertionError("boom")

            record_path = next(Path(tmp).glob("record_test-*.json"))
            record = json.loads(record_path.read_text())

        self.assertEqual(record["events"][0]["type"], "user")
        self.assertEqual(record["result"], {"x": 1})
        self.assertEqual(record["judge"]["verdicts"][0]["rule"], "r")
        self.assertEqual(record["judge"]["verdicts"][0]["evidence"], "hi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
