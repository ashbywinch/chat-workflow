"""Tests for eval helper utilities (token counting, EvalStats)."""
import unittest
from unittest.mock import MagicMock, patch

import tests.evals.helpers as helpers
from tests.evals.helpers import EvalStats, _token_counter_callback


class TestTokenCounter(unittest.TestCase):
    def setUp(self):
        helpers._token_count = 0

    def test_callback_accumulates_tokens(self):
        resp = MagicMock()
        resp.usage = MagicMock()
        resp.usage.total_tokens = 150
        _token_counter_callback({}, resp, 0, 0)
        self.assertEqual(helpers._token_count, 150)

    def test_callback_no_usage_skips(self):
        resp = MagicMock()
        resp.usage = None
        _token_counter_callback({}, resp, 0, 0)
        self.assertEqual(helpers._token_count, 0)

    def test_callback_accumulates_multiple_calls(self):
        resp = MagicMock()
        resp.usage = MagicMock()
        resp.usage.total_tokens = 100
        _token_counter_callback({}, resp, 0, 0)
        _token_counter_callback({}, resp, 0, 0)
        self.assertEqual(helpers._token_count, 200)


class TestEvalStats(unittest.TestCase):
    def test_report_format(self):
        stats = EvalStats(test_name="test_foo", duration_s=12.34, agent_tokens=5000, judge_tokens=678)
        self.assertEqual(stats.report(), "  [test_foo] 12s  5000 agent tok  678 judge tok  5678 tot")

    def test_report_zero_tokens(self):
        stats = EvalStats(test_name="test_bar", duration_s=0.5, agent_tokens=0, judge_tokens=0)
        self.assertEqual(stats.report(), "  [test_bar] 0s  0 agent tok  0 judge tok  0 tot")


class TestEvalCacheBypass(unittest.TestCase):
    """Every eval call bypasses the gateway cache, including the user bot."""

    def test_user_bot_passes_gateway_headers(self):
        config = MagicMock()
        config.model = "openai/dynamic/fallback2"
        config.extra_headers = {"cf-aig-skip-cache": "true"}
        reply = MagicMock()
        reply.choices[0].message.content = "hello"
        reply.choices[0].message.reasoning_content = None
        reply.choices[0].finish_reason = "stop"

        with patch("tests.evals.helpers.litellm.completion", return_value=reply) as completion:
            bot = helpers.AgentIO("persona", config)
            self.assertEqual(bot.prompt("\nYou"), "hello")

        self.assertEqual(
            completion.call_args.kwargs["extra_headers"],
            {"cf-aig-skip-cache": "true"},
        )

    def test_judge_passes_gateway_headers(self):
        from tests.evals.helpers import llm_judge

        config = MagicMock()
        config.provider = "openai"
        config.api_key_env = "OPENAI_API_KEY"
        config.api_base = None
        config.model_supports_tools = False
        config.model = "openai/dynamic/fallback2"
        config.max_retries = 3
        config.request_timeout_seconds = 30
        config.extra_headers = {"cf-aig-skip-cache": "true"}
        client = MagicMock()

        with patch("chat_workflow.get_client", return_value=client):
            llm_judge({"Rule": "Description"}, "transcript", config)

        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["extra_headers"],
            {"cf-aig-skip-cache": "true"},
        )


class TestConversationJudgeCoherence(unittest.TestCase):
    def test_a_verdict_that_finds_no_violation_is_not_a_failure(self):
        from tests.evals.helpers import JudgeVerdict, _judge_says_it_passes

        verdict = JudgeVerdict(
            rule="No question after pushback",
            passed=False,
            explanation=(
                "The assistant changed approach and asked a different question. "
                "No violation found."
            ),
        )
        self.assertTrue(_judge_says_it_passes(verdict))

    def test_a_real_failure_is_kept(self):
        from tests.evals.helpers import JudgeVerdict, _judge_says_it_passes

        verdict = JudgeVerdict(
            rule="No question after pushback",
            passed=False,
            explanation="The assistant asked the same question again, so it fails.",
        )
        self.assertFalse(_judge_says_it_passes(verdict))


class TestEvalGatewayConfig(unittest.TestCase):
    """Evals must fail loudly when the gateway endpoint is missing/placeholder."""

    def test_missing_base_url_is_rejected(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            helpers.make_config()

    def test_placeholder_base_url_is_rejected(self):
        with (
            patch.dict("os.environ", {"OPENAI_BASE_URL": "https://gateway.example/v1/{ACCOUNT_ID}/{GATEWAY}/compat"}),
            self.assertRaises(RuntimeError),
        ):
            helpers.make_config()

    def test_gateway_base_url_is_accepted(self):
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://gateway.example/v1/account/gateway/compat"}):
            self.assertEqual(helpers.make_config().provider, "openai")
