"""Tests for eval helper utilities (token counting, EvalStats)."""
import unittest
from unittest.mock import MagicMock

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
        stats = EvalStats(test_name="test_foo", duration_s=12.34, total_tokens=5678)
        self.assertEqual(stats.report(), "  [test_foo] 12s  5678 tok")

    def test_report_zero_tokens(self):
        stats = EvalStats(test_name="test_bar", duration_s=0.5, total_tokens=0)
        self.assertEqual(stats.report(), "  [test_bar] 0s  0 tok")
