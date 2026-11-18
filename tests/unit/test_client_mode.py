"""The client's mode: JSON, or native tool calling when the model can."""

from __future__ import annotations

import os
import unittest
from typing import override
from unittest.mock import patch

import instructor

from chat_workflow.llm_interaction import get_client


class TestClientMode(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        self.env.start()

    @override
    def tearDown(self) -> None:
        self.env.stop()

    @patch("chat_workflow.llm_interaction.instructor.from_litellm")
    def test_default_is_json_mode(self, from_litellm) -> None:
        get_client("openai")
        assert from_litellm.call_args.kwargs["mode"] is instructor.Mode.JSON

    @patch("chat_workflow.llm_interaction.instructor.from_litellm")
    def test_a_model_with_tools_gets_tool_mode(self, from_litellm) -> None:
        """A model that can call tools returns the structured result around
        them; without TOOLS mode the tools have no path through the client."""
        get_client("openai", model_supports_tools=True)
        assert from_litellm.call_args.kwargs["mode"] is instructor.Mode.TOOLS
