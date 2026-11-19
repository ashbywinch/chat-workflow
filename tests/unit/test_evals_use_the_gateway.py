"""Every eval routes through the house gateway, never a provider endpoint."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from chat_workflow.config import Config

CONFIG = Path(__file__).resolve().parents[2] / "config.json"


class TestEvalsUseTheGateway(unittest.TestCase):
    def test_the_eval_config_names_no_provider_endpoint(self) -> None:
        """A base in the repo is a host the house gateway does not own: it
        needs its own credentials, and CI has none — every eval call fails
        there with that host's error instead of testing the change."""
        llm = json.loads(CONFIG.read_text())["llm"]
        assert not llm.get("api_base"), f"config.json pins {llm['api_base']!r}; the eval must use the gateway"

    def test_the_model_is_a_gateway_route(self) -> None:
        llm = json.loads(CONFIG.read_text())["llm"]
        assert llm["model"].startswith("openai/dynamic/"), llm["model"]

    def test_the_key_is_the_gateway_token(self) -> None:
        assert Config(CONFIG).api_key_env == "OPENAI_API_KEY"
