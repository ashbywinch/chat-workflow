"""The eval endpoint: a stated base wins, an openai preset is pinned to the gateway."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_workflow.config import Config


def _config(llm: dict) -> Config:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"llm": {"model": "openai/dynamic/fallback2", **llm}}, handle)
    path = Path(handle.name)
    try:
        return Config(path)
    finally:
        path.unlink()


class TestApiBase(unittest.TestCase):
    def test_a_stated_base_wins(self) -> None:
        config = _config({"provider": "openai", "api_base": "https://example.test/v1"})
        assert config.api_base == "https://example.test/v1"

    def test_an_openai_preset_is_pinned_to_the_gateway(self) -> None:
        """Without this, a preset with no base falls through to litellm's
        ambient base — which is how the evals reached an endpoint that wanted a
        header nobody sends, and every model call failed there instead."""
        config = _config({"provider": "openai"})
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://gateway.test/compat"}):
            assert config.api_base == "https://gateway.test/compat"

    def test_another_provider_keeps_its_default(self) -> None:
        config = _config({"provider": "openrouter"})
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://gateway.test/compat"}):
            assert config.api_base is None
