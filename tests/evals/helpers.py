"""Shared helpers for eval tests that call real LLMs."""

from pathlib import Path

from chat_workflow import ConversationFlowState, ConversationTools
from chat_workflow.config import Config

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def make_config() -> Config:
    return Config(_CONFIG_PATH)


def make_tools(io) -> ConversationTools:
    return ConversationTools(
        io=io,
        state=ConversationFlowState(),
        config=make_config(),
    )
