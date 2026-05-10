"""Shared helpers for eval tests that call real LLMs."""

from pathlib import Path

from chat_workflow.config import Config
from chat_workflow.conversation_log import ConversationLog
from chat_workflow.conversation_tools import ConversationTools

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def make_config() -> Config:
    return Config(_CONFIG_PATH)


def make_tools(io) -> ConversationTools:
    return ConversationTools(
        io=io,
        state=ConversationLog(),
        config=make_config(),
    )
