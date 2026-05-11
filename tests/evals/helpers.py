"""Shared helpers for eval tests that call real LLMs."""

from pathlib import Path

from chat_workflow import Config, Session, SessionLog

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"


def make_config() -> Config:
    return Config(_CONFIG_PATH)


def make_tools(io) -> Session:
    return Session(
        io=io,
        state=SessionLog(),
        config=make_config(),
    )
