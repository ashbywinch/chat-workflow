"""Shared helpers for eval tests that call real LLMs."""

from pathlib import Path

from chat_workflow import Config, Session, SessionLog

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"

# The gateway caches a completion by request body and replays it without
# calling the model. An eval that is served a stored answer measures the cache,
# not the prompt — and a single stored refusal can fail the suite repeatedly
# until the entry expires.
#
# A cold reasoning route also exceeds the 30s production timeout: the first
# eval of a CI run has been seen to spend every attempt on the clock. An eval
# that reports that as a hang measures the network, not the prompt.
_EVAL_OVERRIDES = {
    "llm": {
        "extra_headers": {"cf-aig-skip-cache": "true"},
        "request_timeout_seconds": 90,
    }
}


def make_config() -> Config:
    """The house config, with the gateway cache bypassed and room for a cold call."""
    return Config(_CONFIG_PATH, overrides=_EVAL_OVERRIDES)


def make_tools(io) -> Session:
    return Session(
        io=io,
        state=SessionLog(),
        config=make_config(),
    )


def eval_timeout(config: Config) -> int:
    """A ceiling for one eval, derived from the budget the workflow may spend.

    ``request_timeout_seconds`` per attempt, ``max_retries`` extra attempts: a
    tighter alarm fails the framework for using the retry policy it was
    configured with, and reads as a defect.
    """
    return int(config.request_timeout_seconds * (config.max_retries + 1)) + 15
