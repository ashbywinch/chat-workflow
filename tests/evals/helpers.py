"""Shared helpers for eval tests that call real LLMs."""

from pathlib import Path

import litellm

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


class AgentIO:
    """UserIO implementation backed by an LLM — plays a realistic user role.

    Instead of hardcoded MockIO responses, this generates natural user responses
    by calling the LLM with a persona prompt. This creates more realistic
    multi-turn conversations for eval tests.

    The user bot is an expert in their own domain (their meetings) but knows
    nothing about workflow decomposition. They just want help with their problem.
    """

    def __init__(self, persona_prompt: str, config: Config):
        self.persona_prompt = persona_prompt
        self.config = config
        self.outputs: list[str] = []
        # Conversation history from user bot's perspective:
        # - workflow agent messages are "user" (they're addressing the user bot)
        # - user bot responses are "assistant" (the bot is responding)
        self._history: list[dict[str, str]] = []

    def echo(self, message: str) -> None:
        """Capture what the workflow agent said."""
        self.outputs.append(message)
        # Strip the "\nAssistant: " prefix that Session.run adds
        clean = message.removeprefix("\nAssistant: ").strip()
        if clean:
            # From user bot's perspective, the workflow is the "user" talking to them
            self._history.append({"role": "user", "content": clean})

    def prompt(self, label: str) -> str:
        """Generate a user response using the LLM."""
        # Build messages: system persona + conversation history
        messages = [{"role": "system", "content": self.persona_prompt}]
        messages.extend(self._history)

        # Suppress litellm debug output
        litellm.suppress_debug_info = True

        # Call LLM to generate user response
        response = litellm.completion(
            model=self.config.model,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )

        user_text = response.choices[0].message.content or ""
        # Store this response in history
        self._history.append({"role": "assistant", "content": user_text})
        return user_text


def llm_judge(question: str, content: str, config: Config, max_tokens: int = 200) -> tuple[bool, str]:
    """Ask an LLM to evaluate whether *content* satisfies *question*.

    Uses the same model as the evals. The question should be a yes/no
    question about the content. Returns (passed, reasoning) where
    *passed* is True/False and *reasoning* is the judge's explanation.

    This is useful for evaluating non-deterministic outcomes like
    "is this analysis relevant to meeting minutes?" where keyword
    matching would be too brittle. The reasoning helps diagnose
    what the judge didn't like.
    """
    litellm.suppress_debug_info = True
    response = litellm.completion(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict but fair evaluator. Answer the question about the "
                    "content below. First say YES or NO on its own line, "
                    "then on the next line give a brief reason for your verdict."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContent: {content}",
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    answer = (response.choices[0].message.content or "").strip()
    lines = answer.split("\n")
    verdict = lines[0].strip().upper()
    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    passed = verdict.startswith("YES")
    return passed, reasoning
