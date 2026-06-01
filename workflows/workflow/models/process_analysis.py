# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow


class ProcessAnalysis(BaseModel):
    """Output of analyze_process()."""

    phases: list[str] = Field(..., description="Logical process phases in order", min_length=1)
    activities: list[str] = Field(..., description="Specific business activities", min_length=1)
    orchestrating_component: str = Field(..., description="Which domain coordinates this workflow", min_length=1)
    participants: list[str] = Field(..., description="All roles/systems involved", min_length=1)

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        process_description: Annotated[str, "A description of the business process to analyze"],
        outputs: Annotated[list[Output] | None, "The outputs the process produces, if already known"] = None,
        inputs: Annotated[list[Input] | None, "The inputs the process needs, if already known"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ProcessAnalysis:
        """You help the user understand the structure of their process end-to-end.

        The user has already described what they want to produce (their outputs) and what
        they have to work with (their inputs). Now help them see how it all fits together —
        the steps involved, in what order, and who or what is involved.

        IMPORTANT RULES:
        - Speak in the user's language. If they're meal planning, talk about "thinking of meals, checking what you can make, planning the week, writing the shopping list" — not "phases and activities."
        - NEVER use model field names like "phases", "activities", "orchestrating_component", or "participants" with the user. Instead ask: "what are the main stages?", "what happens in each stage?", "who or what makes it happen?"
        - Ask one question at a time. Do not overwhelm the user with multiple questions in a single turn.
        - Use plain, natural language. You are having a conversation, not writing a report. Analyze and synthesize what the user tells you to help them see their process more clearly.
        - Propose what you can based on what the user told you. Summarize what you've heard: "So from what you've described, it sounds like there are three stages: deciding what you want to eat, checking what you can make, and writing up the plan — is that right?" Or suggest a structure: "How about we break it down as: first you think about what you fancy, then you check what you can actually make, then you write it all up — does that sound about right?"
        - If the user is confused, simplify your language. You started in plain language, so there's no jargon to "drop out of."
        - Never put fabricated values in the final output. Propose ideas and let the user confirm/correct.
        """
        ...  # type: ignore[reportReturnType]


from .input import Input  # noqa: E402 — placed after class def to break circular import
from .output import Output  # noqa: E402
