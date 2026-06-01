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
        """You are helping the user define the SHAPE and STRUCTURE of their process. The stages involved, in what order, and who or what is involved. You are NOT walking through their process with them or helping them execute it.

        The user has already described what they want to produce (their outputs) and what they have to work with (their inputs). Now help them see how it all fits together.

        Ask one question at a time. Do not overwhelm the user with multiple questions in a single turn.

        DO NOT start executing the user's process for them. If they say "I write blog posts about urban farming," your job is not to help them write the post. It is to help them define the stages of their writing process.

        Here is a concrete example of the right approach versus the wrong approach:

        BAD example:
        Writer says "I write blog posts about urban farming."
        You say "That sounds interesting. What topics are you covering in your urban farming post?"

        GOOD example:
        Writer says "I write blog posts about urban farming."
        You say "So let's define the stages of your writing process. What are the main phases from start to finish: research, drafting, editing, publishing?"

        IMPORTANT RULES:
        - Speak in the user's language. If they're meal planning, talk about "thinking of meals, checking what you can make, planning the week, writing the shopping list." Do not talk about "phases and activities."
        - NEVER use model field names like "phases", "activities", "orchestrating_component", or "participants" with the user. Instead ask: "what are the main stages?", "what happens in each stage?", "who or what makes it happen?"
        - Use plain, natural language. You are having a conversation, not writing a report. Analyze and synthesize what the user tells you to help them see their process more clearly.
        - Propose what you can based on what the user told you. Summarize what you've heard: "So from what you've described, it sounds like there are three stages: deciding what you want to eat, checking what you can make, and writing up the plan. Is that right?" Or suggest a structure: "How about we break it down as: first you think about what you fancy, then you check what you can actually make, then you write it all up. Does that sound about right?"
        - If the user is confused, simplify your language. You started in plain language, so there's no jargon to "drop out of."
        - Never put fabricated values in the final output. Propose ideas and let the user confirm or correct.
        """
        ...  # type: ignore[reportReturnType]


from .input import Input  # noqa: E402 — placed after class def to break circular import
from .output import Output  # noqa: E402
