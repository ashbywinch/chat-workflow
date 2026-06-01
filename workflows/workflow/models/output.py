# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .process_analysis import ProcessAnalysis


class Output(BaseModel):
    """A single workflow output."""

    consumer: str = Field(..., description="Which components use this output", min_length=1)
    format: str = Field(..., description="Exact format/structure", min_length=1)
    success_criteria: str = Field(..., description="How to measure output quality", min_length=1)
    integration_points: str = Field(..., description="How outputs connect downstream")
    storage_requirements: str = Field(..., description="Where/how outputs are preserved")

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis | None, "The process analysis, if already available"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Output]:
        """You are helping the user figure out what results or outputs their task or process produces.
        The user has described what they're trying to do.

        IMPORTANT RULES:
        - Speak in the user's language, not technical jargon. If they mention cooking, talk about food and recipes, not "outputs" and "consumers."
        - NEVER use model field names with the user. Instead of "consumer" say "who uses this." Instead of "format" say "what should it look like." Instead of "success_criteria" say "how do you know it's good?" Instead of "integration_points" say "what happens next with this?" Instead of "storage_requirements" say "where should this live?"
        - Never adopt an analytical or consulting tone. Start naturally — your first message should sound like a helpful friend, not a business analyst sent to document their process.
        - Propose what you can based on what you actually know. If the user gave you enough detail, summarize it back to confirm: "So you want a meal plan organized by day, and a shopping list to go with it — is that right?" If you're filling in gaps or suggesting something new, make that clear: "How about we organise it by day, with each day listing breakfast, lunch and dinner — does that sound like it would work for you?" or "So would you normally start by listing the meals you fancy, and then adding more to fill the week?"
        - Be honest about what you know vs what you're suggesting. Don't pretend the user told you something they didn't. But you can offer ideas as suggestions.
        - When the user confirms something, move on. Do not re-ask or re-confirm what was already settled.
        - Never put fabricated values in the final output. Only include what the user has confirmed. But you can propose ideas in conversation.
        """
        ...  # type: ignore[reportReturnType]
