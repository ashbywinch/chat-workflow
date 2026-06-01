# ruff: noqa: E501 — LLM prompt docstrings contain long example dialogue lines
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .output import Output
from .process_analysis import ProcessAnalysis


class Input(BaseModel):
    """A single workflow input."""

    source: str = Field(..., description="Where this input originates", min_length=1)
    format: str = Field(..., description="Exact format/structure", min_length=1)
    trigger_conditions: str = Field(..., description="What initiates workflow execution")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required inputs from other components",
    )
    validation_criteria: str = Field(..., description="How to verify input completeness", min_length=1)

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        analysis: Annotated[ProcessAnalysis | None, "The process analysis, if already available"] = None,
        outputs: Annotated[list[Output] | None, "The outputs the process should produce, if already known"] = None,
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> list[Input]:
        """You are helping the user figure out what ingredients or raw materials their process needs.

        The user has described what they want to produce (their outputs). Your job is to figure
        out what inputs or starting materials they need to make those outputs happen.

        IMPORTANT RULES:
        - Speak in the user's language. If they're cooking, talk about ingredients, kitchen equipment, and recipes.
        - NEVER use model field names with the user. Instead of "source" ask "where does this come from?" Instead of "format" ask "what form is it in?" Instead of "trigger_conditions" ask "what kicks things off?" Instead of "validation_criteria" ask "how do you know you have everything you need?"
        - Never adopt an analytical or consulting tone. Talk like a helpful person helping out, not an analyst documenting a process.
        - Propose what you can based on what you know. If the user gave enough detail, summarize it back: "So from what you've said, you'd need a list of meals you fancy, a note of how many days you're cooking for, and a record of what kitchen equipment you have — does that capture everything?" If you're suggesting something new, make that clear: "How about we start with a list of meals you fancy, and then figure out what ingredients you'd need for each — does that sound like it would work?"
        - If the user is confused, simplify your language. Don't re-explain — just use simpler words.
        - When the user confirms, move on. Don't re-ask.
        - Never put fabricated values in the final output. Propose ideas and let the user confirm/correct.
        """
        ...  # type: ignore[reportReturnType]
