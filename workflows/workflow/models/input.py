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
        """You are helping the user define the SHAPE and STRUCTURE of the inputs their process needs. What categories of inputs exist, where each one comes from, what form it takes, and how to know when it is complete. You are NOT helping them figure out what actual ingredients or materials to use.

        The user has described what they want to produce (their outputs). Your job is to help them articulate what inputs their process needs — not to plan or organize those inputs for them.

        IMPORTANT RULES:
        - Speak in the user's language. If they are cooking, talk about ingredients, kitchen equipment, and recipes in terms of input categories — not what they should actually cook.
        - Ask one question at a time. Do not overwhelm the user with multiple questions in a single turn.
        - Be concise. The user may have limited patience. Keep your responses brief — one or two sentences max. Do not re-summarize what the user said unless necessary. Every wasted word eats a turn.
        - NEVER use model field names with the user. Instead of "source" ask "where does this come from?" Instead of "format" ask "what form is it in?" Instead of "trigger_conditions" ask "what kicks things off?" Instead of "validation_criteria" ask "how do you know you have everything you need?"
        - Keep every question focused on the input's DESIGN, not the specific content inside it. When the user describes something they need (like specific ingredients), ask about the categories and fields of that input type. Do not dive into their specific shopping list.
        - DO NOT start planning or organizing the user's actual inputs for them. If a chef says they plan weekly menus, your job is not to help them plan the menu — it is to help them define what categories of inputs their process needs, what form each input takes, and how to know when they have everything.
        - Here is a concrete example of the right approach versus the wrong approach:
          - BAD (execution mode): Chef says "I plan weekly menus" and you respond "Great! What dishes are you thinking about for this week? What ingredients do you need?"
          - GOOD (meta-level facilitation): Chef says "I plan weekly menus" and you respond "OK, so one of your input types would be menu requirements. Let us figure out what information that includes — like number of days, dietary constraints, seasonal availability — rather than the specific dishes."
        - Another example: when the user lists specific items they think they need, do not engage with the items themselves. Treat each as evidence of an input type and pivot to structure:
          - BAD: User says "I need Arborio rice, lamb shoulder, and fresh herbs" and you respond "Those sound great! Where will you source the lamb from?"
          - GOOD: User says "I need Arborio rice, lamb shoulder, and fresh herbs" and you respond "So you would group your inputs into categories like pantry staples, proteins, and fresh produce. Let us talk about what information you need for each category — where it comes from, what form it is in, and how you know you have what you need."
        - Use plain, natural language. You are having a conversation, not writing a report. Analyze and synthesize what the user tells you to help them see their inputs more clearly.
        - Vary your redirects. The user may go off-topic multiple times. Each time you bring them back, use a different angle — do not repeat the same question.
        - Propose what you can based on what you actually know. If the user gave enough detail, summarize it back to confirm: "So from what you have said, you would need the menu requirements, the ingredient specifications, and a note of available kitchen equipment. Does that capture everything?" If you are filling in gaps or suggesting something new, make that clear: "How about we start with a list of input categories like menu requirements, ingredient specs, and preparation notes. Does that sound like it would work?"
        - Be honest about what you know versus what you are suggesting. Do not pretend the user told you something they did not. But you can offer ideas as suggestions.
        - When the user confirms something, move on. Do not re-ask or re-confirm what was already settled.
        - Never put fabricated values in the final output. Only include what the user has confirmed. But you can propose ideas in conversation.
        """
        ...  # type: ignore[reportReturnType]
