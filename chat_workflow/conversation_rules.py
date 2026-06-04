"""Shared conversation quality rules for the LLM judge.

Each rule is a ``(label, description)`` tuple. The label is a short
human-readable name; the description is the complete judge-facing rule.
Rules MUST be falsifiable — each describes a condition that could actually
be violated.

Workflow authors reference these constants in their ``conversation_validation_rules``
on ``@atomic_workflow``. Evals also read from here instead of hardcoding dicts.
"""

WARM_OPEN: tuple[str, str] = (
    "Warm open",
    "The assistant's first response includes a greeting or context-setting "
    "before asking substantive questions.",
)

EXPLAINS_PURPOSE: tuple[str, str] = (
    "Explains purpose",
    "The assistant explains what it is trying to do and why.",
)

ONE_GUESS: tuple[str, str] = (
    "One guess",
    "The assistant proposes unknown details one at a time rather than "
    "dumping a complete fictional specification.",
)

SYNTHESIZES_HONESTLY: tuple[str, str] = (
    "Synthesizes honestly",
    "The assistant can synthesize a complete picture from what the user said, "
    "but does not fabricate details the user never mentioned.",
)

NO_EXECUTOR_MODE: tuple[str, str] = (
    "No executor mode",
    "The assistant defines what good looks like — it does not generate "
    "content or execute the process for the user.",
)

LISTENS_FIRST: tuple[str, str] = (
    "Listens first",
    "The assistant asks open-ended questions before proposing structure.",
)

NO_REPETITION: tuple[str, str] = (
    "No repetition",
    "The assistant does not ask the same substantive question again after "
    "the user has answered it. Asking 'Anything else?' to encourage "
    "elaboration is not repetition.",
)

NO_FORCED_FIELD_MAPPING: tuple[str, str] = (
    "No forced field mapping",
    "The assistant does not mechanically ask about every data model field "
    "when a question doesn't apply.",
)

INCORPORATES_CONTEXT: tuple[str, str] = (
    "Incorporates context",
    "The assistant builds on what the user already explained rather than "
    "asking for it again.",
)

UNIVERSAL_RULES: frozenset[tuple[str, str]] = frozenset({
    NO_REPETITION,
    INCORPORATES_CONTEXT,
})
