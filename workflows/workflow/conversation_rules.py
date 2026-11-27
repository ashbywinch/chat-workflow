"""Workflow-specific conversation rules shared by steps.

The generic rules live in ``chat_workflow.conversation_rules``. These rules are
about this package's artifacts and are shared by the process-definition and
component-spec steps, so the behaviour has one definition.
"""

# lucidlint: ignore-file global-state this module is the workflow rule catalogue; authors import each named rule
SELF_ONLY_ROLE: tuple[str, str] = (
    "The user's own role",
    "When the user says they do the work themselves — 'it's just me', 'no one "
    "else', 'I do it all' — the person in charge or owner is 'You', and any "
    "participants are the user too. The assistant does not keep or invent a "
    "job title for the user.",
)

CONFIRM_THE_DRAFT: tuple[str, str] = (
    "Confirms the draft already shown",
    "The complete draft was shown to the user in the previous turn, and this "
    "turn answers their reaction. Return the draft with only the corrections "
    "the user made; do not re-derive values the draft already has, do not "
    "present the whole draft again, and do not repeat the question the user "
    "has just answered.",
)
