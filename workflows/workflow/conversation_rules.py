"""Workflow-specific conversation rules shared by steps.

The generic rules live in ``chat_workflow.conversation_rules``. These rules are
about this package's artifacts and are shared by the process-definition and
component-spec steps, so the behaviour has one definition.
"""

# lucidlint: ignore-file global-state this module is the workflow rule catalogue; authors import each named rule
SELF_ONLY_ROLE: tuple[str, str] = (
    "The user's own role",
    "When the user says the work is theirs alone — no one else does it — a "
    "role they do themselves is written '<role> (You)', for example 'Expense "
    "Reporter (You)', or 'You' when there is no role name. The marker makes "
    "it the user, not a separate person. The assistant does not propose or "
    "keep a bare job title for the user, even if a gap seems to need one.",
)

CONFIRM_THE_DRAFT: tuple[str, str] = (
    "Confirms the draft already shown",
    "The complete draft was shown to the user in the previous turn, and this "
    "turn answers their reaction. Return the draft with only the corrections "
    "the user made; do not re-derive values the draft already has, do not "
    "present the whole draft again, and do not repeat the question the user "
    "has just answered.",
)
