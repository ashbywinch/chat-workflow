"""Eval tests for ProcessDefinition sub-workflows (_gather_notes, _generate_from_notes).

These call real LLMs and require a configured config.json with a valid API key.
Evals fail (not skip) when infrastructure is missing.
"""

import unittest

from chat_workflow.conversation_rules import NO_REPETITION
from tests.evals.helpers import run_multi_turn_eval
from workflows.workflow.models import ProcessDefinition, generate_from_chat

# A ProcessDefinition describes the user's own work; it is not a tool or a
# template. The judge asks whether the assistant derived the breakdown from
# what the user said (grouping steps into stages), not whether it proposed a
# product.
PROCESS_DEFINITION_RULES = {
    NO_REPETITION[0]: NO_REPETITION[1],
    "Uses expertise": (
        "The `_generate_from_notes` Output contains the stages, steps, "
        "participants, and role in charge, and they are derived from the step's "
        "Input notes. The `_gather_notes` Conversation may have proposed a "
        "plain-language breakdown; that proposal is the rendering of the "
        "output, and the assistant does not need to re-present the structured "
        "result in a later message. Asking for one walk-through to fill a gap "
        "is not asking the user to name stages."
    ),
    "Honest about provenance": (
        "Every stage, step, participant, and role in the `_generate_from_notes` "
        "Output comes from its Input notes or the `_gather_notes` Conversation, "
        "or is a clearly stated inference. A role the user's own description "
        "implies (for example 'Writer' for a content writer) is acceptable."
    ),
}


class TestGatherNotesEval(unittest.TestCase):
    """Eval tests for _gather_notes — the exploration phase."""

    def test_warm_open_exploration(self):
        """Assistant starts with an open-ended question, not a structural one."""
        user_persona = (
            "You are a busy professional who attends lots of meetings. You take sketchy "
            "notes in a hurry and need help turning them into proper meeting minutes. "
            "Describe your current process naturally — what you do, what's frustrating."
        )
        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs=dict(max_turns=14),
            user_persona=user_persona,
            judge_rules={
                **PROCESS_DEFINITION_RULES,
                "Warm open": (
                    "The assistant's first response is an open-ended invitation "
                    "about the user's work, not a request to name stages, steps, "
                    "or fields."
                ),
            },
        )
        self.assertIsInstance(result, ProcessDefinition)

    def test_anything_else_pattern(self):
        """Assistant asks 'Anything else?' to encourage elaboration, not a new structural question."""
        user_persona = (
            "You are describing how you write blog posts. You mention the main stages "
            "— research, drafting, editing — but you're a bit vague. Respond naturally "
            "but briefly. If asked 'anything else?', elaborate a bit."
        )
        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs=dict(max_turns=14),
            user_persona=user_persona,
            judge_rules={
                **PROCESS_DEFINITION_RULES,
                "Anything else?": (
                    "In the `_gather_notes` Conversation, after the user's "
                    "first description, the assistant invites elaboration with "
                    "one open invitation such as 'Anything else?', not a new "
                    "structural question."
                ),
            },
        )
        self.assertIsInstance(result, ProcessDefinition)

    def test_no_premature_structuring(self):
        """Assistant explores openly without jumping to phases/activities mid-exploration."""
        user_persona = (
            "You are describing how you plan team events. You start by talking about "
            "your current ad-hoc approach — emails, spreadsheets, confusion. You haven't "
            "thought about it as a 'process' before."
        )
        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs=dict(max_turns=14),
            user_persona=user_persona,
            judge_rules={
                **PROCESS_DEFINITION_RULES,
                "Open exploration": (
                    "In the `_gather_notes` Conversation, the assistant explores "
                    "the user's work before naming stages or steps. Before the "
                    "breakdown is proposed, it asks how the work actually "
                    "happens; proposing the derived breakdown at the end, for "
                    "confirmation, is the next phase and is not premature "
                    "structuring. A proposal in the first response, or before "
                    "the user has described what they do, is."
                ),
            },
        )
        self.assertIsInstance(result, ProcessDefinition)


class TestGenerateFromNotesEval(unittest.TestCase):
    """Eval tests for _generate_from_notes — synthesis phase."""

    def test_synthesizes_honestly(self):
        """Assistant presents a complete picture based on what the user said, not fabrications."""
        user_persona = (
            "You are a content writer describing your writing process. "
            "You use a six-stage process: topic selection, research, "
            "outlining, drafting, editing, and publishing. Describe these "
            "stages naturally when the assistant asks. If asked about a "
            "specific stage, explain what you do in that stage."
        )
        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs=dict(max_turns=14),
            user_persona=user_persona,
            judge_rules=PROCESS_DEFINITION_RULES,
        )
        self.assertIsInstance(result, ProcessDefinition)

    def test_gap_filling_one_at_a_time(self):
        """When the user hasn't mentioned something, assistant suggests ONE possibility."""
        user_persona = (
            "You described your expense reporting process: collect receipts, fill out a "
            "form, submit for approval. You didn't mention who approves or what happens "
            "after submission. If asked about these gaps, respond briefly."
        )
        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs=dict(max_turns=14),
            user_persona=user_persona,
            judge_rules={
                **PROCESS_DEFINITION_RULES,
                "Suggests one possibility": (
                    "In the `_gather_notes` Conversation, when the user has not "
                    "mentioned a detail the breakdown needs, the assistant "
                    "proposes one concrete possibility and asks them to confirm "
                    "it, rather than asking an open question."
                ),
            },
        )
        self.assertIsInstance(result, ProcessDefinition)
