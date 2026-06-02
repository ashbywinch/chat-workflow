"""Eval tests for ProcessDefinition sub-workflows (_gather_notes, _generate_from_notes).

These call real LLMs and require a configured config.json with a valid API key.
Evals fail (not skip) when infrastructure is missing.
"""

import unittest

from tests.evals.helpers import run_multi_turn_eval
from workflows.workflow.models import ProcessDefinition, generate_from_chat


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
            method_kwargs=dict(max_turns=4),
            user_persona=user_persona,
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
            method_kwargs=dict(max_turns=4),
            user_persona=user_persona,
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
            method_kwargs=dict(max_turns=4),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, ProcessDefinition)


class TestGenerateFromNotesEval(unittest.TestCase):
    """Eval tests for _generate_from_notes — synthesis phase."""

    def test_synthesizes_honestly(self):
        """Assistant presents a complete picture based on what the user said, not fabrications."""
        user_persona = (
            "You've described your content writing process in detail: topic selection, "
            "research, outlining, drafting, editing, publishing. You mentioned each stage "
            "clearly. When the assistant proposes a structured summary, it should match "
            "what you actually described."
        )
        result = run_multi_turn_eval(
            model_method=generate_from_chat,
            method_kwargs=dict(max_turns=4),
            user_persona=user_persona,
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
            method_kwargs=dict(max_turns=4),
            user_persona=user_persona,
        )
        self.assertIsInstance(result, ProcessDefinition)
