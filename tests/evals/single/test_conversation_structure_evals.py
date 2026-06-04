"""Eval tests for conversation structure: warm opening, domain knowledge proposals, and adaptive questioning.

These tests capture failure modes from the transcript where the agent:
- Opened with an abrupt interrogation (no warmup or context-setting)
- Failed to propose meeting minutes structure from domain knowledge
- Repeated the same question when user asked for suggestions
- Asked nonsensical field-mapped questions (e.g., "how do you know when your notes are complete?")

These evals should FAIL with current prompts and PASS after prompt improvements.
"""

from __future__ import annotations

import unittest

from tests.conftest import timeout
from tests.evals.helpers import run_multi_turn_eval
from workflows.workflow.models import Deliverable, Resource

SKEPTICAL_FIRST_TIME_PERSONA = (
    "You are trying out this tool for the first time because you heard it can "
    "help you document your processes. You are a bit skeptical — you have tried "
    "similar things before and they were always more hassle than they were worth. "
    "You need the assistant to win you over.\n\n"
    "You are an expert in your own work — you write up meeting minutes every week. "
    "You know exactly what goes into them. But you have no idea what 'outputs', "
    "'processes', or 'workflows' mean — those are the tool's job to figure out.\n\n"
    "You expect the assistant to explain what it is doing and why, not just start "
    "asking questions. If the assistant opens without any introduction, you will "
    "be put off and hesitate to engage."
)

DOMAIN_EXPECTATION_PERSONA = (
    "You write up meeting minutes regularly for your team. Meeting minutes are a "
    "completely standard type of document — you are honestly a bit surprised when "
    "the assistant acts like it has never seen them before.\n\n"
    "You know what goes into good minutes: the date, who attended, what was "
    "discussed, decisions made, action items with owners and due dates, and "
    "the date of the next meeting if there is one.\n\n"
    "You are happy to describe your specific needs, but you expect the assistant "
    "to start from a foundation of understanding instead of asking you to explain "
    "from scratch what meeting minutes even are."
)

PARTICIPATORY_USER_PERSONA = (
    "You are someone who wants help structuring your meeting minutes. You have "
    "described your basic situation: you take notes during meetings and need "
    "to turn them into proper minutes.\n\n"
    "You are happy to answer questions, but you also want the assistant to "
    "contribute their own ideas. If the assistant keeps asking open-ended "
    "questions like 'what information should this include?' without ever "
    "suggesting anything themselves, you push back: 'what do you think?' "
    "or 'can you suggest something?' or 'you tell me.'\n\n"
    "If the assistant responds to your pushback by repeating the same question "
    "again or rephrasing it slightly, you get frustrated. You expect them "
    "to actually offer ideas when asked."
)


WARM_OPEN_RULES: dict[str, str] = {
    "Warm opening": (
        "The assistant's first response includes a greeting, introduction, "
        "or context-setting before asking any substantive questions about "
        "the user's work. Starting with 'Hi there! I am here to help you...' "
        "or similar is good. Starting with an abrupt question like 'Let us "
        "start by defining the first output you produce' without any preamble "
        "is not."
    ),
    "Explains purpose": (
        "Early in the conversation (within the first 1-2 turns), the assistant "
        "explains what it is trying to do and why. The user should understand "
        "the purpose of the conversation and what will happen — not be left "
        "wondering why they are being asked random questions about their work."
    ),
}

DOMAIN_PROPOSAL_RULES: dict[str, str] = {
    "Proposes from knowledge": (
        "When the user mentions a common, well-known document type (like "
        "meeting minutes, invoices, reports, blog posts), the assistant "
        "proposes a structure based on what that document typically contains, "
        "rather than asking the user to define it from scratch.\n\n"
        "GOOD: 'Great, meeting minutes! Those typically capture the date, "
        "attendees, discussion points, decisions, and action items. Does "
        "that align with what you need?'\n\n"
        "BAD: 'What information should be included in the minutes?' — this "
        "asks the user to define something that is already commonly known."
    ),
    "No forced field-mapping": (
        "The assistant does not mechanically ask about every data model field "
        "when it does not make sense for the user's domain. If a question "
        "would not apply to the specific output or input the user is describing, "
        "the assistant recognizes this and moves on, rather than pressing or "
        "rephrasing.\n\n"
        "For example, asking 'how do you know when your notes are complete?' "
        "does not make sense for meeting notes — they are notes, they are "
        "whatever the person wrote. If the user pushes back on such a question, "
        "the assistant should acknowledge it does not apply and move on."
    ),
}

ADAPTIVE_CONVERSATION_RULES: dict[str, str] = {
    "Responds to suggestions": (
        "When the user says 'what do you think?' or 'suggest something' or "
        "'you tell me', the assistant provides concrete suggestions and ideas "
        "rather than repeating or rephrasing its question. The assistant "
        "pivots from asking mode to proposing mode."
    ),
    "No question after pushback": (
        "The assistant does not ask the same substantive question again after "
        "the user has pushed back on it, said 'I don't know', or deflected "
        "with 'what do you think?'. Instead, the assistant changes approach — "
        "it offers ideas, asks about a different topic, or explains why the "
        "question matters."
    ),
    "Detects confusion signals": (
        "When the user expresses confusion ('I don't understand', 'what does "
        "that mean?', 'that does not make sense'), the assistant changes "
        "strategy rather than just re-explaining with different words. "
        "It steps back and tries a fundamentally different approach — such "
        "as explaining the reason for the question, offering an example, "
        "or moving to a different topic."
    ),
}


class TestWarmOpenEval(unittest.TestCase):
    """Assistant opens with greeting and context, not an abrupt interrogation."""

    @timeout(120)
    def test_output_warm_open(self):
        """Deliverable.generate_from_chat opens warmly, not with an abrupt question."""
        result = run_multi_turn_eval(
            model_method=Deliverable.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=SKEPTICAL_FIRST_TIME_PERSONA,
            judge_rules=WARM_OPEN_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)


class TestDomainProposalEval(unittest.TestCase):
    """Assistant proposes structure from domain knowledge instead of asking user to define basics."""

    @timeout(120)
    def test_output_proposes_structure(self):
        """Deliverable proposes meeting minutes structure from domain knowledge."""
        result = run_multi_turn_eval(
            model_method=Deliverable.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=DOMAIN_EXPECTATION_PERSONA,
            judge_rules=DOMAIN_PROPOSAL_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)

    @timeout(120)
    def test_resource_no_forced_field_mapping(self):
        """Resource does not force nonsensical field-mapped questions."""
        result = run_multi_turn_eval(
            model_method=Resource.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=DOMAIN_EXPECTATION_PERSONA,
            judge_rules=DOMAIN_PROPOSAL_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)


class TestAdaptiveConversationEval(unittest.TestCase):
    """Assistant adapts when user pushes back or expresses confusion."""

    @timeout(120)
    def test_output_adapts_to_suggestions(self):
        """Deliverable pivots to proposing mode when user says 'what do you think?' or 'suggest something'."""
        result = run_multi_turn_eval(
            model_method=Deliverable.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=PARTICIPATORY_USER_PERSONA,
            judge_rules=ADAPTIVE_CONVERSATION_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)

    @timeout(120)
    def test_resource_adapts_to_suggestions(self):
        """Resource pivots to proposing mode when user says 'what do you think?' or 'suggest something'."""
        result = run_multi_turn_eval(
            model_method=Resource.generate_from_chat,
            method_kwargs={"max_turns": 8},
            user_persona=PARTICIPATORY_USER_PERSONA,
            judge_rules=ADAPTIVE_CONVERSATION_RULES,
        )
        if result is not None:
            self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
