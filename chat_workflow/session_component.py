"""The session wrapper: greet, run a step, sign off.

``Session`` is state and context; ``SessionComponent`` is behaviour. It is
stateless and can wrap any session at any depth. A receptionist step runs in a
parent session, spawns a child session and calls ``converse`` on it; the same
component runs the top-level conversation.

The wrap is explicit. The decorators never auto-wrap a step, so a nested step
does not greet or sign off unless the caller asks for it.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import cleandoc
from typing import Any, TypeVar

from .conversation_rules import WARM_OPEN
from .decorators import atomic_workflow, composite_workflow
from .models import TextReply
from .session import Session
from .transcript import entries

T = TypeVar("T")

_FALLBACK_GREETING = "Hi there — what are you working on?"
_FALLBACK_CLOSING = "I have what I need."


_DISPLAY_LABELS: dict[str, str] = {
    "name": "name",
    "purpose": "what it does",
    "description": "what it is",
    "consumer": "who it is for",
    "format": "format",
    "success_criteria": "the check it must pass",
    "integration_points": "what happens next",
    "storage_requirements": "where it goes",
    "source": "where it comes from",
    "dependencies": "what it needs first",
    "scope_description": "scope",
    "owner_role": "owner",
    "expert_role": "expertise",
    "must_prioritize": "always cover",
    "auto_suggest": "suggest",
    "user_pain_points": "watch for",
    "phases": "stages",
    "activities": "steps",
    "orchestrating_component": "in charge",
    "participants": "involved",
}


def _describe_item(item: Any) -> str:
    if isinstance(item, str):
        return f"- {item}"
    if hasattr(item, "model_dump"):
        data = item.model_dump()
    elif isinstance(item, dict):
        data = item
    else:
        return ""
    if data.get("phases") and data.get("activities"):
        phases = "; ".join(str(value) for value in data["phases"])
        steps = "; ".join(str(value) for value in data["activities"])
        people = "; ".join(str(value) for value in data.get("participants") or [])
        owner = data.get("orchestrating_component") or ""
        sentence = f"The work moves through {phases}. The steps are {steps}."
        if people:
            sentence += f" {people} are involved."
        if owner:
            sentence += f" {owner} keeps it moving."
        return sentence
    heading = data.get("name") or data.get("text") or ""
    # A component was already shown in full by the composite's own
    # presentation; the closing recap names it and its owner, not its scope
    # again.
    keys = ("owner_role",) if "scope_description" in data else _DISPLAY_LABELS
    parts: list[str] = []
    for key in keys:
        label = _DISPLAY_LABELS[key]
        value = data.get(key)
        if value in (None, "", []):
            continue
        if key == heading:
            continue
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(entry) for entry in value)
        parts.append(f"{label}: {value}")
    if heading and parts:
        return f"- {heading} — " + "; ".join(parts)
    if heading:
        return f"- {heading}"
    return "- " + "; ".join(parts) if parts else ""


def _describe_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        body = "\n".join(
            line for line in (_describe_item(item) for item in result) if line
        )
    else:
        body = _describe_item(result)
    return f"Here's what I'd propose:\n{body}" if body else ""


class SessionComponent:
    """Runs one conversation: a greeting, the step, and a closing."""

    @atomic_workflow(
        chat=False,
        single_turn=True,
        conversation_validation_rules=[WARM_OPEN],
    )
    def greet(self, *, session: Session, max_turns: int = 1) -> TextReply:
        """Open the conversation.

        Give one short greeting, say in one sentence what you are here to help
        with — describing their work so it can be built into something they can
        use — and ask what they would like help with. Keep it warm and plain:
        no list of questions, no form, no jargon. Do not ask the user to
        introduce themselves.
        """
        ...  # type: ignore[reportReturnType]

    @atomic_workflow(chat=False, single_turn=True)
    def sign_off(self, *, session: Session, max_turns: int = 1) -> TextReply:
        """Close the conversation.

        The conversation already happened; you are only closing it, and its
        transcript is not shown to you. Say in one short sentence that you
        have what the user needs to give. Do not ask a question, offer another
        step, describe what the conversation produced, or say there was no
        conversation.
        """
        ...  # type: ignore[reportReturnType]

    def present(self, result: Any) -> str:
        """Describe the result in the user's words, without a model call.

        The result's own fields are already in the user's words. A deterministic
        summary is complete and cannot be cut off mid-sentence the way a model
        completion can.
        """
        return _describe_result(result)

    @composite_workflow
    def converse(self, step: Callable[..., T], *, session: Session) -> T:
        """Hold one conversation with the user about their own work.

        Talk about the user's own work in their words: short sentences, their
        everyday language, one question or one proposal at a time. Use the
        user's word for a thing when they give one. When you already know the
        thing from the step's inputs or from their domain, name it yourself and
        let them correct you; never ask the user to name a thing from scratch.
        Acknowledge an off-topic remark in one clause without praising it, then
        return to the question; never develop their domain content. Describe
        the picture in their words, not as a list of field names.

        The step's inputs are established facts. Open the step by proposing
        what you can already see from them — the result, or the part of it they
        cover — and ask for a reaction. Do not ask the user to supply or
        confirm something the inputs already contain. Before the first
        question, say in one sentence what you are doing and why. The task at
        hand asks for something specific: do that task's work, not the next
        one's. When the user agrees with the picture you described, finish in
        the same response; never re-present an unchanged draft or ask for a
        second approval. If their last message asks you something, answer it
        before finishing. When they ask for a suggestion or say "you tell me",
        the reply is the suggestion itself: name the concrete sections,
        fields, or format in that message, and do not answer with a question.
        Do not give up on the thing: propose it in their own words, and stop
        only if they reject every concrete proposal.
        """
        if session.instructions is None:
            session.instructions = cleandoc(self.converse.__doc__ or "")

        # A top-level step that makes no user turns is not a conversation: it
        # has nothing to greet or close. Composite steps default to chat-on.
        if getattr(step, "_chat", True) is False:
            session.active = True
            try:
                result = step(session=session)
                session.state.final_result = result
                return result
            finally:
                session.active = False

        session.active = True
        try:
            if not session.state.messages:
                try:
                    greeting = self.greet(
                        session=session,
                        _first_user_input="Start the conversation.",
                    ).text.strip()
                except Exception:
                    greeting = _FALLBACK_GREETING
                session.say(greeting or _FALLBACK_GREETING)
            before = len(session.state.messages)
            result = step(session=session)
            session.state.final_result = result
            # A step that answered without ever speaking left the user with
            # nothing to react to. The wrapper owns the presentation turn, so
            # the result is described in the user's words before the close.
            if result is not None and _needs_presentation(
                session.state.messages, before
            ):
                presented = self.present(result)
                if presented:
                    session.say(presented)
            try:
                closing = self.sign_off(
                    session=session,
                    _first_user_input=(
                        "The user has finished. Say in one sentence that you "
                        "have what they need."
                    ),
                ).text.strip()
            except Exception:
                # The close is cosmetic. A closing that fails, or that asks a
                # question, must not fail a workflow that already produced its
                # result, so the wrapper owns a plain fallback.
                closing = _FALLBACK_CLOSING
            if not closing or "?" in closing:
                closing = _FALLBACK_CLOSING
            session.say(closing)
            return result
        finally:
            session.active = False


def _needs_presentation(messages: list[Any], before: int) -> bool:
    """True when the step's result has not been said back to the user.

    A step that ends on the user's answer returned its result without speaking
    it; a step that ends on its own assistant turn has already said something
    (a question or a presentation). A step with no conversational turn at all
    answered silently.
    """
    conversational = [
        kind
        for kind, _ in entries(messages[before:])
        if kind in ("user", "assistant")
    ]
    return not conversational or conversational[-1] == "user"
