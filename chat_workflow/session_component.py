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


def _plain_type(type_expr: Any) -> str:
    """Say a type expression in business words."""
    text = str(type_expr).strip()
    optional = text.endswith("| None") or text.startswith("Optional[")
    text = text.replace("| None", "").strip()
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1]
    if text == "str":
        plain = "text"
    elif text in {"int", "float"}:
        plain = "a number"
    elif text == "bool":
        plain = "yes or no"
    elif text.startswith("list[") and text.endswith("]"):
        plain = "a list of " + _plain_type(text[len("list[") : -1])
    elif text.startswith("dict[") or text == "dict":
        plain = "a set of values"
    else:
        spaced = "".join(
            f" {character}" if character.isupper() else character
            for character in text
        )
        plain = spaced.replace("_", " ").strip().casefold()
    return plain + (" (optional)" if optional else "")


def _describe_item(item: Any, hints: dict[str, str] | None = None) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "model_dump"):
        data = item.model_dump()
    elif isinstance(item, dict):
        data = item
    else:
        return ""

    if data.get("phases") and data.get("activities"):
        phases = "; ".join(str(value) for value in data["phases"])
        steps = "; ".join(str(value) for value in data["activities"])
        self_words = {"the user", "user", "me", "i", "myself"}
        people = [
            str(value)
            for value in data.get("participants") or []
            if str(value).strip().casefold() not in self_words
        ]
        owner = data.get("orchestrating_component") or ""
        sentence = f"The work moves through {phases}. The steps are {steps}."
        if people:
            sentence += f" {'; '.join(people)} are involved."
        else:
            sentence += " You are involved."
        if owner.strip().casefold() in self_words:
            sentence += " You keep it moving."
        elif owner:
            sentence += f" {owner} keeps it moving."
        return sentence

    if "fields" in data and "what_good_looks_like" in data:
        field_text = "; ".join(
            f"{field.get('name')} is {field.get('domain_description')}"
            for field in data["fields"]
            if isinstance(field, dict)
        )
        quality = "; ".join(str(value) for value in data["what_good_looks_like"])
        name = data.get("name") or "The thing"
        description = data.get("description") or ""
        return (
            f"{name}: {description} You capture {field_text}. What makes it good: "
            f"{quality}. The role is {data.get('expert_role') or 'not stated'}."
        )

    if "fields" in data and "model_validators" in data:
        field_text = "; ".join(
            f"{field.get('name')} is "
            f"{(hints or {}).get(str(field.get('name')), _plain_type(field.get('type_expr')))}"
            for field in data["fields"]
            if isinstance(field, dict)
        )
        rules = "; ".join(
            str(validator.get("rule"))
            for validator in data["model_validators"]
            if isinstance(validator, dict)
        )
        return f"The shape is {field_text}. The rules are {rules}."

    if "must_prioritize" in data:
        return (
            "I'll always cover "
            + "; ".join(str(value) for value in data["must_prioritize"])
            + ". I'll suggest "
            + "; ".join(str(value) for value in data.get("auto_suggest") or [])
            + ". I'll watch for "
            + "; ".join(str(value) for value in data.get("user_pain_points") or [])
            + "."
        )

    if "source" in data:
        sentence = f"You start with {data['source']}"
        if data.get("format"):
            sentence += f", which is {data['format']}"
        if data.get("dependencies"):
            sentence += f"; you need {'; '.join(data['dependencies'])} first"
        if data.get("validation_criteria"):
            sentence += f". You know it is ready when {data['validation_criteria']}"
        return sentence + "."

    if "consumer" in data:
        name = data.get("name") or "It"
        sentence = f"{name} is {data.get('description') or 'the thing you make'}."
        if data.get("consumer"):
            sentence += f" It is for {data['consumer']}."
        if data.get("format"):
            sentence += f" It is a {data['format']}."
        if data.get("success_criteria"):
            sentence += f" You know it is ready when {data['success_criteria']}."
        if data.get("integration_points"):
            sentence += f" After that, {data['integration_points']}."
        if data.get("storage_requirements"):
            sentence += f" It is kept {data['storage_requirements']}."
        return sentence

    if "scope_description" in data:
        sentence = f"{data.get('name') or 'This part'}: {data.get('purpose') or ''}"
        if data.get("scope_description"):
            sentence += f" {data['scope_description']}"
        if data.get("owner_role"):
            sentence += f" Owned by {data['owner_role']}."
        return sentence

    heading = data.get("name") or data.get("text") or ""
    parts: list[str] = []
    for key, label in _DISPLAY_LABELS.items():
        value = data.get(key)
        if value in (None, "", []):
            continue
        if key == heading:
            continue
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(entry) for entry in value)
        parts.append(f"{label}: {value}")
    if heading and parts:
        return f"{heading}: " + "; ".join(parts)
    return "; ".join(parts) if parts else ""


_SUGGESTION_MARKERS = (
    "you tell me",
    "what do you think",
    "what works best",
    "suggest",
    "should i",
    "recommend",
    "a rule",
    "checklist",
    "minimum number",
    "how do i know",
    "when is it ready",
    "complete enough",
)


def _asks_for_a_suggestion(question: str) -> bool:
    text = question.casefold()
    return any(marker in text for marker in _SUGGESTION_MARKERS)


def _suggest_from_item(item: Any) -> str:
    data = item.model_dump() if hasattr(item, "model_dump") else item
    if not isinstance(data, dict):
        return ""
    if data.get("source"):
        form = str(data.get("format") or "a simple list")
        structure_words = (
            "bullet",
            "list",
            "table",
            "heading",
            "section",
            "template",
            "timestamp",
            "line",
            "column",
            "one per",
        )
        if not any(word in form.casefold() for word in structure_words):
            form += ", written as short lines, one per point, in the order they come up"
        sentence = f"Here's what I'd suggest: keep {data['source']} as {form}"
        if data.get("dependencies"):
            sentence += f", and have {'; '.join(data['dependencies'])} ready first"
        sentence += "."
        if data.get("validation_criteria"):
            sentence += f" You know it's ready when {data['validation_criteria']}."
        return sentence
    if data.get("consumer"):
        sentence = f"Here's what I'd suggest: {data.get('name') or 'it'} as {data.get('format', 'a simple document')}."
        if data.get("success_criteria"):
            sentence += f" It is done when {data['success_criteria']}."
        return sentence
    return ""


def _describe_result(
    result: Any, *, question: str = "", hints: dict[str, str] | None = None
) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if _asks_for_a_suggestion(question):
        candidates = result if isinstance(result, (list, tuple)) else [result]
        for item in candidates:
            suggestion = _suggest_from_item(item)
            if suggestion:
                return suggestion
    if isinstance(result, (list, tuple)):
        body = "\n".join(
            line for line in (_describe_item(item, hints) for item in result) if line
        )
    else:
        body = _describe_item(result, hints)
    return f"Thanks — here's what I'd propose:\n{body}" if body else ""


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
        with — describing what they do so it can be built into something they
        can use — and ask what they would like help with. They may describe a
        personal routine, not only a job. Keep it warm and plain:
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

    def present(
        self,
        result: Any,
        *,
        question: str = "",
        hints: dict[str, str] | None = None,
    ) -> str:
        """Describe the result in the user's words, without a model call.

        The result's own fields are already in the user's words. A deterministic
        summary is complete and cannot be cut off mid-sentence the way a model
        completion can. When the user's last message asked for a suggestion,
        the reply is the suggestion itself, not the list again.
        """
        return _describe_result(result, question=question, hints=hints)

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
            session.presented = False
            result = step(session=session)
            session.state.final_result = result
            # A step that answered without ever speaking left the user with
            # nothing to react to. The wrapper owns the presentation turn, so
            # the result is described in the user's words before the close.
            # A composite's own presentation counts only while it is still
            # the last thing said. If the user replied and the step returned
            # without speaking, the result is the answer to that reply and the
            # wrapper must say it — unless it is the same result the user has
            # already seen, which would be repetition.
            question = _last_user_text(session.state.messages)
            # A composite already showed its derived result. Re-stating it is
            # repetition; only a direct request for a suggestion earns
            # another turn.
            already_presented = session.presented and not _asks_for_a_suggestion(
                question
            )
            if (
                result is not None
                and not already_presented
                and _needs_presentation(session.state.messages, before)
            ):
                presented = self.present(
                    result,
                    question=question,
                    hints=session.presented_hints,
                )
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


def _last_user_text(messages: list[Any]) -> str:
    """The last turn a person typed, ignoring framework prompts."""
    for kind, text in reversed(entries(messages)):
        if kind == "user":
            return text
    return ""


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
