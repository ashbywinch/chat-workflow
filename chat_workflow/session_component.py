"""The session wrapper: greet, run a step, sign off.

``Session`` is state and context; ``SessionComponent`` is behaviour. It is
stateless and can wrap any session at any depth. A receptionist step runs in a
parent session, spawns a child session and calls ``converse`` on it; the same
component runs the top-level conversation.

The wrap is explicit. The decorators never auto-wrap a step, so a nested step
does not greet or sign off unless the caller asks for it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from inspect import cleandoc
from typing import Annotated, Any, TypeVar

from .conversation_rules import WARM_OPEN
from .decorators import atomic_workflow_step, composite_workflow_step
from .session import Session

T = TypeVar("T")

_FALLBACK_CLOSING = "I have what I need."


def _is_structured_text(text: str) -> bool:
    """True when a one-turn text is a JSON object or array, not a sentence."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        return isinstance(json.loads(stripped), (dict, list))
    except (ValueError, TypeError):
        return False


class SessionComponent:
    """Runs one conversation: a greeting, the step, and a closing."""

    @atomic_workflow_step(
        chat=False,
        single_turn=True,
        conversation_validation_rules=[WARM_OPEN],
    )
    def greet(self, *, session: Session, max_turns: int = 1) -> str:
        """Open the conversation.

        Give one short greeting, say in one sentence what you are here to help
        with — describing their work so it can be built into something they can
        use — and ask what they would like help with. Keep it warm and plain:
        no list of questions, no form, no jargon. Do not ask the user to
        introduce themselves.
        """
        ...  # type: ignore[reportReturnType]

    @atomic_workflow_step(chat=False, single_turn=True)
    def sign_off(
        self,
        *,
        session: Session,
        result: Annotated[
            Any,
            "The thing the conversation produced; internal reference only",
        ] = None,
        max_turns: int = 1,
    ) -> str:
        """Close the conversation.

        Say in one short sentence that you have what you need. The result is an
        internal reference: never repeat it, quote it, list its parts, or
        describe any part of it in the closing. Do not ask another question.
        """
        ...  # type: ignore[reportReturnType]

    @composite_workflow_step
    def converse(self, step: Callable[..., T], *, session: Session) -> T:
        """Hold one conversation with the user about their own work.

        Talk about the user's own work in their words: short sentences, their
        everyday language, one question or one proposal at a time. If you need
        a name for the thing, ask what they call it. If the user uses a word,
        use theirs; if they say a word is confusing, use their word for it
        instead. Acknowledge an off-topic remark in one clause without praising
        it, then return to the question; never develop their domain content.
        Describe the picture in their words, not as a list of field names.

        The task at hand asks for something specific. Do that task's work,
        not the next one's. When the user agrees with the picture you
        described, finish in the same response; never re-present an unchanged
        draft or ask for a second approval. If their last message asks you
        something, answer it before finishing. Do not give up on the thing:
        propose it in their own words, and stop only if they reject every
        concrete proposal.
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
                session.say(self.greet(session=session))
            result = step(session=session)
            session.state.final_result = result
            closing = self.sign_off(result=result, session=session)
            if _is_structured_text(closing):
                # A closing is one sentence. If the model copies the result
                # instead, say the plain closing rather than leak field names.
                closing = _FALLBACK_CLOSING
            session.say(closing)
            return result
        finally:
            session.active = False
