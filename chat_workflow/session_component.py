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
from typing import ClassVar, TypeVar

from .conversation_rules import WARM_OPEN
from .decorators import atomic_workflow, composite_workflow
from .models import TextReply
from .session import Session

T = TypeVar("T")

class SessionComponent:
    """Runs one conversation: a greeting, the step, and a closing."""

    _FALLBACK_GREETING: ClassVar[str] = (
        "Hi there — I am here to help you describe what you do so it can be "
        "built into something you can use. What are you working on?"
    )
    _FALLBACK_CLOSING: ClassVar[str] = "I have what I need."

    @atomic_workflow(
        chat=False,
        single_turn=True,
        conversation_validation_rules=[WARM_OPEN],
    )
    # lucidlint: ignore detached-method the step decorator injects the session; the method stays on the component
    def greet(self, *, session: Session, max_turns: int = 1) -> TextReply:
        """Open the conversation.

        Give one short greeting, say in one sentence what you are here to help
        with — describing what they do so it can be built into something they
        can use — and ask what they would like help with. They may describe a
        personal routine, not only a job. Keep it warm and plain:
        no list of questions, no form, no jargon. Do not ask the user to
        introduce themselves.

        This is a single turn: return success with the greeting, and put the
        same sentence in the result's `text` field. Do not return a continue
        and do not use a placeholder.
        """
        ...  # type: ignore[reportReturnType]  # the step decorator supplies the body

    @atomic_workflow(chat=False, single_turn=True)
    # lucidlint: ignore detached-method the step decorator injects the session; the method stays on the component
    def sign_off(self, *, session: Session, max_turns: int = 1) -> TextReply:
        """Close the conversation.

        The conversation already happened; you are only closing it, and its
        transcript is not shown to you. Say in one short sentence that you
        have what you need — for example "I have what I need." Do not ask a
        question, offer another step, describe what the conversation produced,
        or say there was no conversation.

        This is a single turn: return success with the closing, and put the
        same sentence in the result's `text` field.
        """
        ...  # type: ignore[reportReturnType]  # the step decorator supplies the body

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

        The task at hand asks for something specific: do that task's work, not
        the next one's. When the user agrees with a draft you presented, finish
        in the same response if it is unchanged; a changed draft is a new
        proposal and is confirmed once. If their last message asks you something, answer it
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
                greeting = self._spoken(
                    self.greet,
                    fallback=self._FALLBACK_GREETING,
                    prompt="Start the conversation.",
                    session=session,
                )
                session.say(greeting or self._FALLBACK_GREETING)
            result = step(session=session)
            session.state.final_result = result
            closing = self._spoken(
                self.sign_off,
                fallback=self._FALLBACK_CLOSING,
                prompt=(
                    "The user has finished. Say in one sentence that you "
                    "have what they need."
                ),
                session=session,
            )
            if not closing or "?" in closing:
                closing = self._FALLBACK_CLOSING
            session.say(closing)
            return result
        finally:
            session.active = False

    @staticmethod
    def _spoken(
        step: Callable[..., TextReply],
        *,
        fallback: str,
        prompt: str,
        session: Session,
    ) -> str:
        """The cosmetic open or close; its failure must not fail the workflow."""
        try:
            return step(session=session, _first_user_input=prompt).text.strip()
        # lucidlint: ignore broad-except the open/close is cosmetic: a failure falls back to plain text
        except Exception:
            return fallback
