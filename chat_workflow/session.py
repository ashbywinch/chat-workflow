from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterator
from typing import Any, override

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from .atomic_workflow import AtomicWorkflow
from .models import TurnResult
from .session_log import SessionLog, _record_atomic_workflow


class UserIO:
    """Interface for user interaction (input/output).

    The :class:`Session` uses this to talk to the user.
    """

    def echo(self, message: str) -> None: ...
    def prompt(self, label: str) -> str: ...


class _RecordingIO(UserIO):
    """Record composite-workflow IO in the session transcript.

    Atomic workflows record their own transcript in
    :class:`~chat_workflow.conversation.Conversation`. Composite workflows call
    ``session.io`` directly; without this wrapper those echoes and prompts
    never reach the transcript the user and the judge read.
    """

    def __init__(self, io: UserIO, log: SessionLog) -> None:
        self._io = io
        self._log = log

    @override
    def echo(self, message: str) -> None:
        clean = message.removeprefix("\nAssistant: ").strip()
        if clean:
            self._log.messages.append(ModelResponse(parts=[TextPart(clean)]))
        self._io.echo(message)

    @override
    def prompt(self, label: str) -> str:
        self._log.messages.append(ModelResponse(parts=[TextPart(label)]))
        answer = self._io.prompt(label)
        self._log.messages.append(ModelRequest.user_text_prompt(answer))
        return answer


def _last_role(messages: list[ModelMessage]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, ModelResponse):
        return "assistant"
    if isinstance(last, ModelRequest):
        return "user"
    return ""


@dataclasses.dataclass
class Session:
    """Runtime context passed through a workflow call chain.

    A session is one conversation: the user-visible conversation at the top
    level, or one meeting spawned from inside it. The same class is used at
    every depth. ``state`` holds this session's own transcript; a child does
    not inherit the parent's conversation, and the parent records the child so
    the tree can be rendered and audited.
    """

    io: UserIO
    state: SessionLog
    config: Any = None
    parent: Session | None = None
    children: list[Session] = dataclasses.field(default_factory=list)
    # The overall how for this session. ``SessionComponent.converse`` fills it
    # from its docstring when the session has none.
    instructions: str | None = None
    # The one flag: the wrapper sets it while this session is being run. The
    # greeting condition is ``not state.messages``, not a second flag.
    active: bool = False
    # True once a composite has shown its derived result in the user's words.
    # The wrapper then does not describe the same result a second time (and
    # with the wrong vocabulary).
    presented: bool = False
    presented_at: int = 0
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str | None = None
    _raw_io: UserIO = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw = self.io
        self._raw_io = raw
        self.io = _RecordingIO(raw, self.state)

    @property
    def raw_io(self) -> UserIO:
        """The caller's IO, without transcript recording."""
        return self._raw_io

    def spawn(
        self,
        *,
        title: str | None = None,
        instructions: str | None = None,
    ) -> Session:
        """Create a child session: same user, same provider, clean context.

        The child records its own transcript; the parent keeps the child in
        ``children`` so the tree can be rendered and audited. The brief for
        the child is passed as parameters to its step, not as the parent's
        conversation.
        """
        child = Session(
            io=self._raw_io,
            state=SessionLog(),
            config=self.config,
            parent=self,
            instructions=instructions,
            title=title,
        )
        self.children.append(child)
        return child

    def walk(self) -> Iterator[Session]:
        """This session and every descendant, in spawn order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def child(self, session_id: str) -> Session | None:
        """Find a descendant session by id."""
        return next((item for item in self.walk() if item.id == session_id), None)

    def present(self, text: str) -> None:
        """Speak a derived result and record that it has been shown."""
        self.presented = True
        self.say(text)
        self.presented_at = len(self.state.messages)

    def say(self, text: str) -> None:
        """Speak an assistant turn that the user sees and the transcript keeps."""
        clean = text.removeprefix("\nAssistant: ").strip()
        if not clean:
            return
        self.state.messages.append(ModelResponse(parts=[TextPart(clean)]))
        self._raw_io.echo(f"\nAssistant: {clean}")

    def run[TResult](
        self,
        workflow: AtomicWorkflow[TResult],
        first_user_input: str,
        *,
        chat: bool = True,
    ) -> TurnResult[TResult]:
        """Run an atomic workflow to completion.

        Sends the first user input, then loops over user prompts until the
        workflow signals completion (success or failure). A chat-off workflow
        has no user to ask: a ``continue`` is fed back as an in-context
        correction and retried within the turn budget.
        """
        try:
            first = first_user_input
            if not first and not chat:
                # A one-shot step has no user to open the conversation. The
                # opening prompt would make it greet instead of producing the
                # result, so state the job instead.
                first = (
                    "No user is available. Produce the result now from the "
                    "instructions and the inputs."
                )
            elif not first and chat and _last_role(workflow.messages) == "assistant":
                # The assistant spoke last (a greeting or a composite echo);
                # the next turn belongs to the user.
                first = self._raw_io.prompt("\nYou")
            result = workflow.process_turn(first)
            if not result.is_complete:
                self._raw_io.echo(f"\nAssistant: {result.message}")

            while not result.is_complete:
                if chat:
                    user_input = self._raw_io.prompt("\nYou")
                    result = workflow.process_turn(user_input)
                else:
                    # No user to ask. The framework turn is a correction, not
                    # an authored prompt: it may say what the step must return.
                    result = workflow.process_turn(
                        "No user is available to answer. Return success with the "
                        "completed result, or failure if it cannot be produced."
                    )
                if not result.is_complete:
                    self._raw_io.echo(f"\nAssistant: {result.message}")

            return result
        finally:
            _record_atomic_workflow(self.state, workflow)
