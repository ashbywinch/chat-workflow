from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterator
from typing import Any, override

from .atomic_workflow_step import AtomicWorkflowStep
from .models import TurnResult
from .session_log import SessionLog, _record_atomic_workflow_step


class UserIO:
    """Interface for user interaction (input/output).

    The :class:`Session` uses this to talk to the user.
    """

    def echo(self, message: str) -> None: ...
    def prompt(self, label: str, **kwargs) -> str: ...


def _append_message(
    messages: list[dict[str, str]], role: str, content: str
) -> None:
    """Append one conversation turn, merging a repeated role.

    A session records what the user saw, so a framework echo can follow an
    assistant turn with no user turn between them. Providers reject two
    assistant messages in a row; joining them keeps the history valid without
    losing either sentence.
    """
    clean = content.strip()
    if not clean:
        return
    if messages and messages[-1].get("role") == role:
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{clean}"
        return
    messages.append({"role": role, "content": clean})


class _RecordingIO(UserIO):
    """Record composite-workflow IO in the session transcript.

    Atomic workflow steps record their messages through
    :func:`_record_atomic_workflow_step`; composite workflow steps call ``session.io``
    directly. Without this wrapper those prompts and answers never reach the
    transcript the judge reads.
    """

    def __init__(self, io: UserIO, log: SessionLog) -> None:
        self._io = io
        self._log = log

    @override
    def echo(self, message: str) -> None:
        clean = message.removeprefix("\nAssistant: ").strip()
        if clean:
            self._log.record(
                "framework", clean, kind="composite_echo", audience="user"
            )
        self._io.echo(message)

    @override
    def prompt(self, label: str, **kwargs: Any) -> str:
        self._log.record(
            "assistant", label, kind="composite_prompt", audience="user"
        )
        _append_message(self._log.messages, "assistant", label)
        answer = self._io.prompt(label, **kwargs)
        self._log.record(
            "user", answer, kind="composite_prompt", audience="user"
        )
        _append_message(self._log.messages, "user", answer)
        return answer


@dataclasses.dataclass
class Session:
    """Runtime context passed through a workflow call chain.

    A session is one conversation: the user-visible conversation at the top
    level, or one meeting spawned from inside it. The same class is used at
    every depth. ``state`` holds this session's own messages and audit events;
    a child does not inherit the parent's conversation, and the parent records
    the child as an event so the tree can be rendered and audited.
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

        The child records its own messages; the parent records a child-session
        event (id, title, order) so the tree can be rendered and audited. The
        brief for the child is passed as parameters to its step, not as the
        parent's conversation.
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
        self.state.record(
            "framework",
            None,
            kind="child_session",
            audience="internal",
            session=child.id,
            title=title,
            order=len(self.children),
        )
        return child

    def walk(self) -> Iterator[Session]:
        """This session and every descendant, in spawn order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def child(self, session_id: str) -> Session | None:
        """Find a descendant session by id."""
        return next((item for item in self.walk() if item.id == session_id), None)

    def say(self, text: str) -> None:
        """Speak an assistant turn that the user sees and the history keeps."""
        clean = text.removeprefix("\nAssistant: ").strip()
        if not clean:
            return
        _append_message(self.state.messages, "assistant", clean)
        self.state.record("assistant", clean, audience="user")
        self._raw_io.echo(f"\nAssistant: {clean}")

    def run[TResult](
        self,
        workflow_step: AtomicWorkflowStep[TResult],
        first_user_input: str,
    ) -> TurnResult[TResult]:
        """Run an atomic workflow step to completion.

        Sends the first user input, then loops over user prompts
        until the step signals completion (success or failure).
        """
        try:
            first = first_user_input
            if (
                not first
                and workflow_step.chat
                and workflow_step.messages[-1].get("role") == "assistant"
            ):
                # The assistant spoke last (a greeting or a composite echo);
                # the next turn belongs to the user.
                first = self._raw_io.prompt("\nYou")
            result = workflow_step.process_turn(first)
            if not result.is_complete and workflow_step.chat:
                self._raw_io.echo(f"\nAssistant: {result.message}")

            while not result.is_complete:
                if workflow_step.chat:
                    user_input = self._raw_io.prompt("\nYou")
                    result = workflow_step.process_turn(user_input)
                else:
                    # No user to ask. Feed the failure back in context so the
                    # agent can return a complete object or an explicit failure.
                    # This is a repair turn, not an authored prompt: it may say
                    # where the text belongs so a chat-off step can finish.
                    workflow_step.messages.append({
                        "role": "user",
                        "content": (
                            "No user is available to answer. Return success now: "
                            "put the completed result in `result` (not `message`), "
                            "or failure if it cannot be produced."
                        ),
                    })
                    result = workflow_step.process_turn("")
                if not result.is_complete and workflow_step.chat:
                    self._raw_io.echo(f"\nAssistant: {result.message}")

            return result
        finally:
            _record_atomic_workflow_step(self.state, workflow_step)
