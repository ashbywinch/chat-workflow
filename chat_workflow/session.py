from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, override

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from .atomic_workflow import AtomicWorkflow
from .models import TurnResult
from .rendering import RenderContext, render_result
from .session_log import SessionLog, record_atomic_workflow
from .turn_budget import TurnBudget

if TYPE_CHECKING:
    from .rendering import Renderable


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
    # The session-wide runaway guard. A child shares its parent's budget; the
    # counter is not reset by entering a step.
    budget: TurnBudget | None = None
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

        The child records its own transcript; the parent keeps the child in
        ``children`` so the tree can be rendered and audited. The brief for
        the child is passed as parameters to its step, not as the parent's
        conversation.
        """
        child = Session(
            io=self._raw_io,
            state=SessionLog(),
            config=self.config,
            budget=self.budget,
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

    def propose(
        self,
        result: Renderable | Sequence[Renderable],
        *,
        changed: bool = False,
        note: str | None = None,
    ) -> None:
        """Render a derived result as the proposal turn, if it has anything to say.

        This is the only path a composite uses to show a draft: the rendering
        is validated before it reaches the transcript, and an empty result
        (a list a step is allowed to leave empty) says nothing. A changed
        proposal asks about the change, not the same question again. ``note``
        says what the proposal leaves out — "everything else stays as we
        agreed" — without becoming a second message.

        The parameter type is the contract: a proposal is a record that
        renders itself, so a non-renderable cannot be passed. Rendering the
        wrong shape is a type error here rather than a ``RenderError`` in the
        middle of a conversation.
        """
        text = render_result(result, RenderContext())
        if not text:
            return
        if changed:
            head = "Here's what I've changed"
            question = "Is that the change you meant?"
        else:
            head = "Here's what I've got so far — tell me what you'd change"
            question = "Does that look right, or should I change something?"
        parts = [f"{head}:\n\n{text}"]
        if note:
            parts.append(note)
        parts.append(question)
        self.say("\n\n".join(parts))

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
        *,
        chat: bool = True,
    ) -> TurnResult[TResult]:
        """Run an atomic workflow to completion.

        Sends the first user input, then loops over user prompts until the
        workflow signals completion (success or failure). A chat-off workflow
        has no user to ask: a ``continue`` is fed back as an in-context
        correction and retried within the turn budget.
        """
        result: TurnResult[TResult] | None = None
        # A direct Session.run gets the step's budget; a wrapped session was
        # already given the top-level step's max_turns before the greeting.
        if self.budget is None:
            self.budget = TurnBudget(workflow.max_turns)
        try:
            if not chat:
                # No user turn to add: the step's docstring is the instruction,
                # and its output schema has no `continue` to ask with. The
                # workflow's own turn budget owns the failure correction.
                result = workflow.process_turn()
            else:
                first = ""
                if _last_role(workflow.messages) == "assistant":
                    # The assistant spoke last (a greeting or a composite echo);
                    # the next turn belongs to the user.
                    first = self._raw_io.prompt("\nYou")
                result = workflow.process_turn(first)
                while not result.is_complete:
                    self._raw_io.echo(f"\nAssistant: {result.message or ''}")
                    result = workflow.process_turn(self._raw_io.prompt("\nYou"))
        finally:
            record_atomic_workflow(self.state, workflow, result=result)

        assert result is not None
        # A success carries the result, not a message. A caller that supplies
        # one through TurnResult still has it spoken; the framework's own
        # success path does not. A one-shot step has no user to show it to.
        if chat and result.result is not None and result.message:
            self.say(result.message)
        return result
