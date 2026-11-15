from __future__ import annotations

import dataclasses
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
            self._log.messages.append({"role": "assistant", "content": clean})
            self._log.record(
                "framework", clean, kind="composite_echo", audience="user"
            )
        self._io.echo(message)

    @override
    def prompt(self, label: str, **kwargs: Any) -> str:
        self._log.messages.append({"role": "assistant", "content": label})
        self._log.record(
            "assistant", label, kind="composite_prompt", audience="user"
        )
        answer = self._io.prompt(label, **kwargs)
        self._log.messages.append({"role": "user", "content": answer})
        self._log.record(
            "user", answer, kind="composite_prompt", audience="user"
        )
        return answer


@dataclasses.dataclass
class Session:
    """Runtime context passed through a workflow call chain.

    Wraps IO, session state, and config so that atomic and composite
    workflows can interact with the user and record history.
    """

    io: UserIO
    state: SessionLog
    config: Any = None
    _raw_io: UserIO = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw = self.io
        self._raw_io = raw
        self.io = _RecordingIO(raw, self.state)

    @property
    def raw_io(self) -> UserIO:
        """The caller's IO, without transcript recording."""
        return self._raw_io

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
            result = workflow_step.process_turn(first_user_input)
            if not result.is_complete:
                self._raw_io.echo(f"\nAssistant: {result.message}")

            while not result.is_complete:
                if workflow_step.chat:
                    user_input = self._raw_io.prompt("\nYou")
                    result = workflow_step.process_turn(user_input)
                else:
                    # No user to ask. Feed the failure back in context so the
                    # agent can return a complete object or an explicit failure.
                    workflow_step.messages.append({
                        "role": "user",
                        "content": (
                            "No user is available to answer. Return success with the "
                            "completed output, or failure if it cannot be produced."
                        ),
                    })
                    result = workflow_step.process_turn("")
                if not result.is_complete:
                    self._raw_io.echo(f"\nAssistant: {result.message}")

            return result
        finally:
            _record_atomic_workflow_step(self.state, workflow_step)
