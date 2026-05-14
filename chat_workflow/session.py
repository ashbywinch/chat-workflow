from __future__ import annotations

import dataclasses
from typing import Any

from .atomic_workflow import AtomicWorkflow
from .models import TurnResult
from .session_log import SessionLog, _record_atomic_workflow


class UserIO:
    """Interface for user interaction (input/output).

    The :class:`Session` uses this to talk to the user.
    """

    def echo(self, message: str) -> None: ...
    def prompt(self, label: str, **kwargs) -> str: ...


@dataclasses.dataclass
class Session:
    """Runtime context passed through a workflow call chain.

    Wraps IO, session state, and config so that atomic and composite
    workflows can interact with the user and record history.
    """

    io: UserIO
    state: SessionLog
    config: Any = None

    def run[TResult](
        self,
        workflow: AtomicWorkflow[TResult],
        first_user_input: str,
    ) -> TurnResult[TResult]:
        """Run an atomic workflow to completion.

        Sends the first user input, then loops over user prompts
        until the workflow signals completion (success or failure).
        """
        try:
            result = workflow.process_turn(first_user_input)
            self.io.echo(f"\nAssistant: {result.message}")

            while not result.is_complete:
                user_input = self.io.prompt("\nYou")
                result = workflow.process_turn(user_input)
                self.io.echo(f"\nAssistant: {result.message}")

            return result
        finally:
            _record_atomic_workflow(self.state, workflow)