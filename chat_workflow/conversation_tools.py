import dataclasses
from typing import Any

from chat_workflow.conversation_log import ConversationLog, _record_orchestrator
from chat_workflow.models import ConversationResult
from chat_workflow.protocols import ConversationIO, ConversationOrchestratorLike


@dataclasses.dataclass
class ConversationTools:
    """Provides chat interaction with IO and state tracking.

    Wraps an :class:`ConversationIO` instance and a
    :class:`ConversationFlowState` to offer a simple ``chat`` method
    that drives a multi-turn conversation with the user.
    """

    io: ConversationIO
    state: ConversationLog
    config: Any = None

    def chat[TResult](
        self,
        orchestrator: ConversationOrchestratorLike[TResult],
        first_user_input: str,
    ) -> ConversationResult[TResult]:
        """Run a multi-turn conversation to completion.

        Sends the first user input, then loops over user prompts
        until the orchestrator signals completion.

        Args:
            orchestrator: The orchestrator driving the conversation.
            first_user_input: The initial user message to process.

        Returns:
            The final :class:`ConversationResult` produced by the
            orchestrator.
        """
        try:
            result = orchestrator.process_turn(first_user_input)
            self.io.echo(f"\nAssistant: {result.message}")

            while not result.is_complete:
                user_input = self.io.prompt("\nYou")
                result = orchestrator.process_turn(user_input)
                self.io.echo(f"\nAssistant: {result.message}")

            return result
        finally:
            _record_orchestrator(self.state, orchestrator)
