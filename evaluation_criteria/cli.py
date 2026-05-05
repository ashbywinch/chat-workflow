#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional

import typer

from prompt_core import ConversationFlowState, ConversationIO
from prompt_core.exceptions import (
    APIKeyError,
    ConfigFileError,
    ConfigurationError,
    ConversationFailedError,
    CriteriaValidationError,
    PromptCoreError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
    TurnLimitExceededError,
)
from prompt_core.session_logging import log_session

from .flows import generate_reviewed_criteria
import traceback

app = typer.Typer(help="Generate and work with evaluation criteria using LLMs")


class TyperConversationIO(ConversationIO):
    def echo(self, message: str) -> None:
        typer.echo(message)

    def prompt(self, label: str) -> str:
        return typer.prompt(label)


def handle_error(error: Exception):
    if isinstance(error, ConfigFileError):
        typer.secho(
            f"\n✗ Configuration file error: {error.message}",
            err=True,
            fg=typer.colors.RED,
        )
    elif isinstance(error, ConfigurationError):
        typer.secho(
            f"\n✗ Configuration error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, APIKeyError):
        typer.secho(
            f"\n✗ API key error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, ProviderNotSupportedError):
        typer.secho(
            f"\n✗ Provider error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, ProviderNotFoundError):
        typer.secho(
            f"\n✗ Provider not found: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, TurnLimitExceededError):
        typer.secho(f"\n✗ {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ConversationFailedError):
        typer.secho(
            f"\n✗ Conversation failed: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, CriteriaValidationError):
        typer.secho(
            f"\n✗ Validation error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, PromptCoreError):
        typer.secho(f"\n✗ Error: {error.message}", err=True, fg=typer.colors.RED)
    else:
        typer.secho(
            f"\n✗ Unexpected error: {str(error)[:200]}", err=True, fg=typer.colors.RED
        )
        message = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        typer.secho(f"\n {message}")
    raise typer.Exit(1)


@app.command()
def converse(
    context: str = typer.Option(
        "", "--context", "-c", help="Initial context for the conversation"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save successful criteria to JSON file"
    ),
    max_turns: int = typer.Option(
        10,
        "--max-turns",
        "-t",
        help="Maximum number of conversation turns",
        min=1,
        max=20,
    ),
):
    typer.echo(f"Starting conversation... (Ctrl+C to quit, max {max_turns} turns)")
    io = TyperConversationIO()
    flow_state = ConversationFlowState()

    try:
        if context:
            typer.echo(f"Context: {context}")

        criteria = generate_reviewed_criteria(
            context=context,
            max_turns=max_turns,
            io=io,
            state=flow_state,
        )

        if output:
            with open(output, "w") as f:
                json.dump(criteria.model_dump(), f, indent=2)
            typer.echo(f"\n✓ Saved to {output}")

        success_judgement = typer.confirm(
            "\nWas this experience successful?", default=True
        )

        feedback_text: str | None = None
        if not success_judgement:
            feedback_text = typer.prompt("What went wrong? (optional)", default="")
            if feedback_text == "":
                feedback_text = None

        try:
            log_path = log_session(
                messages=flow_state.messages,
                criteria=criteria.model_dump(),
                success_judgement=success_judgement,
                feedback_text=feedback_text,
                model=flow_state.model,
                turn_count=flow_state.turn_count,
                context=context,
            )
            typer.echo(f"\n📝 Session logged to: {log_path}")
        except Exception as e:
            typer.secho(f"\n⚠️  Failed to log session: {e}", fg=typer.colors.YELLOW)

    except KeyboardInterrupt:
        typer.echo("\n\nConversation cancelled.")
        raise typer.Exit(0)
    except ConversationFailedError as e:
        typer.secho(
            f"\n✗ Conversation failed: {e.message}", err=True, fg=typer.colors.RED
        )
        success_judgement = typer.confirm(
            "\nWas this experience successful?", default=False
        )
        feedback_text: str | None = None
        if not success_judgement:
            feedback_text = typer.prompt("What went wrong? (optional)", default="")
            if feedback_text == "":
                feedback_text = None

        log_path = log_session(
            messages=flow_state.messages,
            criteria=None,
            success_judgement=success_judgement,
            feedback_text=feedback_text,
            model=flow_state.model,
            turn_count=flow_state.turn_count,
            context=context,
        )
        typer.echo(f"\n📝 Session logged to: {log_path}")
        raise typer.Exit(1)
    except TurnLimitExceededError as e:
        typer.secho(f"\n✗ {e.message}", err=True, fg=typer.colors.RED)

        log_path = log_session(
            messages=flow_state.messages,
            criteria=None,
            success_judgement=False,
            feedback_text="Turn limit reached",
            model=flow_state.model,
            turn_count=flow_state.turn_count,
            context=context,
        )
        typer.echo(f"\n📝 Session logged to: {log_path}")
        raise typer.Exit(1)
    except Exception as e:
        handle_error(e)


def main():
    app()


if __name__ == "__main__":
    main()
