"""Run a caller's own tools, then return a structured answer.

The framework's client can hand a model tool definitions, but nothing executes
them: instructor treats `response_model` as the tool that *delivers* the answer,
so a caller's tools are ignored (measured: the model answers without calling,
`tool_choice="required"` is ignored, and a call without `response_model` is
refused). This module is the missing half — the loop — so a step may dig with
real tools and still end in a validated result.

The schema of the result is rendered here, by the framework, into the system
prompt: a caller then describes WHAT to decide and never how the answer is
shaped, which is the duplication this module removes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from litellm import completion
from pydantic import BaseModel, ValidationError

MAX_TURNS = 8


class ToolLoopError(RuntimeError):
    """The model neither called a tool nor produced a valid result within the
    turn budget — the loop reports it rather than inventing an answer."""


def schema_section(model: type[BaseModel]) -> str:
    """The result's JSON schema, as the prompt text that defines it."""
    return json.dumps(model.model_json_schema(), indent=2)


def run_with_tools(
    *,
    model: str,
    system: str,
    user: str,
    tools: Sequence[Mapping[str, Any]],
    handlers: Mapping[str, Callable[..., Any]],
    result_model: type[BaseModel],
    max_turns: int = MAX_TURNS,
    completion_fn: Callable[..., Any] = completion,
    **completion_kwargs: Any,
) -> BaseModel:
    """Call the model until it answers, executing the caller's tools on the way.

    A turn either calls tools — each is executed and its result returned to the
    model — or states the result, which is validated into *result_model*. The
    system prompt carries the result's schema; the caller does not restate it.
    """
    unknown = {name for name in handlers if not any(_tool_name(tool) == name for tool in tools)}
    if unknown:
        raise ToolLoopError(f"handlers without a tool definition: {sorted(unknown)}")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{system}\n\nReturn this JSON shape:\n{schema_section(result_model)}"},
        {"role": "user", "content": user},
    ]
    for turn in range(max_turns):
        # The budget's final turn asks for the ANSWER, not another dig: a model
        # given a schema and tools will otherwise keep searching past the point
        # where it could answer (measured: four searches in a row, content
        # beginning to answer each time and never stated).
        final = turn == max_turns - 1
        response = completion_fn(
            model=model,
            messages=messages,
            tools=list(tools),
            tool_choice="none" if final else "auto",
            **completion_kwargs,
        )
        message = response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])
        if not calls:
            try:
                return result_model.model_validate_json(message.content or "{}")
            except ValidationError as exc:
                raise ToolLoopError(f"the answer does not fit {result_model.__name__}: {exc}") from exc
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_call_dict(call) for call in calls],
            }
        )
        for call in calls:
            name = call.function.name
            if name not in handlers:
                raise ToolLoopError(f"the model called an undefined tool: {name!r}")
            arguments = json.loads(call.function.arguments or "{}")
            result = handlers[name](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result if isinstance(result, str) else json.dumps(result),
                }
            )
    raise ToolLoopError(f"the model did not conclude within {max_turns} turns")


def _tool_name(tool: Mapping[str, Any]) -> str | None:
    function = tool.get("function")
    return function.get("name") if isinstance(function, Mapping) else None


def _call_dict(call: Any) -> dict[str, Any]:
    """One tool call as the message the API expects back."""
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }
