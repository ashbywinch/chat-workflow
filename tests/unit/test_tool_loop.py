"""The tool loop: tools execute, then the answer is validated."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from pydantic import BaseModel, Field

from chat_workflow.tool_loop import ToolLoopError, run_with_tools


class Answer(BaseModel):
    confidence: str = Field(description="the reviewer's certainty")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_items",
            "description": "search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
]


def _call(name: str, arguments: str):
    return SimpleNamespace(id="call-1", function=SimpleNamespace(name=name, arguments=arguments))


def _response(content: str | None, calls: list | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))])


class TestToolLoop(unittest.TestCase):
    def test_the_tool_runs_and_its_result_reaches_the_model(self) -> None:
        seen: list[str] = []
        choices: list[object] = []
        responses = [
            _response(None, [_call("search_items", json.dumps({"query": "walter"}))]),
            _response('{"confidence": "think_so"}'),
        ]

        def completion_fn(**kwargs):
            seen[:] = [message["role"] for message in kwargs["messages"]]
            choices.append(kwargs.get("tool_choice"))
            return responses.pop(0)

        result = run_with_tools(
            model="test",
            system="You are the genealogist.",
            user="Walter was Pearl's brother.",
            tools=TOOLS,
            handlers={"search_items": lambda query: f"found:{query}"},
            result_model=Answer,
            max_turns=2,
            completion_fn=completion_fn,
        )
        assert result.confidence == "think_so"
        assert seen == ["system", "user", "assistant", "tool"], seen
        assert choices == ["auto", "none"], "the final turn must ask for the answer, not another dig"

    def test_the_schema_is_rendered_by_the_framework(self) -> None:
        captured: dict = {}

        def completion_fn(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _response('{"confidence": "definitely"}')

        run_with_tools(
            model="test",
            system="You are the genealogist.",
            user="A claim.",
            tools=TOOLS,
            handlers={"search_items": lambda query: ""},
            result_model=Answer,
            completion_fn=completion_fn,
        )
        system = captured["messages"][0]["content"]
        assert '"confidence"' in system and "the reviewer's certainty" in system

    def test_a_call_to_an_undefined_tool_fails_loudly(self) -> None:
        def completion_fn(**kwargs):
            return _response(None, [_call("delete_everything", "{}")])

        with self.assertRaises(ToolLoopError):
            run_with_tools(
                model="test",
                system="s",
                user="u",
                tools=TOOLS,
                handlers={"search_items": lambda query: ""},
                result_model=Answer,
                completion_fn=completion_fn,
            )

    def test_a_handler_without_a_tool_definition_fails_loudly(self) -> None:
        with self.assertRaises(ToolLoopError):
            run_with_tools(
                model="test",
                system="s",
                user="u",
                tools=TOOLS,
                handlers={"search_items": lambda query: "", "ghost": lambda: ""},
                result_model=Answer,
                completion_fn=lambda **k: _response("{}"),
            )

    def test_no_conclusion_within_the_budget_fails_loudly(self) -> None:
        def completion_fn(**kwargs):
            return _response(None, [_call("search_items", json.dumps({"query": "x"}))])

        with self.assertRaises(ToolLoopError):
            run_with_tools(
                model="test",
                system="s",
                user="u",
                tools=TOOLS,
                handlers={"search_items": lambda query: ""},
                result_model=Answer,
                max_turns=2,
                completion_fn=completion_fn,
            )
