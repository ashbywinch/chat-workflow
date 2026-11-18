"""A workflow's tools: executed by the workflow, then answered in its model."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from chat_workflow import AgentIntent, AgentResponse, AtomicWorkflow
from chat_workflow.atomic_workflow_config import AtomicWorkflowConfig


class Result(BaseModel):
    ok: bool = True


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


def _answer(content: str | None, calls: list | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))])


def _call(name: str, arguments: str):
    return SimpleNamespace(id="call-1", function=SimpleNamespace(name=name, arguments=arguments))


def _workflow(handler, *, tool_turns: int = 2) -> AtomicWorkflow:
    return AtomicWorkflow(
        config=AtomicWorkflowConfig(
            system_prompt="You are the genealogist.",
            response_model=AgentResponse[Result],
            max_turns=3,
            tools=TOOLS,
            tool_handlers={"search_items": handler},
            tool_turns=tool_turns,
        )
    )


class TestWorkflowTools(unittest.TestCase):
    def test_the_tool_runs_and_the_final_call_carries_its_result(self) -> None:
        workflow = _workflow(lambda query: f"found:{query}")
        answer = AgentResponse[Result](message="done", intent=AgentIntent.SUCCESS, result=Result())
        with (
            patch(
                "litellm.completion",
                return_value=_answer(None, [_call("search_items", json.dumps({"query": "walter"}))]),
            ),
            patch("chat_workflow.llm_interaction.get_client") as get_client,
        ):
            client = get_client.return_value
            client.chat.completions.create.return_value = answer
            assert workflow._call_llm() is answer
            sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert any(message.get("role") == "tool" and message.get("content") == "found:walter" for message in sent)

    def test_an_undefined_tool_fails_loudly(self) -> None:
        workflow = _workflow(lambda query: "")
        with (
            patch("litellm.completion", return_value=_answer(None, [_call("delete_all", "{}")])),
            self.assertRaises(ValueError),
        ):
            workflow._with_tools()

    def test_the_last_round_asks_for_the_answer(self) -> None:
        workflow = _workflow(lambda query: "", tool_turns=1)
        with patch("litellm.completion", return_value=_answer("answered")) as completion:
            workflow._with_tools()
        assert completion.call_args.kwargs["tool_choice"] == "none"
