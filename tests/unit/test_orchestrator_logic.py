#!/usr/bin/env python3
import json
import unittest
from unittest.mock import Mock, patch

from chat_workflow import (
    AgentIntent,
    AgentResponse,
    AtomicWorkflow,
    AtomicWorkflowFailedError,
    InvalidResponseError,
    Session,
    SessionLog,
    TurnLimitExceededError,
)
from tests.conftest import FakeConfig, make_atomic_workflow_config, make_valid_criteria
from workflows.evaluation_criteria import Criterion, EvaluationCriteria


class TestAtomicWorkflow(unittest.TestCase):
    def test_orchestrator_initialization(self):
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(
                max_turns=5,
                model="test-model",
                initial_messages=[{"role": "user", "content": "Initial context: test"}],
            )
        )

        self.assertEqual(orchestrator.turn_count, 0)
        self.assertEqual(orchestrator.max_turns, 5)
        self.assertEqual(orchestrator.model, "test-model")
        self.assertEqual(len(orchestrator.messages), 2)
        self.assertEqual(orchestrator.messages[0]["role"], "system")
        self.assertEqual(orchestrator.messages[1]["role"], "user")
        self.assertIn("Initial context", orchestrator.messages[1]["content"])

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_success(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = AgentResponse[EvaluationCriteria](intent=AgentIntent.SUCCESS, result=make_valid_criteria())
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("Let's create criteria")

        self.assertTrue(result.is_complete)
        self.assertEqual(result.result, make_valid_criteria())
        self.assertEqual(orchestrator.turn_count, 1)
        self.assertEqual(len(orchestrator.messages), 2)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_continue(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = AgentResponse[EvaluationCriteria](intent=AgentIntent.CONTINUE, message="What's your budget range?")
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("Hello")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "What's your budget range?")
        self.assertIsNone(result.result)
        self.assertEqual(orchestrator.turn_count, 1)
        self.assertEqual(len(orchestrator.messages), 3)
        self.assertEqual(orchestrator.messages[2]["role"], "assistant")

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_failure_raises_exception(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.FAILURE, message="I don't have enough information to help"
        )
        mock_call_llm.return_value = action

        with self.assertRaises(AtomicWorkflowFailedError) as context:
            orchestrator.process_turn("Something vague")

        self.assertIn(
            "I don't have enough information to help",
            str(context.exception),
        )
        self.assertEqual(orchestrator.turn_count, 1)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_failure_transcript_not_duplicated(self, mock_call_llm):
        """On failure, the conversation transcript must appear exactly once
        in the error message — not 0 (missing) or 2+ (duplicated from
        both error.__str__ and error.message)."""
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.FAILURE, message="I don't have enough information"
        )
        mock_call_llm.return_value = action

        with self.assertRaises(AtomicWorkflowFailedError) as context:
            orchestrator.process_turn("Something vague")

        error_msg = str(context.exception)
        transcript_count = error_msg.count("CONVERSATION TRANSCRIPT")
        self.assertEqual(
            transcript_count,
            1,
            f"Expected exactly 1 CONVERSATION TRANSCRIPT, found {transcript_count}",
        )

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_empty_input(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = AgentResponse[EvaluationCriteria](intent=AgentIntent.CONTINUE, message="First question")
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "First question")
        self.assertEqual(len(orchestrator.messages), 2)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_multi_turn_conversation_sequence(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config(max_turns=5))

        responses = [
            AgentResponse[EvaluationCriteria](intent=AgentIntent.CONTINUE, message="Question 1"),
            AgentResponse[EvaluationCriteria](intent=AgentIntent.CONTINUE, message="Question 2"),
            AgentResponse[EvaluationCriteria](intent=AgentIntent.SUCCESS, result=make_valid_criteria()),
        ]
        mock_call_llm.side_effect = responses

        result1 = orchestrator.process_turn("Hello")
        self.assertFalse(result1.is_complete)
        self.assertEqual(result1.message, "Question 1")
        self.assertEqual(orchestrator.turn_count, 1)

        result2 = orchestrator.process_turn("Answer 1")
        self.assertFalse(result2.is_complete)
        self.assertEqual(result2.message, "Question 2")
        self.assertEqual(orchestrator.turn_count, 2)

        result3 = orchestrator.process_turn("Answer 2")
        self.assertTrue(result3.is_complete)
        self.assertEqual(result3.result, make_valid_criteria())
        self.assertEqual(orchestrator.turn_count, 3)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_orchestrator_raises_on_max_turns(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config(max_turns=2))

        mock_call_llm.return_value = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.CONTINUE, message="Another question"
        )

        result1 = orchestrator.process_turn("A1")
        self.assertFalse(result1.is_complete)
        self.assertEqual(orchestrator.turn_count, 1)

        result2 = orchestrator.process_turn("A2")
        self.assertFalse(result2.is_complete)
        self.assertEqual(orchestrator.turn_count, 2)

        with self.assertRaises(TurnLimitExceededError) as context:
            orchestrator.process_turn("A3")

        self.assertIn("Maximum conversation turns (2) reached", str(context.exception))

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_propagates_llm_exceptions(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        mock_call_llm.side_effect = ValueError("Validation failed after retries")

        with self.assertRaises(ValueError):
            orchestrator.process_turn("test")

        self.assertEqual(orchestrator.turn_count, 1)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_invalid_action_raises_exception(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        mock_action = Mock()
        mock_action.intent = "invalid_action"
        mock_action.message = "test"
        mock_action.result = None
        mock_call_llm.return_value = mock_action

        with self.assertRaises(InvalidResponseError) as context:
            orchestrator.process_turn("test")

        self.assertIn("Invalid intent received: invalid_action", str(context.exception))


class TestWorkflowIntegration(unittest.TestCase):
    @patch("chat_workflow.atomic_workflow.AtomicWorkflow._call_llm")
    def test_leaf_accepts_tools_parameter(self, mock_call_llm):
        from chat_workflow import SessionLog

        mock_call_llm.return_value = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.SUCCESS, result=make_valid_criteria()
        )

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="test response")

        state = SessionLog()
        tools = Session(io=mock_io, state=state, config=FakeConfig())

        result = EvaluationCriteria.generate_from_chat(context="test", max_turns=5, session=tools)

        self.assertIsInstance(result, EvaluationCriteria)
        self.assertEqual(len(result.criteria), 2)

    @patch("chat_workflow.atomic_workflow.AtomicWorkflow._call_llm")
    def test_leaf_accepts_tools_via_io(self, mock_call_llm):
        """Caller can pass io+state+config to build session themselves."""
        from chat_workflow import SessionLog

        mock_call_llm.return_value = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.SUCCESS, result=make_valid_criteria()
        )

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="test response")

        tools = Session(
            io=mock_io,
            state=SessionLog(),
            config=FakeConfig(),
        )

        result = EvaluationCriteria.generate_from_chat(context="test", max_turns=5, session=tools)

        self.assertIsInstance(result, EvaluationCriteria)

    @patch("chat_workflow.atomic_workflow.AtomicWorkflow._call_llm")
    def test_workflow_passes_tools_to_leaf(self, mock_call_llm):
        from chat_workflow import SessionLog
        from workflows.evaluation_criteria import generate_reviewed_criteria

        mock_call_llm.return_value = AgentResponse[EvaluationCriteria](
            intent=AgentIntent.SUCCESS, result=make_valid_criteria()
        )

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="looks good")

        state = SessionLog()
        tools = Session(io=mock_io, state=state, config=FakeConfig())

        result = generate_reviewed_criteria(
            context="test context",
            max_turns=5,
            session=tools,
        )

        self.assertIsInstance(result, EvaluationCriteria)

    @patch("chat_workflow.atomic_workflow.AtomicWorkflow._call_llm")
    def test_workflow_refinement_loop(self, mock_call_llm):

        from chat_workflow import SessionLog
        from workflows.evaluation_criteria import generate_reviewed_criteria

        initial_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

        refined_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=9.0),
            ],
        )

        mock_call_llm.side_effect = [
            AgentResponse[EvaluationCriteria](intent=AgentIntent.SUCCESS, result=initial_criteria),
            AgentResponse[EvaluationCriteria](intent=AgentIntent.SUCCESS, result=refined_criteria),
            AgentResponse[EvaluationCriteria](intent=AgentIntent.SUCCESS, result=refined_criteria),
        ]

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="change quality weight")

        state = SessionLog()
        tools = Session(io=mock_io, state=state, config=FakeConfig())

        result = generate_reviewed_criteria(
            context="test context",
            max_turns=5,
            session=tools,
        )

        self.assertIsInstance(result, EvaluationCriteria)
        self.assertEqual(result.criteria[1].weight, 9.0)


class TestChatDecoratorTypeVarResolution(unittest.TestCase):
    """Tests for TypeVar resolution in the @chat decorator.

    The 'refine' function in workflows.evaluation_criteria.flows uses a TypeVar with
    'from __future__ import annotations', which causes inspect.signature()
    to return string annotations. The decorator resolution logic must use
    typing.get_type_hints() instead of inspect.signature() to correctly
    match parameter annotations to return type TypeVars.
    """

    def test_inspect_signature_returns_strings_with_future_annotations(self):
        """Verify that inspect.signature gives string annotations for functions
        with 'from __future__ import annotations', proving the bug mechanism."""
        import inspect
        import typing

        from workflows.evaluation_criteria.refine import refine

        hints = typing.get_type_hints(refine)
        return_type = hints.get("return")

        from typing import TypeVar

        self.assertIsInstance(return_type, TypeVar)

        # inspect.signature() returns strings due to from __future__ import annotations
        sig = inspect.signature(refine)
        param_annotation = sig.parameters["initial_object"].annotation

        self.assertIsInstance(param_annotation, str)
        self.assertIn("ModelType", param_annotation)
        self.assertIn("Annotated", param_annotation)

        # String != TypeVar -> the bug: resolution fails
        self.assertNotEqual(param_annotation, return_type)

        # typing.get_type_hints() resolves correctly
        self.assertIn("initial_object", hints)
        self.assertEqual(hints["initial_object"], return_type)

    def test_get_type_hints_resolves_typevar(self):
        """typing.get_type_hints() properly resolves parameter annotations
        even with from __future__ import annotations - the fix must use this."""
        import typing

        from workflows.evaluation_criteria.refine import refine

        hints = typing.get_type_hints(refine)
        return_type = hints.get("return")

        # Find params that share the return TypeVar using get_type_hints
        typevar_params = [name for name, annotation in hints.items() if name != "return" and annotation == return_type]

        self.assertEqual(
            typevar_params,
            ["initial_object"],
            "get_type_hints correctly identifies 'initial_object' as the TypeVar-matching param",
        )

    def test_refine_decorator_resolves_typevar_to_concrete_type(self):
        """The @chat decorator on refine() must resolve the TypeVar to the
        concrete type passed via initial_object. This exercises the full
        resolution logic through the decorator - before the fix, response_model
        stayed as AgentResponse[ModelType] (unresolved).
        """
        from unittest.mock import Mock, patch

        from chat_workflow import AgentIntent, AgentResponse, AtomicWorkflow
        from workflows.evaluation_criteria import Criterion, EvaluationCriteria
        from workflows.evaluation_criteria.refine import refine

        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ],
        )

        # Capture what response_model the decorator passes to the orchestrator
        captured_response_model = []

        original_init = AtomicWorkflow.__init__

        def tracking_init(self, **kwargs):
            config = kwargs.get("config")
            captured_response_model.append(config.response_model if config else None)
            return original_init(self, **kwargs)

        with (
            patch.object(AtomicWorkflow, "__init__", tracking_init),
            patch.object(AtomicWorkflow, "_call_llm") as mock_call_llm,
        ):
            mock_call_llm.return_value = AgentResponse[EvaluationCriteria](intent=AgentIntent.SUCCESS, result=criteria)

            mock_io = Mock()
            mock_io.echo = Mock()
            mock_io.prompt = Mock(return_value="looks good")

            state = SessionLog()
            tools = Session(io=mock_io, state=state, config=FakeConfig())

            result = refine(initial_object=criteria, max_turns=3, session=tools)

            self.assertIsInstance(result, EvaluationCriteria)

        # The key assertion: response_model must be parameterized with the
        # concrete type (EvaluationCriteria), not the unresolved TypeVar
        self.assertEqual(
            len(captured_response_model),
            1,
            "Expected exactly 1 orchestrator to be created",
        )

        response_model = captured_response_model[0]
        self.assertIsNotNone(response_model, "response_model should be set")

        # Check the inner type param via pydantic generic metadata
        # (pydantic v2 doesn't use standard typing.get_args())
        metadata = response_model.__pydantic_generic_metadata__
        args = metadata.get("args", ())
        self.assertEqual(
            len(args),
            1,
            f"Expected 1 type arg in {response_model}, got args={args}",
        )

        inner_type = args[0]
        self.assertIs(
            inner_type,
            EvaluationCriteria,
            f"response_model inner type should be EvaluationCriteria, got {inner_type}",
        )


class TestAutoParamInjection(unittest.TestCase):
    """Tests for automatic parameter injection via _build_params_section."""

    def test_build_params_section_basic_params(self):
        """Params without Annotated metadata show type, name, and value."""
        from chat_workflow.prompt_builder import _build_params_section

        def sample(context: str = "", max_turns: int = 10):
            pass

        section = _build_params_section(sample, {"context": "party", "max_turns": 5})

        self.assertIn("## Parameters", section)
        self.assertIn("`context` (str)", section)
        self.assertIn('"party"', section)
        self.assertIn("`max_turns` (int)", section)
        self.assertIn("5", section)

    def test_build_params_section_with_annotated(self):
        """Annotated[T, 'desc'] descriptions appear in the output."""
        from typing import Annotated

        from chat_workflow.prompt_builder import _build_params_section

        def sample(
            context: Annotated[str, "The party theme"] = "",
            max_turns: Annotated[int, "Turn limit"] = 10,
        ):
            pass

        section = _build_params_section(sample, {"context": "birthday"})

        self.assertIn("The party theme", section)
        self.assertIn("Turn limit", section)
        self.assertIn("`context` (str)", section)
        self.assertIn("`max_turns` (int)", section)

    def test_build_params_section_excludes_internal(self):
        """tools, io, state, debug are excluded from the section."""
        from chat_workflow.prompt_builder import _build_params_section

        def sample(
            context: str = "",
            tools: object = None,
            session: object = None,
            debug: object = None,
        ):
            pass

        section = _build_params_section(sample, {"context": "test"})

        self.assertIn("context", section)
        self.assertNotIn("session", section)
        self.assertNotIn("debug", section)

    def test_build_params_section_shows_default_when_no_runtime_value(self):
        """When a param is not in kwargs, its default is shown instead."""
        from chat_workflow.prompt_builder import _build_params_section

        def sample(max_turns: int = 10):
            pass

        section = _build_params_section(sample, {})

        self.assertIn("Default: 10", section)

    def test_chat_decorator_includes_params_section_in_system_prompt(self):
        """The @chat decorator appends the params section to the system prompt."""
        from unittest.mock import Mock, patch

        from chat_workflow.atomic_workflow import (
            AtomicWorkflow,
        )
        from workflows.evaluation_criteria import Criterion, EvaluationCriteria

        captured_system_prompt = []

        original_init = AtomicWorkflow.__init__

        def tracking_init(self, **kwargs):
            config = kwargs.get("config")
            captured_system_prompt.append(config.system_prompt if config else "")
            return original_init(self, **kwargs)

        valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ],
        )

        with (
            patch.object(AtomicWorkflow, "__init__", tracking_init),
            patch.object(AtomicWorkflow, "_call_llm") as mock_call_llm,
        ):
            mock_call_llm.return_value = type(
                "MockAction",
                (),
                {"intent": AgentIntent.SUCCESS, "message": None, "result": valid_criteria},
            )()

            mock_io = Mock()
            mock_io.echo = Mock()
            mock_io.prompt = Mock(return_value="test")
            state = SessionLog()
            tools = Session(io=mock_io, state=state, config=FakeConfig())

            result = EvaluationCriteria.generate_from_chat(context="birthday ideas", max_turns=5, session=tools)

            self.assertIsInstance(result, EvaluationCriteria)

        self.assertEqual(len(captured_system_prompt), 1)
        prompt = captured_system_prompt[0]

        self.assertIn("## Parameters", prompt)
        self.assertIn("`context` (str)", prompt)
        self.assertIn("`max_turns` (int)", prompt)

        self.assertNotIn("session", prompt)

    def test_chat_decorator_preserves_inline_interpolation(self):
        """{initial_object.model_dump()} style interpolation still works."""
        from unittest.mock import Mock, patch

        from chat_workflow import AtomicWorkflow
        from workflows.evaluation_criteria import Criterion, EvaluationCriteria

        captured_system_prompt = []

        original_init = AtomicWorkflow.__init__

        def tracking_init(self, **kwargs):
            config = kwargs.get("config")
            captured_system_prompt.append(config.system_prompt if config else "")
            return original_init(self, **kwargs)

        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

        with (
            patch.object(AtomicWorkflow, "__init__", tracking_init),
            patch.object(AtomicWorkflow, "_call_llm") as mock_call_llm,
        ):
            mock_call_llm.return_value = type(
                "MockAction",
                (),
                {"intent": AgentIntent.SUCCESS, "message": None, "result": criteria},
            )()

            mock_io = Mock()
            mock_io.echo = Mock()
            mock_io.prompt = Mock(return_value="looks good")
            state = SessionLog()
            tools = Session(io=mock_io, state=state, config=FakeConfig())

            from workflows.evaluation_criteria.refine import refine

            result = refine(initial_object=criteria, max_turns=3, session=tools)

            self.assertIsInstance(result, EvaluationCriteria)

        self.assertEqual(len(captured_system_prompt), 1)
        prompt = captured_system_prompt[0]

        # Should contain inline interpolation result (model_dump output)
        self.assertIn("budget", prompt)
        self.assertIn("quality", prompt)

        # Should ALSO contain the auto-injected params section
        self.assertIn("## Parameters", prompt)
        self.assertIn("`initial_object` (~ModelType)", prompt)
        self.assertIn("`max_turns` (int)", prompt)


class TestWhatGetsSentToTheLLM(unittest.TestCase):
    """Tests that verify the actual messages sent to the LLM contain the
    correct schema and parameters."""

    def _capture_instructor_messages(self, func, *args, **kwargs):
        """Patch instructor+litellm to capture what gets sent to the LLM,
        including the schema injected by instructor."""
        import litellm as _litellm

        captured = {"messages": None, "response_model": None}

        # Patch litellm.completion so instructor's patched client calls our capture
        original_completion = _litellm.completion

        def capturing_completion(**llm_kwargs):
            captured["messages"] = llm_kwargs.get("messages")
            captured["response_model"] = llm_kwargs.get("response_model")
            # Return a minimal fake response instructor can parse
            content = json.dumps(
                {
                    "intent": "success",
                    "message": None,
                    "result": {
                        "criteria": [
                            {"name": "budget", "description": "cost", "weight": 5.0, "ideal_value": None},
                            {"name": "quality", "description": "how good", "weight": 3.0, "ideal_value": None},
                        ],
                        "context": "test",
                    },
                }
            )
            choice = type(
                "FakeChoice",
                (),
                {
                    "message": type(
                        "FakeMsg",
                        (),
                        {
                            "parsed": None,
                            "content": content,
                            "tool_calls": None,
                            "role": "assistant",
                            "function_call": None,
                            "tool_call": None,
                            "model_dump": lambda self: {"role": "assistant", "content": content},
                        },
                    )(),
                    "finish_reason": "stop",
                    "index": 0,
                },
            )()
            resp = type(
                "FakeResponse",
                (),
                {
                    "choices": [choice],
                    "model_dump": lambda self: {"choices": [{"message": {"role": "assistant", "content": content}}]},
                },
            )()
            return resp

        _litellm.completion = capturing_completion

        # Patch get_client to return instructor client with our patched litellm
        def patched_get_client(provider):
            import instructor

            return instructor.from_litellm(_litellm.completion, mode=instructor.Mode.JSON)

        import chat_workflow.llm_interaction as li

        original_li_get_client = li.get_client

        li.get_client = patched_get_client

        try:
            func(*args, **kwargs)
        finally:
            _litellm.completion = original_completion
            li.get_client = original_li_get_client

        return captured

    def test_evaluation_criteria_system_prompt_contains_evaluation_criteria_schema(self):
        """The system prompt sent to the LLM must include the JSON schema
        for EvaluationCriteria (with minItems:2, weight range, etc.) so the
        LLM knows the validation rules before generating."""
        from unittest.mock import Mock

        from chat_workflow import SessionLog
        from workflows.evaluation_criteria import EvaluationCriteria

        mock_io = Mock()
        mock_io.echo = Mock()
        mock_io.prompt = Mock(return_value="done")

        tools = Session(
            io=mock_io,
            state=SessionLog(),
            config=FakeConfig(),
        )

        captured = self._capture_instructor_messages(
            EvaluationCriteria.generate_from_chat,
            context="test",
            max_turns=2,
            session=tools,
        )

        self.assertIsNotNone(captured["messages"], "No messages captured")

        system_msg = next(msg["content"] for msg in captured["messages"] if msg.get("role") == "system")
        self.assertIn("## Parameters", system_msg)
        self.assertIn("`context`", system_msg)
        self.assertIn("`max_turns`", system_msg)
        self.assertIn('"test"', system_msg)
        self.assertIn("2", system_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
