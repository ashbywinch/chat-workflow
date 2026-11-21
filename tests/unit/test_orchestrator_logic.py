#!/usr/bin/env python3
import unittest
from unittest.mock import Mock, patch

from chat_workflow import (
    AgentAnswer,
    AgentIntent,
    AtomicWorkflow,
    AtomicWorkflowFailedError,
    Continue,
    Failure,
    InvalidResponseError,
    Session,
    SessionLog,
    Success,
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
        self.assertEqual(len(orchestrator.messages), 1)
        from pydantic_ai.messages import UserPromptPart

        initial = orchestrator.messages[0]
        self.assertEqual(initial.kind, "request")
        text = next(part.content for part in initial.parts if isinstance(part, UserPromptPart))
        self.assertIn("Initial context", text)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_success(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=make_valid_criteria(),
            )
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("Let's create criteria")

        self.assertTrue(result.is_complete)
        self.assertEqual(result.result, make_valid_criteria())
        self.assertEqual(orchestrator.turn_count, 1)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_continue(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = Continue(intent=AgentIntent.CONTINUE, message="What's your budget range?")
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("Hello")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "What's your budget range?")
        self.assertIsNone(result.result)
        self.assertEqual(orchestrator.turn_count, 1)

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_process_turn_failure_raises_exception(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config())

        action = Failure(intent=AgentIntent.FAILURE, message="I don't have enough information to help")
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

        action = Failure(intent=AgentIntent.FAILURE, message="I don't have enough information")
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

        action = Continue(intent=AgentIntent.CONTINUE, message="First question")
        mock_call_llm.return_value = action

        result = orchestrator.process_turn("")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "First question")

    @patch.object(AtomicWorkflow, "_call_llm")
    def test_multi_turn_conversation_sequence(self, mock_call_llm):
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config(max_turns=5))

        responses = [
            Continue(intent=AgentIntent.CONTINUE, message="Question 1"),
            Continue(intent=AgentIntent.CONTINUE, message="Question 2"),
            Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=make_valid_criteria(),
            ),
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

        mock_call_llm.return_value = Continue(intent=AgentIntent.CONTINUE, message="Another question")

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

        mock_call_llm.return_value = Success[EvaluationCriteria](
            intent=AgentIntent.SUCCESS,
            message="Done.", result=make_valid_criteria()
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

        mock_call_llm.return_value = Success[EvaluationCriteria](
            intent=AgentIntent.SUCCESS,
            message="Done.", result=make_valid_criteria()
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

        mock_call_llm.return_value = Success[EvaluationCriteria](
            intent=AgentIntent.SUCCESS,
            message="Done.", result=make_valid_criteria()
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
            Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=initial_criteria,
            ),
            Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=refined_criteria,
            ),
            Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=refined_criteria,
            ),
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
        stayed as AgentAnswer[ModelType] (unresolved).
        """
        from unittest.mock import Mock, patch

        from chat_workflow import AtomicWorkflow
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
            mock_call_llm.return_value = Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                message="Done.",
                result=criteria,
            )

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

        # The decorator resolved the TypeVar: the concrete union, not the
        # unresolved AgentAnswer[ModelType].
        self.assertEqual(response_model, AgentAnswer[EvaluationCriteria])


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
                {"intent": AgentIntent.SUCCESS, "message": "Done.", "result": valid_criteria},
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
                {"intent": AgentIntent.SUCCESS, "message": "Done.", "result": criteria},
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


class TestTheStepOnTheTranscript(unittest.TestCase):
    """The step's turns and the transcript are the same conversation."""

    def test_a_continue_then_a_success_are_recorded_in_order(self):
        from chat_workflow.transcript import entries
        from tests.conftest import make_conversation

        conversation = make_conversation(
            [
                {"intent": "continue", "message": "What is the budget?"},
                {"intent": "success", "result": make_valid_criteria().model_dump()},
            ]
        )
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config(conversation=conversation))

        first = orchestrator.process_turn("I need criteria")
        second = orchestrator.process_turn("About fifty pounds")

        self.assertFalse(first.is_complete)
        self.assertTrue(second.is_complete)
        kinds = [kind for kind, _ in entries(orchestrator.messages)]
        self.assertEqual(kinds, ["user", "assistant", "user", "output"])

    def test_the_session_log_receives_the_transcript_and_usage(self):
        from tests.conftest import make_conversation

        conversation = make_conversation([{"intent": "success", "result": make_valid_criteria().model_dump()}])
        orchestrator = AtomicWorkflow(config=make_atomic_workflow_config(conversation=conversation))
        mock_io = Mock()
        mock_io.echo = Mock()
        state = SessionLog()
        session = Session(io=mock_io, state=state, config=FakeConfig())

        session.run(workflow=orchestrator, first_user_input="hello")

        self.assertEqual(len(state.messages), 4)  # request, answer call, return, words
        self.assertEqual(state.usage.requests, 1)
        self.assertEqual(state.turn_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
