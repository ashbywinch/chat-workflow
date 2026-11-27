#!/usr/bin/env python3

import unittest
from typing import override
from unittest.mock import Mock

from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RunUsage

from chat_workflow import (
    AgentAnswer,
    AgentIntent,
    AtomicWorkflow,
    AtomicWorkflowFailedError,
    Continue,
    Failure,
    InvalidResponseError,
    ModelTurn,
    Session,
    SessionLog,
    Success,
    TurnLimitExceededError,
    TurnResult,
    atomic_workflow_step,
)
from chat_workflow.session import UserIO
from tests.conftest import (
    FakeConfig,
    ScriptedModel,
    make_atomic_workflow_config,
    make_valid_criteria,
)
from workflows.evaluation_criteria import Criterion, EvaluationCriteria


class _ScriptedConversation:
    """A Conversation-protocol fake: it replays outputs and records prompts."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.messages = []
        self.prompts = []
        self.final_turns = []

    def send(self, prompt=None, *, final_turn=False, chat_off=False):
        self.prompts.append(prompt)
        self.final_turns.append(final_turn)
        index = min(len(self.prompts) - 1, len(self._outputs) - 1)
        output = self._outputs[index]
        if isinstance(output, BaseException):
            raise output
        return ModelTurn(output=output, messages=[], usage=RunUsage())


class _RecordingIO(UserIO):
    """A UserIO that records echoes and repeats one answer."""

    def __init__(self, answer: str = "test response"):
        self.answer = answer
        self.echoes: list[str] = []

    @override
    def echo(self, message: str) -> None:
        self.echoes.append(message)

    @override
    def prompt(self, label: str) -> str:
        return self.answer


class _CapturingSession:
    """A Session-protocol fake that captures the workflow a decorator builds."""

    def __init__(self, result):
        self.config = FakeConfig()
        self.state = SessionLog()
        self.instructions = None
        self.io = _RecordingIO()
        self.workflows = []
        self._result = result

    def run(self, *, workflow, chat):
        self.workflows.append(workflow)
        return TurnResult[EvaluationCriteria].success(self._result)


def _integration_session(responses, io=None) -> Session:
    """A real Session whose model is a scripted FunctionModel (no network)."""
    config = FakeConfig()
    config.model_client = FunctionModel(ScriptedModel(responses))
    return Session(io=io or _RecordingIO(), state=SessionLog(), config=config)


def _success_payload(result) -> dict:
    return {"intent": "success", "result": result.model_dump()}


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

    def test_process_turn_success(self):
        action = Success[EvaluationCriteria](
            intent=AgentIntent.SUCCESS,
            result=make_valid_criteria(),
        )
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=_ScriptedConversation([action]))
        )

        result = orchestrator.process_turn("Let's create criteria")

        self.assertTrue(result.is_complete)
        self.assertEqual(result.result, make_valid_criteria())
        self.assertEqual(orchestrator.turn_count, 1)

    def test_process_turn_continue(self):
        action = Continue(intent=AgentIntent.CONTINUE, message="What's your budget range?")
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=_ScriptedConversation([action]))
        )

        result = orchestrator.process_turn("Hello")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "What's your budget range?")
        self.assertIsNone(result.result)
        self.assertEqual(orchestrator.turn_count, 1)

    def test_process_turn_failure_raises_exception(self):
        """A failure is provisional: the step gets one turn to continue."""
        action = Failure(
            intent=AgentIntent.FAILURE,
            message="I don't have enough information to help",
        )
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=_ScriptedConversation([action]))
        )

        with self.assertRaises(AtomicWorkflowFailedError) as context:
            orchestrator.process_turn("Something vague")

        self.assertIn("I don't have enough information to help", str(context.exception))
        self.assertEqual(orchestrator.turn_count, 2)

    def test_process_turn_failure_transcript_not_duplicated(self):
        """On failure, the conversation transcript must appear exactly once."""
        action = Failure(intent=AgentIntent.FAILURE, message="I don't have enough information")
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=_ScriptedConversation([action]))
        )

        with self.assertRaises(AtomicWorkflowFailedError) as context:
            orchestrator.process_turn("Something vague")

        error_msg = str(context.exception)
        transcript_count = error_msg.count("CONVERSATION TRANSCRIPT")
        self.assertEqual(
            transcript_count,
            1,
            f"Expected exactly 1 CONVERSATION TRANSCRIPT, found {transcript_count}",
        )

    def test_process_turn_empty_input(self):
        action = Continue(intent=AgentIntent.CONTINUE, message="First question")
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=_ScriptedConversation([action]))
        )

        result = orchestrator.process_turn("")

        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "First question")

    def test_multi_turn_conversation_sequence(self):
        responses = [
            Continue(intent=AgentIntent.CONTINUE, message="Question 1"),
            Continue(intent=AgentIntent.CONTINUE, message="Question 2"),
            Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                result=make_valid_criteria(),
            ),
        ]
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(
                max_turns=5,
                conversation=_ScriptedConversation(responses),
            )
        )

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

    def test_no_max_turns_never_forces_a_final_turn(self):
        responses = [
            Continue(intent=AgentIntent.CONTINUE, message="Question 1"),
            Continue(intent=AgentIntent.CONTINUE, message="Question 2"),
            Continue(intent=AgentIntent.CONTINUE, message="Question 3"),
            Success[EvaluationCriteria](
                intent=AgentIntent.SUCCESS,
                result=make_valid_criteria(),
            ),
        ]
        conversation = _ScriptedConversation(responses)
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(
                max_turns=None,
                conversation=conversation,
            )
        )

        for answer in ("A1", "A2", "A3"):
            result = orchestrator.process_turn(answer)
            self.assertFalse(result.is_complete)

        result = orchestrator.process_turn("A4")

        self.assertTrue(result.is_complete)
        self.assertEqual(orchestrator.turn_count, 4)
        self.assertFalse(
            any(conversation.final_turns),
            "a step with no turn limit has no final turn to force",
        )

    def test_orchestrator_raises_on_max_turns(self):
        action = Continue(intent=AgentIntent.CONTINUE, message="Another question")
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(
                max_turns=2,
                conversation=_ScriptedConversation([action]),
            )
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

    def test_process_turn_propagates_llm_exceptions(self):
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(
                conversation=_ScriptedConversation([ValueError("Validation failed after retries")])
            )
        )

        with self.assertRaises(ValueError):
            orchestrator.process_turn("test")

        self.assertEqual(orchestrator.turn_count, 1)

    def test_invalid_action_raises_exception(self):
        mock_action = Mock()
        mock_action.intent = "invalid_action"
        mock_action.message = "test"
        mock_action.result = None
        orchestrator = AtomicWorkflow(
            config=make_atomic_workflow_config(conversation=_ScriptedConversation([mock_action]))
        )

        with self.assertRaises(InvalidResponseError) as context:
            orchestrator.process_turn("test")

        self.assertIn("Invalid intent received: invalid_action", str(context.exception))


class TestWorkflowIntegration(unittest.TestCase):
    def test_leaf_accepts_tools_parameter(self):
        session = _integration_session(
            [_success_payload(make_valid_criteria())], io=_RecordingIO("test response")
        )

        result = EvaluationCriteria.generate_from_chat(
            context="test", max_turns=5, session=session
        )

        self.assertIsInstance(result, EvaluationCriteria)
        self.assertEqual(len(result.criteria), 2)

    def test_leaf_accepts_tools_via_io(self):
        """Caller can pass io+state+config to build session themselves."""
        session = _integration_session(
            [_success_payload(make_valid_criteria())], io=_RecordingIO("test response")
        )

        result = EvaluationCriteria.generate_from_chat(
            context="test", max_turns=5, session=session
        )

        self.assertIsInstance(result, EvaluationCriteria)

    def test_workflow_passes_tools_to_leaf(self):
        from workflows.evaluation_criteria import generate_reviewed_criteria

        session = _integration_session(
            [_success_payload(make_valid_criteria())], io=_RecordingIO("looks good")
        )

        result = generate_reviewed_criteria(
            context="test context",
            max_turns=5,
            session=session,
        )

        self.assertIsInstance(result, EvaluationCriteria)

    def test_workflow_refinement_loop(self):
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
        session = _integration_session(
            [
                _success_payload(initial_criteria),
                _success_payload(refined_criteria),
                _success_payload(refined_criteria),
            ],
            io=_RecordingIO("change quality weight"),
        )

        result = generate_reviewed_criteria(
            context="test context",
            max_turns=5,
            session=session,
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
        """The @chat decorator on refine() resolves the TypeVar to the concrete type."""
        from workflows.evaluation_criteria import Criterion, EvaluationCriteria
        from workflows.evaluation_criteria.refine import refine

        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ],
        )
        session = _CapturingSession(criteria)

        result = refine(initial_object=criteria, max_turns=3, session=session)

        self.assertIsInstance(result, EvaluationCriteria)
        self.assertEqual(len(session.workflows), 1)
        response_model = session.workflows[0].response_model
        self.assertIsNotNone(response_model, "response_model should be set")
        self.assertEqual(response_model, AgentAnswer[EvaluationCriteria])

    def test_a_caller_can_remove_the_turn_limit(self):
        session = _CapturingSession(make_valid_criteria())

        EvaluationCriteria.generate_from_chat(
            context="test", max_turns=None, session=session
        )

        self.assertIsNone(session.workflows[0].max_turns)

    def test_a_step_declared_default_is_the_turn_bound(self):
        from workflows.evaluation_criteria import Criterion
        from workflows.evaluation_criteria.refine import refine

        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ],
        )
        session = _CapturingSession(criteria)

        refine(initial_object=criteria, session=session)

        self.assertEqual(session.workflows[0].max_turns, 5)

    def test_a_step_can_declare_no_turn_limit(self):
        @atomic_workflow_step
        def step(context: str, max_turns: int | None = None) -> EvaluationCriteria:
            """Build criteria from {context}."""
            return make_valid_criteria()

        session = _CapturingSession(make_valid_criteria())

        step(context="test", session=session)

        self.assertIsNone(session.workflows[0].max_turns)


class TestAutoParamInjection(unittest.TestCase):
    """Tests for automatic parameter injection via build_params_section."""

    def testbuild_params_section_basic_params(self):
        """Params without Annotated metadata show type, name, and value."""
        from chat_workflow.prompt_builder import build_params_section

        def sample(context: str = "", max_turns: int = 10):
            pass

        section = build_params_section(sample, {"context": "party", "max_turns": 5})

        self.assertIn("## Parameters", section)
        self.assertIn("`context` (str)", section)
        self.assertIn('"party"', section)
        self.assertIn("`max_turns` (int)", section)
        self.assertIn("5", section)

    def testbuild_params_section_with_annotated(self):
        """Annotated[T, 'desc'] descriptions appear in the output."""
        from typing import Annotated

        from chat_workflow.prompt_builder import build_params_section

        def sample(
            context: Annotated[str, "The party theme"] = "",
            max_turns: Annotated[int, "Turn limit"] = 10,
        ):
            pass

        section = build_params_section(sample, {"context": "birthday"})

        self.assertIn("The party theme", section)
        self.assertIn("Turn limit", section)
        self.assertIn("`context` (str)", section)
        self.assertIn("`max_turns` (int)", section)

    def testbuild_params_section_excludes_internal(self):
        """tools, io, state, debug are excluded from the section."""
        from chat_workflow.prompt_builder import build_params_section

        def sample(
            context: str = "",
            tools: object = None,
            session: object = None,
            debug: object = None,
        ):
            pass

        section = build_params_section(sample, {"context": "test"})

        self.assertIn("context", section)
        self.assertNotIn("session", section)
        self.assertNotIn("debug", section)

    def testbuild_params_section_shows_default_when_no_runtime_value(self):
        """When a param is not in kwargs, its default is shown instead."""
        from chat_workflow.prompt_builder import build_params_section

        def sample(max_turns: int = 10):
            pass

        section = build_params_section(sample, {})

        self.assertIn("Default: 10", section)

    def test_chat_decorator_includes_params_section_in_system_prompt(self):
        """The @chat decorator appends the params section to the system prompt."""
        from workflows.evaluation_criteria import Criterion, EvaluationCriteria

        valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ],
        )
        session = _CapturingSession(valid_criteria)

        result = EvaluationCriteria.generate_from_chat(
            context="birthday ideas", max_turns=5, session=session
        )

        self.assertIsInstance(result, EvaluationCriteria)
        self.assertEqual(len(session.workflows), 1)
        prompt = session.workflows[0]._conversation._instructions
        self.assertIn("## Parameters", prompt)
        self.assertIn("`context` (str)", prompt)
        self.assertIn("`max_turns` (int)", prompt)
        self.assertNotIn("session", prompt)

    def test_the_opening_leaves_the_first_move_to_the_step(self):
        session = _CapturingSession(make_valid_criteria())

        EvaluationCriteria.generate_from_chat(context="birthday ideas", session=session)

        prompt = session.workflows[0]._conversation._instructions
        self.assertIn(
            "say what you can already see from the inputs, or ask the one "
            "question the step needs",
            prompt,
        )
        self.assertNotIn("propose what you can already see from them", prompt)

    def test_chat_decorator_preserves_inline_interpolation(self):
        """{initial_object.model_dump()} style interpolation still works."""
        from workflows.evaluation_criteria import Criterion, EvaluationCriteria
        from workflows.evaluation_criteria.refine import refine

        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )
        session = _CapturingSession(criteria)

        result = refine(initial_object=criteria, max_turns=3, session=session)

        self.assertIsInstance(result, EvaluationCriteria)
        self.assertEqual(len(session.workflows), 1)
        prompt = session.workflows[0]._conversation._instructions
        self.assertIn("budget", prompt)
        self.assertIn("quality", prompt)
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

        session.run(workflow=orchestrator)

        self.assertEqual(len(state.messages), 3)  # request, answer call, return
        self.assertEqual(state.usage.requests, 1)
        self.assertEqual(state.turn_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
