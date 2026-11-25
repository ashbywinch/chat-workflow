"""Decorators for chat-workflow: @atomic_workflow and @composite_workflow."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Annotated, Any, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from .atomic_workflow import AtomicWorkflow
from .atomic_workflow_config import AtomicWorkflowConfig
from .conversation import Conversation
from .conversation_rules import DEFAULT_RULES as _DEFAULT_CONVERSATION_RULES
from .exceptions import AtomicWorkflowFailedError
from .models import AgentAnswer, FinalAnswer, Success, TurnResult
from .prompt_builder import (
    build_params_section,
    format_docstring,
    input_models,
    user_parameter_labels,
)
from .streamingdebug import StreamingDebug
from .types_meta import resolve_return_type


def _inner_model(return_type: Any) -> Any:
    """Return the result model behind a return type, unwrapping list[Model]."""
    origin = get_origin(return_type)
    if origin is Annotated:
        return _inner_model(get_args(return_type)[0])
    if origin is list:
        args = get_args(return_type)
        if args:
            return args[0]
    return return_type


def _is_result_model(return_type: Any) -> bool:
    """True for a Pydantic model, a list of one, or plain text."""
    if return_type is str:
        return True
    inner = _inner_model(return_type)
    return isinstance(inner, type) and issubclass(inner, BaseModel)


def _build_checklist_section(rules: dict[str, str]) -> str:
    lines = ["## Conversation checklist", "Apply these to your own behaviour:"]
    for label, description in rules.items():
        lines.append(f"- **{label}**: {description}")
    return "\n".join(lines)


def _build_validation_rules_section(return_type: Any) -> str:
    """Put the result model's class-level validation rules in the prompt."""
    model = _inner_model(return_type)
    rules = getattr(model, "_validation_rules", None)
    if not rules:
        return ""
    heading = getattr(model, "_validation_rules_heading", "Quality rules for the result")
    lines = [
        f"## {heading}",
        "These describe the finished thing. They are not a request to "
        "review, grade, or report on it.",
    ]
    lines.extend(f"- {rule}" for rule in rules)
    return "\n".join(lines)


def _output_defs(return_type: Any) -> frozenset[str]:
    """Model names already defined in the output schema, so input Shapes can be skipped."""
    if return_type is None:
        return frozenset()
    try:
        schema = TypeAdapter(AgentAnswer[return_type]).json_schema()
    # lucidlint: ignore broad-except a return type that cannot build a schema has no output defs
    except Exception:
        try:
            schema = TypeAdapter(FinalAnswer[return_type]).json_schema()
        # lucidlint: ignore broad-except a return type that cannot build a schema has no output defs
        except Exception:
            return frozenset()
    return frozenset(schema.get("$defs", {}))


def _build_user_vocabulary_section(
    return_type: Any,
    inputs: tuple[type[BaseModel], ...] = (),
    parameter_labels: dict[str, str] | None = None,
) -> str:
    """Tell the model which names are internal and where each phrase lives."""
    parameter_labels = parameter_labels or {}
    type_names = [
        model.__name__
        for model in (*inputs, _inner_model(return_type))
        if isinstance(model, type) and issubclass(model, BaseModel)
    ]
    if not parameter_labels and not type_names:
        return ""
    lines = [
        "## Use the user's words",
        "Each field's title is the phrase to use for it; the key and the type "
        "names are internal engineering text. Refer to a parameter by its "
        "description, and to a type by the thing itself. Never say a field "
        "name, a parameter name, or a type name aloud.",
    ]
    if parameter_labels:
        lines.append("Parameters are how the step receives what it was given:")
        lines.extend(
            f'- `{name}` -> "{description}"'
            for name, description in parameter_labels.items()
        )
    if type_names:
        lines.append("Describe the thing itself; the type names above are internal.")
    return "\n".join(lines)


# lucidlint: ignore long-param-list the prompt sections are independent inputs; one call site names each
def _build_system_prompt(
    raw_func: Callable[..., Any],
    kwargs: dict[str, Any],  # lucidlint: ignore record-shape the step's kwargs are arbitrary per function
    *,
    return_type: Any,
    conversation_rules: dict[str, str],
    chat: bool,
    instructions: str | None = None,
    explore: bool = False,
) -> str:
    """Build the system prompt from the session, the step and its rules.

    ``instructions`` is the session's overall how — the session component's
    docstring — and comes first. The step contributes its docstring (the
    procedure), its parameters, the conversation checklist, the result model's
    class-level rules, and the user's words for the internal names.
    """
    sections: list[str] = []

    if instructions and instructions.strip():
        sections.append(instructions.strip())

    docstring = format_docstring(raw_func.__doc__, **kwargs).strip()
    if docstring:
        sections.append(docstring)

    params_section = build_params_section(
        raw_func,
        kwargs,
        skip_shapes=_output_defs(return_type),
    )
    if params_section:
        sections.append(params_section.strip())

    if chat:
        sections.append(_opening_section(explore))

    if chat and conversation_rules:
        sections.append(_build_checklist_section(conversation_rules))

    validation_section = _build_validation_rules_section(return_type)
    if validation_section:
        sections.append(validation_section)

    # Last, because the schemas and rules above are full of the internal words
    # the model must not say. The instruction to translate them has to be the
    # most recent thing it reads, not the thing it read before the schema.
    vocabulary_section = _build_user_vocabulary_section(
        return_type,
        input_models(raw_func),
        user_parameter_labels(raw_func),
    )
    if vocabulary_section:
        sections.append(vocabulary_section)

    sections.extend(_turn_policy_sections(chat, return_type))

    return "\n\n".join(sections)


def _opening_section(explore: bool) -> str:
    """The start-of-step instruction: listen, or propose from the inputs."""
    if explore:
        start = (
            "## Start of the step\n"
            "Say in one sentence what this step is doing and why. Then ask "
            "one open question about the user's own work and listen. This "
            "step does not name stages, steps, or fields, and does not "
            "propose a structure; the user describes, the next step "
            "structures. Do not summarize the work back for confirmation and "
            "do not offer to structure it: acknowledge in one clause and ask "
            "the next open question. If the user asks for a shape, a section "
            "list, or a suggestion, say in one sentence that the next step "
            "shows one and finish — do not answer with a promised list or a "
            "message that ends with a colon."
        )
    else:
        start = (
            "## Start of the step\n"
            "Say in one sentence what this step is doing and why. Then make "
            "your first substantive move: if the step was given inputs, "
            "propose what you can already see from them — the result, or "
            "the part of it they cover — and ask for a reaction. Do not "
            "ask the user to name a thing the inputs already name, and do "
            "not ask them to supply or confirm something the inputs "
            "already contain. Do not return the complete result in the "
            "first response unless the inputs already contain all of it "
            "and the user has confirmed it."
        )
    return start


def _turn_policy_sections(chat: bool, return_type: Any) -> list[str]:
    """The turn policy, success-message and one-shot rules."""
    sections: list[str] = []
    # The model now sees its own previous structured turns in the transcript,
    # so the turn policy has to be stated rather than inferred: without it a
    # cooperative "that's all, please finalize" gets another question and the
    # step runs to its turn limit.
    sections.append(
        "## When to finish\n"
        "Use intent='continue' only while you still need something from the "
        "user. When the user says they are done, agrees with what you "
        "proposed, or has supplied what the result needs, return "
        "intent='success' — every value in the result must come from what "
        "they told you. A question they left unanswered does not hold the "
        "result back: finish with what they did give. A confirmation is not a "
        "failure: when the user agrees with a draft you already have, return "
        "success with it unchanged — never failure because they supplied "
        "nothing new. Return failure only when you cannot produce the result "
        "at all: the user refuses to engage, the input cannot be interpreted, "
        "or the turn limit is reached without the values the result requires. "
        "Do not invent a value they did not give to avoid that failure."
    )
    sections.append(
        "## The success message\n"
        "A success is still a turn: `message` is what the user reads when the "
        "step ends. Say what you decided, what changed, or that it is "
        "confirmed, in one or two sentences in the user's words. Speak to "
        "the user as 'you'; never describe the conversation in the third "
        "person or say 'the user'. Do not list the result's values there; "
        "they are in `result`. If the user's last message asked something, "
        "answer it in `message` before you finish.\n\n"
        "When the user corrects or adds something, update the result and say in "
        "one sentence what changed. Say it in the turn where it changes; in a "
        "later turn, do not restate a change they already acknowledged. The "
        "framework shows a changed draft to the user and asks for "
        "confirmation, so do not present the result yourself, "
        "do not announce it ('here it is', 'here's your …'), and do not ask "
        "for a second approval of it. When the result is "
        "unchanged, finish now; never re-present it or repeat a question the "
        "user has already answered. If they ask what a word or type means, "
        "explain it in their own words before you finish. Do not comment on "
        "the quality of what they said."
    )
    if not chat and getattr(_inner_model(return_type), "__name__", "") != "TextReply":
        sections.append(
            "## One-shot derivation\n"
            "This step has no user to ask. Return your best draft from the "
            "inputs. Uncertainty is not a failure: a draft is a proposal the "
            "user confirms or corrects in the next step. When the inputs name "
            "the work but not its steps, propose the steps people usually take "
            "for that work; when they name the work but not its inputs, "
            "propose what people usually start with. Return failure only when "
            "the inputs contain nothing relevant to the result at all."
        )

    return sections


# lucidlint: ignore closures,long-param-list the closures capture one step's config; a class adds no state
def _setup_atomic_workflow(
    system_prompt: str,
    actual_return_type: Any,
    max_turns: int,
    debug: Any,
    session: Any,
    *,
    chat: bool,
    tools: Any = (),
    deps: Any = None,
    min_turns: int = 1,
) -> AtomicWorkflow:
    """Create and configure an AtomicWorkflow on the session's transcript.

    A chat-on step is one turn in the session's conversation, so it starts from
    that transcript. A chat-off step (a greeting, a closing, a one-shot
    generation) has no user conversation of its own: its inputs are its
    parameters, and replaying the session would invite it to answer the
    conversation again.
    """
    conversation = Conversation(
        model=session.config.model,
        provider=session.config.provider,
        instructions=system_prompt,
        output_type=AgentAnswer[actual_return_type],
        final_output_type=FinalAnswer[actual_return_type],
        messages=list(session.state.messages) if chat else [],
        api_base=session.config.api_base,
        api_key_env=session.config.api_key_env,
        output_retries=getattr(session.config, "max_retries", 2),
        transport_retries=getattr(session.config, "max_retries", 2),
        timeout_seconds=session.config.request_timeout_seconds,
        extra_headers=getattr(session.config, "extra_headers", None),
        temperature=getattr(session.config, "temperature", None),
        # Every step reasons; the conversation supplies the gateway's effort.
        thinking=True,
        # A scripted model for tests; production leaves this None and builds
        # the real one from the provider fields above.
        model_client=getattr(session.config, "model_client", None),
        tools=tools,
        deps=deps,
    )

    def _on_success(response: Success[Any]) -> TurnResult[Any]:
        return TurnResult[Any].success(response.result, message=response.message)

    return AtomicWorkflow(
        config=AtomicWorkflowConfig(
            system_prompt=system_prompt,
            response_model=AgentAnswer[actual_return_type],
            final_response_model=FinalAnswer[actual_return_type],
            max_turns=max_turns,
            min_turns=min(min_turns, max_turns),
            initial_messages=None,
            on_continue=lambda response: TurnResult[actual_return_type].continuing(
                response.message
            ),
            on_success=_on_success,
            on_failure=lambda response: AtomicWorkflowFailedError(response.message),
            debug=debug,
            model=session.config.model,
            provider=session.config.provider,
            api_base=session.config.api_base,
            api_key_env=session.config.api_key_env,
            transport_retries=getattr(session.config, "max_retries", 2),
            request_timeout_seconds=session.config.request_timeout_seconds,
            extra_headers=getattr(session.config, "extra_headers", None),
            temperature=getattr(session.config, "temperature", None),
            conversation=conversation,
        ),
        chat=chat,
    )


# lucidlint: ignore closures,long-param-list the closures capture one step's config; a class adds no state
def atomic_workflow(
    func: Callable[..., Any] | None = None,
    *,
    # lucidlint: ignore record-shape the pairs come from the shared rule catalogue
    conversation_validation_rules: list[tuple[str, str]] | None = None,
    chat: bool = True,
    single_turn: bool = False,
    min_turns: int = 1,
    explore: bool = False,
) -> Callable[..., Any]:
    """Auto-orchestrates an LLM atomic workflow using its docstring as system prompt.

    ``session``, ``debug``, ``max_turns``, ``tools`` and ``deps`` are the
    decorator's own parameters: they are removed before the prompt's params
    section is built, so a step cannot also name one of them.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    Accepts a ``session`` parameter (:class:`~chat_workflow.session.Session`).

    ``chat=False`` means the workflow has no user conversation: a ``continue``
    is fed back as an in-context correction and retried within the turn budget.

    ``single_turn=True`` is a one-turn text step: a ``continue`` is not
    offered, so it completes without asking the user.

    ``min_turns`` is the minimum number of model turns the task needs before
    it may return a result. A task that must ask or propose first declares
    two; a success on the first turn is fed back as a correction.

    ``explore=True`` is a listening step: it opens with one open question
    about the user's own work, not a proposal. A step that derives from
    established inputs leaves it false.
    """
    author_rules = conversation_validation_rules or []
    merged_rules: dict[str, str] = dict(_DEFAULT_CONVERSATION_RULES)
    for label, description in author_rules:
        merged_rules[label] = description

    def _decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        raw_func = f.__func__ if isinstance(f, classmethod) else f

        @wraps(f)
        def wrapper(*args, **kwargs):
            session = kwargs.pop("session", None)
            debug = kwargs.pop("debug", None)
            # A step's tools are not prompt parameters: they never render into
            # the params section.
            tools = kwargs.pop("tools", ())
            deps = kwargs.pop("deps", None)

            if session is None:
                raise TypeError(
                    f"Atomic workflow '{f.__name__}' requires 'session' parameter"
                )

            state = session.state
            actual_return_type = resolve_return_type(raw_func, f, args, kwargs)

            if actual_return_type is None or not _is_result_model(actual_return_type):
                raise TypeError(
                    f"@atomic_workflow function '{f.__name__}' return type resolves "
                    f"to '{actual_return_type}' which is not a Pydantic model or "
                    "a list of one"
                )

            system_prompt = _build_system_prompt(
                raw_func,
                kwargs,
                return_type=actual_return_type,
                conversation_rules=merged_rules,
                chat=chat,
                instructions=session.instructions,
            )
            max_turns = kwargs.pop("max_turns", 1 if single_turn else 10)

            if debug is None and session.config.debug:
                debug = StreamingDebug()

            workflow = _setup_atomic_workflow(
                system_prompt,
                actual_return_type,
                max_turns,
                debug,
                session,
                chat=chat,
                tools=tools,
                deps=deps,
                min_turns=min_turns,
            )
            result = session.run(workflow=workflow, chat=chat)
            state.initial_result = result

            if result.result is None:
                raise AtomicWorkflowFailedError(
                    "Atomic workflow completed but no result was produced"
                )

            return result.result

        attached = True
        for marker in ("_is_workflow", "_is_workflow_step"):
            setattr(wrapper, marker, attached)
        wrapper._chat = chat  # type: ignore[attr-defined]  # the step decorator attaches this metadata
        wrapper._single_turn = single_turn  # type: ignore[attr-defined]  # the step decorator attaches this metadata
        wrapper._min_turns = min_turns  # type: ignore[attr-defined]  # the step decorator attaches this metadata
        wrapper._explore = explore  # type: ignore[attr-defined]  # the step decorator attaches this metadata
        wrapper.__conversation_rules__ = merged_rules  # type: ignore[attr-defined]  # the step decorator attaches this metadata
        return wrapper

    if func is not None:
        return _decorator(func)
    return _decorator


def composite_workflow(func: Callable[..., Any]) -> Callable[..., Any]:
    """Passes through to the wrapped function. Caller must provide ``session=``."""

    raw_func = func.__func__ if isinstance(func, classmethod) else func

    @wraps(raw_func)
    def wrapper(*args, **kwargs):
        session = kwargs.get("session")
        if session is not None:
            return raw_func(*args, **kwargs)

        raise TypeError(
            f"Composite workflow '{raw_func.__name__}' requires 'session' parameter"
        )

    attached = True
    for marker in ("_is_workflow", "_is_workflow_step"):
        setattr(wrapper, marker, attached)

    if isinstance(func, classmethod):
        return classmethod(wrapper)
    return wrapper
