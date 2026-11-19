"""Decorators for chat-workflow: @atomic_workflow_step and @composite_workflow_step."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .atomic_workflow_step import AtomicWorkflowStep
from .atomic_workflow_step_config import AtomicWorkflowStepConfig
from .conversation_rules import DEFAULT_RULES as _DEFAULT_CONVERSATION_RULES
from .models import AgentResponse, TurnResult
from .prompt_builder import (
    _build_params_section,
    _format_docstring,
    input_models,
    user_parameter_labels,
)
from .types_meta import resolve_return_type


def _inner_model(return_type: Any) -> Any:
    """Return the result model behind a return type, unwrapping list[Model]."""
    origin = get_origin(return_type)
    if origin is list:
        args = get_args(return_type)
        if args:
            return args[0]
    return return_type


def _build_checklist_section(rules: dict[str, str]) -> str:
    lines = ["## Conversation checklist", "Apply these to your own behaviour:"]
    for label, description in rules.items():
        lines.append(f"- **{label}**: {description}")
    return "\n".join(lines)


def _build_validation_rules_section(return_type: Any) -> str:
    """Put the result model's class-level validation rules in the prompt.

    Field-level ``Validation(...)`` rules are already in the JSON schema (after
    ``LLMValidated`` rebuilds it); this covers the class-level rules.
    """
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
        schema = AgentResponse[return_type].model_json_schema()
    except Exception:
        return frozenset()
    return frozenset(schema.get("$defs", {}))


def _build_user_vocabulary_section(
    return_type: Any,
    inputs: tuple[type[BaseModel], ...] = (),
    parameter_labels: dict[str, str] | None = None,
) -> str:
    """Tell the model which names are internal and where each phrase lives.

    Result and input fields already carry their phrase as the field's schema
    title (``UserLabel`` is the one declaration; ``LLMValidated`` puts it in
    the schema). The two names that have no field to live on are a parameter
    name and a model type name; this section maps the first to the parameter
    description and points at the thing itself for the second.
    """
    parameter_labels = parameter_labels or {}
    type_names = [
        model.__name__
        for model in (*inputs, _inner_model(return_type))
        if isinstance(model, type)
        and issubclass(model, BaseModel)
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


def _build_system_prompt(
    raw_func: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    return_type: Any,
    conversation_rules: dict[str, str],
    chat: bool,
    instructions: str | None = None,
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

    docstring = _format_docstring(raw_func.__doc__, **kwargs).strip()
    if docstring:
        sections.append(docstring)

    params_section = _build_params_section(
        raw_func,
        kwargs,
        skip_shapes=_output_defs(return_type),
    )
    if params_section:
        sections.append(params_section.strip())

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

    return "\n\n".join(sections)


def _setup_atomic_workflow_step(
    system_prompt: str,
    actual_return_type: type[BaseModel],
    max_turns: int,
    debug: Any,
    session: Any,
    initial_messages: list[dict[str, str]] | None = None,
    conversation_validation_rules: dict[str, str] | None = None,
    chat: bool = True,
    single_turn: bool = False,
) -> AtomicWorkflowStep:
    """Create and configure an AtomicWorkflowStep."""
    from .exceptions import AtomicWorkflowStepFailedError

    return AtomicWorkflowStep(
        config=AtomicWorkflowStepConfig(
            system_prompt=system_prompt,
            response_model=AgentResponse[actual_return_type],
            max_turns=max_turns,
            initial_messages=initial_messages,
            on_continue=lambda response: TurnResult[actual_return_type].continuing(response.message or ""),
            on_success=lambda response: TurnResult[actual_return_type].success(
                response.result,
                message="Completed successfully!",
            ),
            on_failure=lambda response: AtomicWorkflowStepFailedError(response.message or "No reason given"),
            debug=debug,
            model=session.config.model,
            provider=session.config.provider,
            api_base=session.config.api_base,
            api_key_env=session.config.api_key_env,
            max_retries=session.config.max_retries,
            request_timeout_seconds=session.config.request_timeout_seconds,
            model_supports_tools=session.config.model_supports_tools,
            extra_headers=getattr(session.config, "extra_headers", None),
            conversation_validation_rules=conversation_validation_rules,
            chat=chat,
            single_turn=single_turn,
        )
    )


def atomic_workflow_step(
    func: Callable[..., Any] | None = None,
    *,
    conversation_validation_rules: list[tuple[str, str]] | None = None,
    chat: bool = True,
    single_turn: bool = False,
) -> Callable[..., Any]:
    """Auto-orchestrates an LLM atomic workflow step using its docstring as system prompt.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    Accepts a ``session`` parameter (:class:`~chat_workflow.session.Session`).

    ``chat=False`` means the workflow has no user conversation: it makes one
    LLM call with the parameters and no user prompts. A ``continue`` intent is
    treated as an in-context error; the framework appends a correction and
    retries within the turn budget.

    ``single_turn=True`` is a one-turn text step: a ``continue`` message is
    the step's result, so it completes without asking the user. It is how the
    session's greeting and closing are produced.
    """
    from .exceptions import AtomicWorkflowStepFailedError

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

            if session is None:
                raise TypeError(f"Atomic workflow step '{f.__name__}' requires 'session' parameter")

            state = session.state
            actual_return_type = resolve_return_type(raw_func, f, args, kwargs)

            if actual_return_type is not None:
                origin = get_origin(actual_return_type)
                if origin is list:
                    inner_type = get_args(actual_return_type)[0]
                    if not (isinstance(inner_type, type) and issubclass(inner_type, BaseModel)):
                        raise TypeError(
                            f"@atomic_workflow_step function '{f.__name__}' list return type "
                            f"resolves to list[{inner_type}] which is not a Pydantic model subclass"
                        )

            system_prompt = _build_system_prompt(
                raw_func,
                kwargs,
                return_type=actual_return_type,
                conversation_rules=merged_rules,
                chat=chat,
                instructions=session.instructions,
            )
            max_turns = kwargs.pop("max_turns", 10)

            workflow_step = _setup_atomic_workflow_step(
                system_prompt, actual_return_type, max_turns, debug, session,
                initial_messages=list(state.messages),
                conversation_validation_rules=merged_rules,
                chat=chat,
                single_turn=single_turn,
            )
            result = session.run(workflow_step=workflow_step, first_user_input="")
            state.initial_result = result

            if result.result is None:
                raise AtomicWorkflowStepFailedError("Atomic workflow step completed but no result was produced")

            return result.result

        wrapper._is_workflow_step = True  # pyright: ignore[reportAttributeAccessIssue]
        wrapper._chat = chat  # pyright: ignore[reportAttributeAccessIssue]
        wrapper._single_turn = single_turn  # pyright: ignore[reportAttributeAccessIssue]
        wrapper.__conversation_rules__ = merged_rules  # pyright: ignore[reportAttributeAccessIssue]

        return wrapper

    if func is not None:
        return _decorator(func)
    return _decorator


def composite_workflow_step(func: Callable[..., Any]) -> Callable[..., Any]:
    """Passes through to the wrapped function. Caller must provide ``session=``.

    Handles ``@classmethod`` — if the decorated function is a classmethod
    descriptor, the underlying function is extracted, wrapped, and re-wrapped
    as a ``classmethod`` so that ``cls`` is automatically prepended on call.
    """

    raw_func = func.__func__ if isinstance(func, classmethod) else func

    @wraps(raw_func)
    def wrapper(*args, **kwargs):
        session = kwargs.get("session")
        if session is not None:
            return raw_func(*args, **kwargs)

        raise TypeError(f"Composite workflow step '{raw_func.__name__}' requires 'session' parameter")

    wrapper._is_workflow_step = True  # pyright: ignore[reportAttributeAccessIssue]

    if isinstance(func, classmethod):
        return classmethod(wrapper)
    return wrapper
