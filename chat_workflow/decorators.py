"""Decorators for chat-workflow: @atomic_workflow_step and @composite_workflow_step."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .annotations import user_labels
from .atomic_workflow_step import AtomicWorkflowStep
from .atomic_workflow_step_config import AtomicWorkflowStepConfig
from .conversation_rules import DEFAULT_RULES as _DEFAULT_CONVERSATION_RULES
from .models import AgentResponse, TurnResult
from .prompt_builder import _build_params_section, _format_docstring
from .types_meta import resolve_return_type


def _return_type_name(return_type: Any) -> str:
    """Render a return type for the task section (handles list[Model])."""
    origin = get_origin(return_type)
    if origin is list:
        args = get_args(return_type)
        if args:
            return f"list[{_return_type_name(args[0])}]"
    return getattr(return_type, "__name__", str(return_type))


def _inner_model(return_type: Any) -> Any:
    """Return the result model behind a return type, unwrapping list[Model]."""
    origin = get_origin(return_type)
    if origin is list:
        args = get_args(return_type)
        if args:
            return args[0]
    return return_type


def _build_task_section(return_type: Any, chat: bool) -> str:
    name = _return_type_name(return_type)
    if chat:
        return (
            "## Task\n"
            "Help the user through the conversation below. The JSON schema defines "
            "the output the framework needs; it is for you, not for the user. "
            "Return that JSON object every turn: put the user-facing text in "
            "`message`, set `intent` to continue, success, or failure, and never "
            "reply with plain text. "
            "Your input parameters are what the user already told you in earlier "
            "phases. Use them as established facts; do not ask them to repeat or "
            "re-confirm the same picture. Ask only about what the inputs do not "
            "cover. "
            "If the conversation is empty, open with one short greeting and one "
            "sentence on what you are here to help with, then ask your first "
            "question; never return success before the user has said anything. "
            "At the start of a step, say in one sentence what you are doing and "
            "why, for example 'Let me check what I have understood so far.' "
            "Act as a thinking partner, not a form: propose concrete ideas grounded "
            "in what the user said and ask for a reaction rather than interrogating "
            "them item by item. "
            "Acknowledge an off-topic remark in one clause without praising or "
            "evaluating it, then return to the question at hand; never develop "
            "the user's domain content. "
            "Ask each question once. Once the user has answered it, accept the "
            "answer and move on; never ask the same substantive question again. "
            "If the user pushes back on a question or says 'you tell me', never "
            "ask it again; offer a concrete suggestion and move on. "
            "Match the user's register: if they speak plainly, speak plainly; if "
            "they use specialist terms, use theirs. Never introduce a term for the "
            "output shape, a field, or the framework. This includes 'deliverable', "
            "'output', 'field', 'schema', 'workflow', and 'component'. If the user "
            "uses one of those words, use their word. If they say a word is "
            "confusing, never use that word again in the conversation — re-ask in "
            "their words. Never present the output's fields as a labelled list or "
            "template; write a short paragraph in the user's words. Use the "
            "language the user is using: "
            "if they write in English, reply in English; never switch languages on "
            "your own. Before the user has named something, "
            "use a plain placeholder ('the thing you make', 'your work'); once they "
            "name it, use their word. "
            "When you show a draft, describe it in the user's words; never label "
            "parts with field names or print the schema. "
            "Use continue only while you still need something from the user. When "
            "they agree with your description of the whole picture, return the "
            "result in that same response — never re-present an unchanged draft or "
            "ask for a second approval. When you check a draft, ask only 'Is this "
            "right?' or 'Anything you want to change?'; never invent new phrasing "
            "for the step you are in. When the user asks what you think or asks "
            "for a suggestion, answer with specific ideas and improvements; do "
            "not ask for approval again in that same turn. "
            "Before using failure, propose the thing in the user's own words; "
            "use failure only if they reject every concrete proposal."
        )
    return (
        "## Task\n"
        f"Produce a `{name}` from the parameters below. "
        "The JSON schema below defines the expected output; field descriptions "
        "and constraints communicate the rules your response must satisfy. "
        "Use success when the output is complete; use failure if it cannot be produced."
    )


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
        "These describe the result you return. They are not a request to "
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


def _build_user_vocabulary_section(return_type: Any) -> str:
    """Map internal field names to the words the user should hear.

    The result schema is the model's vocabulary for the artifact, and it tends
    to leak field names into ``message``. These labels are declared next to
    the fields; the user's own word still wins over the phrase.
    """
    labels = user_labels(_inner_model(return_type))
    if not labels:
        return ""
    lines = [
        "## Use the user's words",
        "These are internal labels for the result. Never use a label, its "
        "singular, or a variant of it in `message` unless the user used it "
        "first. Say the phrase, or the user's own word if they have one.",
    ]
    lines.extend(f'- `{name}` -> "{label.phrase}"' for name, label in labels.items())
    return "\n".join(lines)


def _build_system_prompt(
    raw_func: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    return_type: Any,
    conversation_rules: dict[str, str],
    chat: bool,
) -> str:
    """Build the system prompt from the task, skill, checklists and parameters."""
    sections: list[str] = []

    sections.append(_build_task_section(return_type, chat))

    vocabulary_section = _build_user_vocabulary_section(return_type)
    if vocabulary_section:
        sections.append(vocabulary_section)

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

    return "\n\n".join(sections)


def _setup_atomic_workflow_step(
    system_prompt: str,
    actual_return_type: type[BaseModel],
    max_turns: int,
    debug: Any,
    session: Any,
    conversation_validation_rules: dict[str, str] | None = None,
    chat: bool = True,
) -> AtomicWorkflowStep:
    """Create and configure an AtomicWorkflowStep."""
    from .exceptions import AtomicWorkflowStepFailedError

    return AtomicWorkflowStep(
        config=AtomicWorkflowStepConfig(
            system_prompt=system_prompt,
            response_model=AgentResponse[actual_return_type],
            max_turns=max_turns,
            initial_messages=None,
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
        )
    )


def atomic_workflow_step(
    func: Callable[..., Any] | None = None,
    *,
    conversation_validation_rules: list[tuple[str, str]] | None = None,
    chat: bool = True,
) -> Callable[..., Any]:
    """Auto-orchestrates an LLM atomic workflow step using its docstring as system prompt.

    Return type must be a Pydantic model. Docstring supports {param} interpolation.
    Accepts a ``session`` parameter (:class:`~chat_workflow.session.Session`).

    ``chat=False`` means the workflow has no user conversation: it makes one
    LLM call with the parameters and no user prompts. A ``continue`` intent is
    treated as an in-context error; the framework appends a correction and
    retries within the turn budget.
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
            )
            max_turns = kwargs.pop("max_turns", 10)

            workflow_step = _setup_atomic_workflow_step(
                system_prompt, actual_return_type, max_turns, debug, session,
                conversation_validation_rules=merged_rules,
                chat=chat,
            )
            result = session.run(workflow_step=workflow_step, first_user_input="")
            state.initial_result = result

            if result.result is None:
                raise AtomicWorkflowStepFailedError("Atomic workflow step completed but no result was produced")

            return result.result

        wrapper._is_workflow_step = True  # pyright: ignore[reportAttributeAccessIssue]
        wrapper._chat = chat  # pyright: ignore[reportAttributeAccessIssue]
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
