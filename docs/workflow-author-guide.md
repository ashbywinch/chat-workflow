# Workflow Author Guide

This guide explains how to build workflows using the chat-workflow library. It covers the decorator API, Pydantic model patterns, and conversation orchestration.

## Decorator API: `@chat` vs `@workflow`

The library provides two decorators for authoring workflow functions:

### `@chat` - Auto-orchestrated leaf functions

Use `@chat` on functions that directly interact with the LLM. The function body is a `pass` stub, and the decorator handles everything.

```python
from chat_workflow import chat

@chat
def my_workflow_step(
    context: str = "",
    max_turns: int = 10,
) -> MyModel:
    """System prompt for the LLM goes here. {param} interpolation works."""
    pass
```

**How it works:**
1. Inspects the return type (must be a Pydantic `BaseModel`)
2. Uses the docstring as the system prompt and appends an auto-generated `## Parameters` section (all params with types, descriptions, and runtime values)
3. Wraps the return type in `ConversationAction[ReturnType]` as the LLM response model
4. Creates a `StructuredConversationOrchestrator` with default callbacks
5. Runs the multi-turn conversation via `ConversationTools.chat()`
6. Returns the inner Pydantic object

**Required parameters:** `io` or `tools` - these provide the I/O interface for user interaction.

### `@workflow` - Composite functions

Use `@workflow` on functions that compose multiple `@chat` steps. It injects a `ConversationTools` object so the function can pass `tools=tools` to child functions.

```python
from chat_workflow import workflow, ConversationTools

@workflow
def composite_step(
    context: str = "",
    max_turns: int = 10,
    tools: ConversationTools,
) -> MyModel:
    result = my_workflow_step(context=context, max_turns=max_turns, tools=tools)
    # ... additional logic ...
    return result
```

**How it works:**
1. If `tools` is already provided (e.g., called from another `@workflow`), passes through
2. Otherwise, creates a `ConversationTools` from the `io` and `state` parameters
3. Calls the function body with `tools` injected

**CLI auto-discovery**: Functions decorated with `@workflow` are automatically discovered by the CLI. Their parameters (excluding `tools`, `io`, `state`, `debug`) become CLI options. The function name is converted to kebab-case for the command name (e.g., `generate_reviewed_criteria` → `generate-reviewed-criteria`).

## Pydantic Model Patterns

### Business Rules in Models, Not Prompts

Business rules live in Pydantic models (`model_validator`), not prompts. Prompts give behavioral guidance only, and Instructor handles schema formatting.

### Communicate Rules in JSON Schema

A `model_validator` only fires *after* the LLM returns data. It's enforcement, not communication. The LLM reads the JSON schema (appended by Instructor to the system message) to understand what to produce. So rules must be encoded in ways that propagate to `model_json_schema()`:

- Use Pydantic field constraints (`min_length` → `minItems`, `ge`/`le` → `minimum`/`maximum`). These inject directly into the schema.
- Use `Field(description=...)` with plain-English conditional rules (e.g. `'Required when action is "success". Must be null when action is "continue".'`).
- Use class docstrings. Pydantic v2 emits them as the model's `"description"` in JSON schema.
- Use `model_config = dict(json_schema_extra=...)` for non-standard but model-level annotations the LLM should see.
- Test schemas with `Model.model_json_schema()` and verify each rule is visible in the output.

### Example Model

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Generic, TypeVar

TResult = TypeVar("TResult")

class ConversationAction(BaseModel, Generic[TResult]):
    action: Literal["continue", "success", "failure"]
    message: str | None = Field(
        default=None,
        description='Message for the user. Required when action is "continue" or "failure". Must be null when action is "success".',
    )
    result: TResult | None = Field(
        default=None,
        description='The result object. Required when action is "success". Must be null when action is "continue" or "failure".',
    )

    @model_validator(mode="after")
    def validate_action_consistency(self):
        if self.action == "continue":
            if not self.message:
                raise ValueError(
                    "continue action requires a message field with your question for the user. "
                    "Do not include a result field."
                )
            if self.result is not None:
                raise ValueError(
                    "continue action cannot include result. "
                    "Use action='success' if you have complete result to return."
                )
        elif self.action == "failure":
            if not self.message:
                raise ValueError(
                    "failure action requires a message field explaining why."
                )
            if self.result is not None:
                raise ValueError("failure action cannot include result.")
        elif self.action == "success":
            if self.result is None:
                raise ValueError(
                    "success action requires a result field with the complete result."
                )
        return self
```

## Parameter Descriptions via `Annotated`

Use `typing.Annotated[T, "description"]` to add descriptions that appear in the auto-generated `## Parameters` section of the system prompt:

```python
from typing import Annotated

@chat
def my_workflow_step(
    context: Annotated[
        str, "The topic or domain for which to generate data"
    ] = "",
    max_turns: Annotated[
        int, "Maximum number of conversation turns before giving up"
    ] = 10,
) -> MyModel:
    ...
```

This produces `## Parameters` entries like:
```
- `context` (str): The topic or domain for which to generate data
  Value: "example topic"
- `max_turns` (int): Maximum number of conversation turns before giving up
  Value: 10
```

## Docstring Interpolation

The docstring supports:
- `{param_name}` - simple parameter substitution
- `{param.method()}` - method calls on parameter values (e.g., `{initial_object.model_dump()}`)

### Example

```python
@chat
def refine(
    initial_object: Annotated[MyModel, "The object to review"],
    max_turns: Annotated[int, "Maximum refinement turns"] = 5,
) -> MyModel:
    """Review and refine the provided object.

    The user has provided this initial object:
    {initial_object.model_dump()}

    Please help them improve it through conversation."""
    pass
```

## Generic Refinement with TypeVar

Use `TypeVar` to create generic refinement functions that work with any Pydantic model:

```python
from typing import TypeVar
from pydantic import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)

@chat
def refine(
    initial_object: Annotated[ModelType, "The object to review"],
    max_turns: Annotated[int, "Maximum refinement turns"] = 5,
) -> ModelType:
    """System prompt for refinement..."""
    pass
```

The `@chat` decorator resolves the `TypeVar` to the concrete type passed as `initial_object` at runtime.

## ConversationIO Protocol

All `@chat` functions require an I/O adapter implementing:

```python
class ConversationIO(Protocol):
    def echo(self, message: str) -> None: ...   # Display message to user
    def prompt(self, label: str) -> str: ...     # Get input from user
```

The CLI provides `TyperConversationIO` (using `typer.echo`/`typer.prompt`). For tests, use `unittest.mock.Mock()`.

### Custom I/O Implementation

```python
from chat_workflow import ConversationTools, ConversationFlowState, ConversationIO

class MyIO(ConversationIO):
    def echo(self, message: str) -> None:
        print(f"Assistant: {message}")
    
    def prompt(self, label: str) -> str:
        return input(f"{label}: ")

# Use in workflow
result = my_workflow_step(
    context="example",
    max_turns=10,
    tools=ConversationTools(io=MyIO(), state=ConversationFlowState()),
)
```

## Complete Example

Here's a complete example showing all patterns together:

```python
from typing import Annotated, TypeVar
from pydantic import BaseModel, Field
from chat_workflow import chat, workflow, ConversationTools

# Define a simple model
class MyData(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10)
    priority: int = Field(..., ge=1, le=5)

# Leaf chat function
@chat
def generate_data(
    topic: Annotated[str, "Topic for data generation"] = "",
    max_turns: Annotated[int, "Maximum conversation turns"] = 10,
) -> MyData:
    """Help the user create structured data about {topic}.
    
    Guide them through defining title, description, and priority.
    Use clarifying questions to ensure quality data."""
    pass

# Generic refinement function
ModelType = TypeVar("ModelType", bound=BaseModel)

@chat
def refine_data(
    initial_data: Annotated[ModelType, "Data to refine"],
    max_turns: Annotated[int, "Refinement turns"] = 5,
) -> ModelType:
    """Review and improve the provided data.
    
    Initial data: {initial_data.model_dump()}
    
    Ask questions to help the user enhance clarity, completeness, and quality."""
    pass

# Composite workflow
@workflow
def create_and_refine_data(
    topic: str = "",
    max_turns: int = 10,
    tools: ConversationTools,
) -> MyData:
    # Generate initial data
    data = generate_data(topic=topic, max_turns=max_turns, tools=tools)
    
    # Refine it
    refined_data = refine_data(initial_data=data, max_turns=5, tools=tools)
    
    return refined_data
```

## Debugging LLM Interactions

When evals hang or behave unexpectedly, enable debug tracing with an environment variable:

```bash
CHAT_WORKFLOW_DEBUG=1 python -m pytest tests/your_test.py
```

This streams all LLM requests/responses to stderr with timing:
```
[15:44:16.001] ━━━ LLM REQUEST ━━━
[15:44:16.001] Model: openrouter/google/gemini-2.0-flash-lite-001
[15:44:16.001] [0] system: You are a helpful assistant...
[15:44:16.001] Waiting for response...
[15:44:17.234] ━━━ LLM RESPONSE (1233ms) ━━━
```

## Next Steps

See [example-evaluation-criteria.md](example-evaluation-criteria.md) for a complete worked example using these patterns to build an evaluation criteria workflow.