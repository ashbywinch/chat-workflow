# Workflow Author Guide

This guide explains how to build workflows using the chat-workflow library. It covers the decorator API, Pydantic model patterns, and conversation orchestration.

## Decorator API: `@chat` vs `@workflow`

The library provides two decorators for authoring workflow functions:

### `@chat` - Auto-orchestrated leaf functions

Use `@chat` on functions that directly interact with the LLM. The function body is a `pass` stub, and the decorator handles everything.

```python
from chat_workflow import chat

@chat
def generate_essay_from_topic(
    topic: Topic
) -> Essay:
    """You are a helpful essay writing tutor. Facilitate the user in writing a great essay on the given topic. Help them develop their essay writing skills as they go."""
    pass
```

**How it works:**
1. Uses the docstring as a prompt 
2. Runs a multi-turn conversation, including validating the object returned by the LLM and retrying if it's invalid
3. Provides the parameters and details of the expected return type (including validation) to the LLM under the hood
4. Returns the object that the LLM provides, or raises an exception if the LLM was unable to create a valid object.

**Required parameters:** `io` or `tools` - these provide the I/O interface for user interaction.

### `@workflow` - Composite functions

Use `@workflow` on functions that compose multiple `@chat` steps. It injects a `ConversationTools` object so the function can pass `tools=tools` to child functions.

```python
from chat_workflow import workflow, ConversationTools

@chat
def generate_topic(tools:ConversationTools) -> Topic:
    """You are a helpful careers advisor. Help the user (a student) think up a good essay topic for an essay that will be part of their application for a course of some kind. You'll need to ask the user questions to determine what course they're applying for and what background they have that might feed into the essay topic"""

@workflow
def generate_essay(
    tools: ConversationTools,
) -> Essay:
    topic = generate_topic(tools)
    return generate_essay_from_topic(topic, tools)
```

**CLI auto-discovery**: Functions decorated with `@workflow` and exported from the module are automatically discovered by the CLI. Their parameters (excluding `tools`, `io`, `state`, `debug`) become CLI options. The function name is converted to kebab-case for the command name (e.g., `generate_essay` → `generate-essay`).

## Pydantic Model Patterns

### Business Rules in Models, Not Prompts

Business rules live in Pydantic models (`model_validator`), and don't need to be duplicated in the prompt. 

### Communicate Rules in JSON Schema

A `model_validator` only fires *after* the LLM returns data. It's enforcement, not communication. The JSON schema for the model, on the other hand, is sent to the LLM up front to understand what to produce. So rules should be encoded in ways that propagate to `model_json_schema()`:

#### Visible to the LLM and verified automatically:

- Use Pydantic field constraints (`min_length` → `minItems`, `ge`/`le` → `minimum`/`maximum`). This is the best option wherever possible.

#### Visible to the LLM but not verified automatically:
- Use `Field(description=...)` with plain-English conditional rules (e.g. `'Required when action is "success". Must be null when action is "continue".'`). 
- Use class docstrings for class-level validation. These will appear in the JSON schema. 

You'll need to write a `model_validator` if you want these rules to be validated programmatically.

You can test schemas with `Model.model_json_schema()` to verify that your rules are all visible in the output.

### Example Model

This example is from the internals of workflow-chat - it's the model that all LLM responses get wrapped in to support multi turn chat functionality.

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

Use `typing.Annotated[T, "description"]` to add descriptions that appear in the auto-generated `## Parameters` section of your prompt:

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

## Generic Refinement with TypeVar

Use `TypeVar` to create generic chat functions that work with any Pydantic model type:

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
    """Review and improve the provided object.
    
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
