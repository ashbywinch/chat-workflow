# Workflow Author Guide

This guide explains how to build workflows using the chat-workflow library. It covers the decorator API, Pydantic model patterns, and conversation orchestration.

## Decorator API: `@atomic_workflow` vs `@composite_workflow`

The library provides two decorators for authoring workflow functions:

### `@atomic_workflow` - Auto-orchestrated leaf functions

Use `@atomic_workflow` on functions that directly interact with the LLM. The function body is a `pass` stub, and the decorator handles everything.

```python
from chat_workflow import atomic_workflow

@atomic_workflow
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

**Required parameters:** `session` - provides the I/O interface for user interaction.

### `@composite_workflow` - Composite functions

Use `@composite_workflow` on functions that compose multiple `@atomic_workflow` steps. It injects a `Session` object so the function can pass `session=session` to child functions.

```python
from chat_workflow import composite_workflow, Session

@atomic_workflow
def generate_topic(session: Session) -> Topic:
    """You are a helpful careers advisor. Help the user (a student) think up a good essay topic for an essay that will be part of their application for a course of some kind. You'll need to ask the user questions to determine what course they're applying for and what background they have that might feed into the essay topic"""

@composite_workflow
def generate_essay(
    session: Session,
) -> Essay:
    topic = generate_topic(session)
    return generate_essay_from_topic(topic, session)
```

**CLI auto-discovery**: Functions decorated with `@composite_workflow` and exported from the module are automatically discovered by the CLI. Their parameters (excluding `session`, `io`, `state`, `debug`) become CLI options. The function name is converted to kebab-case for the command name (e.g., `generate_essay` → `generate-essay`).

## Pydantic Model Patterns

### Good prompts 

DO give behavioral examples and conversation strategies in your prompt

FOCUS on "what to do", not "what not to do"

### Business Rules in Models, Not Prompts

Business rules live in your Pydantic models, from where they are automatically added to the system prompt when you write a @atomic_workflow function that returns a model. You shouldn't need to add business rules to prompts that you write.

### Communicate Rules in JSON Schema

You can add `model_validators` to your Pydantic model. These will be used to verify what the LLM returns, but they aren't enough get the rules automatically added to your prompts. If the LLM repeatedly fails to satisfy your business rule, the rule is not visible enough in the model. Here's how to fix the model:

#### Best option if you can: Pydantic field constraints.

- Use Pydantic field constraints (`min_length` → `minItems`, `ge`/`le` → `minimum`/`maximum`). These are visible in your prompt and will be verified automatically when the LLM returns an object.

#### Second best option: Docstrings and descriptions
- Use `Field(description=...)` with plain-English conditional rules (e.g. `'Required when intent is SUCCESS. Must be null when intent is CONTINUE.'`). 
- Use class docstrings for class-level validation. These will appear in the JSON schema. 
- Write a `model_validator` that validates the same rules programmatically. Your validator will be called when the LLM returns an object.

You can test schemas with `Model.model_json_schema()` to verify that your rules are all visible in the output.

### SOLID/DRY Principles for Workflow Authors

These principles help you design maintainable, composable workflows.

#### Workflow Classes Are Rich Classes

Pydantic models can carry convenience methods. This is Pythonic and idiomatic. A model that validates data can also provide methods that operate on that data.

```python
class EvaluationCriteria(BaseModel):
    criteria: list[Criterion] = []

    def add_criterion(self, name: str, description: str, weight: float = 1.0) -> None:
        self.criteria.append(Criterion(name=name, description=description, weight=weight))

    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)

    def normalized_weights(self) -> list[float]:
        total = self.total_weight()
        return [c.weight / total for c in self.criteria]
```

Don't extract every method into a service class just for purity. A thin convenience method on the model itself is often clearer than a separate builder class.

#### Business Rules Live in Models, Not Prompts

Business rules go in Pydantic `model_validator` methods and field constraints. Prompts give behavioral guidance only. This separation means:

- Rules are enforced programmatically, not by hoping the LLM follows instructions
- Rules appear in the JSON schema that Instructor sends to the LLM
- Rules are testable with unit tests (no API key needed)
- Prompts stay focused on conversation strategy

#### Keep Prompts Focused on Behavioral Guidance

A prompt should tell the LLM how to behave, not what data format to produce. Instructor handles schema formatting. Your prompt should cover:

- The role the LLM should adopt
- Conversation strategy and approach
- How to interact with the user
- What to do when information is incomplete

#### Compose Workflows from Small, Single-Purpose Functions

Each `@atomic_workflow` function should do one thing well. Compose them with `@composite_workflow` functions.

```python
@atomic_workflow
def gather_requirements(session: Session) -> Requirements:
    """Help the user articulate their requirements through guided questions."""
    pass

@atomic_workflow
def generate_specification(
    requirements: Requirements, session: Session
) -> Specification:
    """Transform requirements into a structured specification."""
    pass

@composite_workflow
def build_specification(
    topic: str = "", session: Session
) -> Specification:
    reqs = gather_requirements(session)
    return generate_specification(reqs, session)
```

This makes each step testable in isolation and reusable across workflows.

### Example Model

This example is from the internals of chat-workflow - it's the model that all LLM responses get wrapped in to support multi turn chat functionality.

```python
from pydantic import BaseModel, Field, model_validator
from typing import Generic, TypeVar
from chat_workflow import AgentIntent

TResult = TypeVar("TResult")

class AgentResponse(BaseModel, Generic[TResult]):
    intent: AgentIntent
    message: str | None = Field(
        default=None,
        description='Message for the user. Required when intent is CONTINUE or FAILURE. Must be null when intent is SUCCESS.',
    )
    result: TResult | None = Field(
        default=None,
        description='The result object. Required when intent is SUCCESS. Must be null when intent is CONTINUE or FAILURE.',
    )

    @model_validator(mode="after")
    def validate_intent_consistency(self):
        if self.intent == AgentIntent.CONTINUE:
            if not self.message:
                raise ValueError(
                    "CONTINUE intent requires a message field with your question for the user. "
                    "Do not include a result field."
                )
            if self.result is not None:
                raise ValueError(
                    "CONTINUE intent cannot include result. "
                    "Use intent=AgentIntent.SUCCESS if you have complete result to return."
                )
        elif self.intent == AgentIntent.FAILURE:
            if not self.message:
                raise ValueError(
                    "FAILURE intent requires a message field explaining why."
                )
            if self.result is not None:
                raise ValueError("FAILURE intent cannot include result.")
        elif self.intent == AgentIntent.SUCCESS:
            if self.result is None:
                raise ValueError(
                    "SUCCESS intent requires a result field with the complete result."
                )
        return self
```

## Parameter Descriptions via `Annotated`

Use `typing.Annotated[T, "description"]` to add descriptions that appear in the auto-generated `## Parameters` section of your prompt:

```python
from typing import Annotated

@atomic_workflow
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

@atomic_workflow
def refine(
    initial_object: Annotated[ModelType, "The object to review"],
    max_turns: Annotated[int, "Maximum refinement turns"] = 5,
) -> ModelType:
    """System prompt for refinement..."""
    pass
```

The `@atomic_workflow` decorator resolves the `TypeVar` to the concrete type passed as `initial_object` at runtime.

## UserIO Base Class

All `@atomic_workflow` functions require an I/O adapter subclassing `UserIO`:

```python
class UserIO:
    def echo(self, message: str) -> None: ...   # Display message to user
    def prompt(self, label: str) -> str: ...     # Get input from user
```

The CLI provides `TyperUserIO` (using `typer.echo`/`typer.prompt`). For tests, use `unittest.mock.Mock()`.

### Custom I/O Implementation

```python
from chat_workflow import Session, SessionLog, UserIO

class MyIO(UserIO):
    def echo(self, message: str) -> None:
        print(f"Assistant: {message}")
    
    def prompt(self, label: str) -> str:
        return input(f"{label}: ")

# Use in workflow
result = my_workflow_step(
    context="example",
    max_turns=10,
    session=Session(io=MyIO(), state=SessionLog()),
)
```

## Complete Example

Here's a complete example showing all patterns together:

```python
from typing import Annotated, TypeVar
from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow, composite_workflow, Session

# Define a simple model
class MyData(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10)
    priority: int = Field(..., ge=1, le=5)

# Leaf atomic_workflow function
@atomic_workflow
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

@atomic_workflow
def refine_data(
    initial_data: Annotated[ModelType, "Data to refine"],
    max_turns: Annotated[int, "Refinement turns"] = 5,
) -> ModelType:
    """Review and improve the provided object.
    
    Ask questions to help the user enhance clarity, completeness, and quality."""
    pass

# Composite workflow
@composite_workflow
def create_and_refine_data(
    topic: str = "",
    max_turns: int = 10,
    session: Session,
) -> MyData:
    # Generate initial data
    data = generate_data(topic=topic, max_turns=max_turns, session=session)
    
    # Refine it
    refined_data = refine_data(initial_data=data, max_turns=5, session=session)
    
    return refined_data
```

For debugging LLM interactions, see the [Contributor Guide](contributor-guide.md#debugging-llm-interactions).

## Next Steps

See [example-evaluation-criteria.md](example-evaluation-criteria.md) for a complete worked example using these patterns to build an evaluation criteria workflow.

## Annotations & Mixins

The library provides two annotations (`Blob`, `Validation`) for tagging model fields with metadata, and two mixins (`BlobSyncMixin`, `LLMValidated`) that act on those annotations to provide file materialization and natural-language validation.

Both mixins are independent: you can use one, both, or neither in any model.

### Blob Annotation

The `Blob` annotation marks a field's content for materialization to a file on disk. Use it with `typing.Annotated` on string fields.

```python
from typing import Annotated
from pydantic import BaseModel
from chat_workflow.annotations import Blob

class DiagramModel(BaseModel):
    title: str
    diagram_code: Annotated[str, Blob(".mmd")] = ""
```

When combined with `BlobSyncMixin`, the field value is written to `{output_dir}/{field_name}{extension}`.

The `Blob` annotation constructor accepts a single `extension` parameter (default `".txt"`) that controls the file suffix.

### Validation Annotation

The `Validation` annotation attaches a natural-language validation rule to a field. The rule is expressed as a plain string.

```python
from typing import Annotated
from pydantic import BaseModel
from chat_workflow.annotations import Validation

class TeamPlan(BaseModel):
    members: Annotated[str, Validation("Must list at least 3 members")]
    timeline: Annotated[str, Validation("Must include specific dates or milestones")]
```

When combined with `LLMValidated`, all rules are verified by an LLM call during model validation. Rules are also automatically injected into the field's JSON schema description.

### BlobSyncMixin

`BlobSyncMixin` automatically materializes `Blob`-annotated fields to files on disk. Add it to your Pydantic model alongside `Blob` annotations.

```python
from pathlib import Path
from typing import Annotated
from pydantic import BaseModel
from chat_workflow.annotations import Blob
from chat_workflow.mixins import BlobSyncMixin

class ArchitectureModel(BlobSyncMixin):
    title: str
    diagram: Annotated[str, Blob(".mmd")] = ""
    config: Annotated[str, Blob(".yaml")] = ""

# Usage
model = ArchitectureModel(title="My App", diagram="graph TD...", config="key: value")
model.materialize_blobs(Path("/tmp/output"))

# Get paths for individual fields
diagram_path = model.get_blob_path("diagram")
config_path = model.get_blob_path("config")
```

The mixin provides:

- `materialize_blobs(output_dir: Path) -> BlobSyncMixin` -- writes all `Blob` fields to `{output_dir}/{field_name}{extension}`. Creates `output_dir` if needed. Returns `self` for chaining.
- `get_blob_path(field: str) -> Path | None` -- returns the filesystem path where a field was materialized, or `None` if not yet materialized.

Under the hood, `get_blob_fields()` inspects `model_fields` metadata for `Blob` instances.

### LLMValidated

`LLMValidated` validates natural-language rules against model data using an LLM call. Add it to your model alongside `Validation` annotations.

```python
from typing import Annotated, ClassVar
from pydantic import BaseModel
from chat_workflow.annotations import Validation
from chat_workflow.mixins import LLMValidated

class ProjectPlan(LLMValidated):
    _validation_rules: ClassVar[list[str]] = [
        "The overall plan must be achievable within the stated timeline",
    ]

    goals: Annotated[str, Validation("Must have at least 2 specific goals")]
    timeline: Annotated[str, Validation("Timeline must be realistic and include milestones")]

plan = ProjectPlan(
    goals="Launch MVP, Acquire 100 users",
    timeline="Q1 2026: MVP launch, Q2 2026: 100 users",
)
# validate_llm_rules runs automatically via @model_validator(mode="after")
```

How it works:

1. **Per-field rules**: `Validation("...")` annotations on individual fields are collected by `collect_per_field_rules()`.
2. **Model-level rules**: Add model-wide rules via the `_validation_rules: ClassVar[list[str]]` class variable.
3. **Schema injection**: `__pydantic_init_subclass__` automatically appends each field's `Validation` rules as bullet points in the field's `description`, making them visible in JSON schemas and LLM prompts.
4. **LLM verification**: `validate_llm_rules` (a `@model_validator(mode="after")`) collects all rules, builds a prompt with the current instance data, and calls the LLM. If any rule is violated, a `ValidationError` is raised.
5. **Graceful failure**: If no API key is configured, validation is silently skipped. This allows model construction in tests and development without an API connection.

### Using Both Mixins Together

The mixins work independently and can be combined in the same model:

```python
from pathlib import Path
from typing import Annotated, ClassVar
from pydantic import BaseModel
from chat_workflow.annotations import Blob, Validation
from chat_workflow.mixins import BlobSyncMixin, LLMValidated

class Specification(BlobSyncMixin, LLMValidated):
    _validation_rules: ClassVar[list[str]] = [
        "The specification must be internally consistent",
    ]

    title: str
    content: Annotated[str, Blob(".md"), Validation("Must be at least 100 words")]

spec = Specification(title="API Design", content="...")
spec.materialize_blobs(Path("./docs"))
```

`BlobSyncMixin` handles file materialization; `LLMValidated` handles rule enforcement. Each operates on its own annotation type without interference.

## Workflow Patterns

TODO: Content to be added in Wave 2

## Code Generation

TODO: Content to be added in Wave 2
