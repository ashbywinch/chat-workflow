# Couch2food Migration Guide

## TL;DR

This document explains how patterns from the couch2food repository map to the chat-workflow-prototype library. If you worked with couch2food's playbook-based architecture, this guide shows you the equivalent concepts in our Python-based system and gives you step-by-step migration patterns. The core idea is the same (LLM-driven conversation to produce structured artifacts), but the implementation is fundamentally different: deterministic Pydantic validation replaces document-based validation, Python workflow functions replace .md playbooks, and the LLM orchestrates the conversation instead of the code driving it.

## Concept Mapping

| Couch2food | chat-workflow-prototype |
|---|---|
| DQS Document (Document Quality Standard) | Pydantic `BaseModel` (e.g., `EvaluationCriteria` in `workflows/evaluation_criteria/evaluation_criteria.py`) |
| Playbook (.md file) | Python workflow function with `@atomic_workflow` or `@composite_workflow` decorator (e.g., `generate_reviewed_criteria` in `workflows/evaluation_criteria/generate_reviewed_criteria.py`) |
| Component Folder (DQS + Playbooks + Index) | Python Package (directory with `__init__.py` exporting workflow functions and models) |
| Component Index File | `__init__.py` that exports workflow methods for CLI discovery |
| Regex-based validation (`_enforce_single_rule`) | `Validation` annotation + `LLMValidated` mixin (`chat_workflow/annotations.py`, `chat_workflow/mixins.py`) |
| Manual file handling | `Blob` annotation + `BlobSyncMixin` for field-to-file materialization (`chat_workflow/annotations.py`, `chat_workflow/mixins.py`) |
| Document-based conversation flow | `@atomic_workflow` decorator driving LLM conversation (`chat_workflow/decorators.py`) |
| CLI discovery by convention | Automatic workflow discovery via `chat_workflow_cli/cli.py` scanning `workflows/` packages |
| `InteractiveEntity` base class | Pydantic `BaseModel` (for new components; `InteractiveEntity` kept for backward compatibility) |
| Python-driven wizard (asking questions field-by-field) | LLM-orchestrated conversation (LLM drives, code handles orchestration/validation/I/O) |
| Code generation via templates | LLM generates raw Python code, verified by `verify_code()` (`chat_workflow/code_generator.py`) |

## Similarities

### Both Use Conversation-Driven LLM Workflow Patterns

Couch2food and chat-workflow-prototype share a fundamental approach: the LLM conducts a multi-turn conversation with the user to gather requirements, ask clarifying questions, and produce a complete artifact. In both systems, the conversation follows a natural back-and-forth pattern where the LLM acts as a coach or facilitator, not just a form-filler.

### Both Produce Structured Artifacts

Couch2food produces DQS documents (structured markdown). chat-workflow-prototype produces Pydantic objects (typed Python data structures). In both cases, the output is validated against rules defined by the workflow author. The difference is in the validation mechanism (regex vs. Python type system), not the overall intent.

### Both Have a Two-Step Process

Both systems follow the same high-level two-step pattern:

1. **Workflow creation** -- Analyze a business process, produce a workflow artifact (Mermaid diagram + analysis), decompose it into component requirements.
2. **Component creation** -- Take a single component requirement, produce the component's model and methods.

This mirrors couch2food's Create Workflow Playbook -> Create Business Component Playbook flow. The decomposition principle (workflow creates, component creates one) follows SOLID/DRY in both systems.

### SOLID/DRY Principles Apply in Both

Both systems enforce Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion principles. In couch2food these are documented in playbook guidance. In chat-workflow-prototype they are encoded in Pydantic validation rules, `Validation` annotations, and workflow function decomposition.

## Key Differences

### Python Deterministic Validation vs. Document-Based Validation

Couch2food uses DQS documents to define what "good" looks like. The LLM reads the DQS and tries to comply, but there is no deterministic enforcement. chat-workflow-prototype uses Pydantic's type system (`ge`, `le`, `min_length`, `@field_validator`, `@model_validator`) for deterministic checks and `Validation` annotations for LLM-backed natural-language rules. This means some rules are enforced at the Python level before any LLM call.

### Pydantic BaseModel vs. DQS Documents

Couch2food DQS documents are markdown files with sections, checklists, and examples. chat-workflow-prototype replaces these with Pydantic `BaseModel` classes that define structure, types, validation, and documentation in a single Python source file. The Pydantic model is the single source of truth.

```python
# Couch2food DQS (conceptual markdown sections):
# - Required fields: name, description, weight
# - Validation: weight must be 0.0 to 10.0
# - Business rules: must include 'budget' criterion

# chat-workflow-prototype equivalent:
class Criterion(BaseModel):
    name: str = Field(..., description="Name of the criterion")
    description: str = Field(..., description="What this criterion measures")
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
```

### Python Workflow Functions vs. .md Playbooks

Couch2foot playbooks are markdown documents with procedural steps written in natural language. chat-workflow-prototype replaces these with Python functions decorated with `@atomic_workflow` or `@composite_workflow`. The decorator turns the function's docstring into a system prompt and handles the LLM conversation loop automatically.

```python
# couch2food: playbook.md with numbered steps
# chat-workflow-prototype:
@atomic_workflow
@classmethod
def generate_from_chat(cls, context: str, ...):
    """You are a helpful assistant. Ask one question at a time.
    Start broad, then ask specific follow-ups."""
    ...
```

### Atomic/Composite Decorators vs. Playbook Structure

Couch2foot playbooks can call other playbooks (making them composite). chat-workflow-prototype makes this explicit with two decorator types:

- `@atomic_workflow` -- A single LLM conversation that produces a typed Pydantic object. The docstring is the system prompt.
- `@composite_workflow` -- A Python function that orchestrates atomic workflows and other logic. No LLM conversation; it coordinates.

This mirrors couch2food's distinction between simple playbooks and playbooks that call sub-playbooks, but makes it explicit at the code level.

### Blob System vs. Manual File Handling

Couch2food handles file I/O manually through playbook steps. chat-workflow-prototype provides `Blob` annotations and `BlobSyncMixin` to materialize model fields to files automatically.

```python
class Workflow(BlobSyncMixin):
    diagram: Annotated[str, Blob(".mmd")] = Field(...)

# One call writes all blob fields to disk
workflow.materialize_blobs(tmp_dir)
path = workflow.get_blob_path("diagram")
```

### Validation Annotations vs. Regex-Based Validation

Couch2food uses regex patterns in `_enforce_single_rule` methods for validation. chat-workflow-prototype uses `Validation` annotations for natural-language rules, enforced by the LLM itself through the `LLMValidated` mixin.

```python
class Workflow(LLMValidated):
    diagram: Annotated[
        str,
        Validation("Must have at least 3 participants"),
        Validation("Message flows must follow a logical sequence"),
    ] = Field(...)

    _validation_rules: ClassVar[list[str]] = [
        "All component names must be descriptive of their function",
    ]
```

Rules are injected into the field's JSON schema description (so the LLM sees them during generation) AND enforced post-generation by an LLM-backed validator. Deterministic rules (`ge`, `le`, `@model_validator`) coexist alongside these natural-language rules.

## Migration Patterns

### Convert DQS Documents to Pydantic Models

1. Identify each required section in the DQS as a field.
2. Choose the appropriate Python type (`str`, `int`, `float`, `list[str]`, nested `BaseModel`).
3. Add `Field(..., description=...)` to communicate "what good looks like" to the LLM.
4. Add deterministic validation (`ge`, `le`, `min_length`, `@field_validator`).
5. Add `Validation` annotations for natural-language rules that need LLM judgment.
6. Add `@model_validator` for cross-field business rules.

```
DQS: 7 Required Sections
  -> name: str, diagram: str, inputs: list[Input], outputs: list[Output], etc.

DQS: Chart Format rules (sequenceDiagram, participant format)
  -> Validation("Must use sequenceDiagram format")
  -> Validation("Participants must use 'Path: Playbook' format")
```

### Convert Playbooks to Workflow Functions

1. Identify whether the playbook is atomic (one conversation) or composite (orchestrates sub-playbooks).
2. For atomic playbooks: create a function decorated with `@atomic_workflow`. The docstring becomes the system prompt. Return type must be a Pydantic model.
3. For composite playbooks: create a function decorated with `@composite_workflow`. Call atomic workflows and handle orchestration logic in Python.
4. Workflow methods follow a naming convention: convention-based constructor methods are called `generate_from_chat`.

```python
# Atomic workflow (one conversation, returns typed object)
@atomic_workflow
@classmethod
def generate_from_chat(cls, context: str, ...) -> Self:
    """System prompt goes here. Tell the LLM how to behave."""
    ...

# Composite workflow (orchestrates sub-workflows)
@composite_workflow
def generate_reviewed_criteria(context: str, ..., session: Session) -> EvaluationCriteria:
    criteria = EvaluationCriteria.generate_from_chat(...)
    for _ in range(max_refinements):
        refined = refine(initial_object=criteria, ...)
        if refined.model_dump() == criteria.model_dump():
            return refined
        criteria = refined
    return criteria
```

### Convert Component Folders to Python Packages

1. Create a directory under `workflows/` with an `__init__.py`.
2. Place each model in its own file (e.g., `Workflow.py`, `Component.py`).
3. Place workflow methods on the relevant model class.
4. Export workflow functions in `__init__.py` for CLI discovery.

```
couch2food:                    chat-workflow-prototype:
components/                    workflows/
  workflow/                      workflow/
    index.md                       __init__.py
    dqs.md                         Workflow.py  (model + methods)
    create-playbook.md             Component.py  (model + methods)
```

The CLI automatically discovers packages under `workflows/` and registers their workflow functions as CLI commands. Each workflow function decorated with `@atomic_workflow` or `@composite_workflow` gets a `_is_workflow` attribute that the CLI detects.

### Handle File I/O with BlobSyncMixin

Fields that need to exist both as model content (for LLM read/write) and as files on disk (for user review) should use the `Blob` annotation.

```python
from typing import Annotated
from pathlib import Path
from chat_workflow.annotations import Blob
from chat_workflow.mixins import BlobSyncMixin

class Workflow(BlobSyncMixin):
    diagram: Annotated[str, Blob(".mmd")] = Field(
        ...,
        description="Mermaid sequenceDiagram",
    )

# In your composite workflow:
workflow = generate_diagram(...)
workflow.materialize_blobs(tmp_dir)
session.io.echo(f"Diagram at {workflow.get_blob_path('diagram')}")
```

Call `materialize_blobs()` after every LLM output and after every refine cycle to keep files in sync.

### Use Validation Annotations Instead of Regex

Replace `_enforce_single_rule()` calls and regex-based validation with `Validation` annotations and the `LLMValidated` mixin. Put deterministic rules (`min_length`, `ge`, `le`, `@model_validator`) alongside natural-language rules.

```python
from typing import Annotated, ClassVar
from chat_workflow.annotations import Validation
from chat_workflow.mixins import LLMValidated

class Workflow(LLMValidated):
    name: str = Field(..., min_length=1, description="Workflow name")

    diagram: Annotated[
        str,
        Validation("Must use sequenceDiagram format"),
        Validation("Must have at least 3 participants"),
    ] = Field(..., description="Mermaid diagram")

    _validation_rules: ClassVar[list[str]] = [
        "All component names must be descriptive",
        "No redundant or overlapping components",
    ]
```

The `LLMValidated` mixin automatically injects `Validation` rules into each field's JSON schema description (so the generation LLM sees them as constraints) and verifies all rules via an LLM-backed validator after construction.

## What to Leave Behind

### InteractiveEntity Subclasses (for New Components)

The old code generator (`chat_workflow/code_generator.py`) emits `InteractiveEntity` subclasses. This class is kept in the codebase for backward compatibility with previously generated workflows, but new components should inherit from Pydantic `BaseModel` directly, optionally with `BlobSyncMixin` and `LLMValidated` mixins.

### Regex-Based Validation (_enforce_single_rule)

The old pattern used regex rules in `_enforce_single_rule()` methods. This is deprecated for new components. Use `Validation` annotations (LLM-backed) and `@model_validator` with `raise ValidationError(...)` (deterministic) instead.

### Python-Driven Conversation Flow

The old `flows.py` pattern had Python code asking the user specific questions: "What would you like to name your workflow?", "Enter a field name:", etc. This is an anti-pattern. The LLM should drive the conversation. Python code handles orchestration, validation, and file I/O, but the LLM decides what questions to ask and in what order.

The difference is subtle but critical:

- **Old (code-driven)**: Python collects name, then fields, then rules separately. The LLM fills templates.
- **New (LLM-orchestrated)**: The LLM conducts a conversation. It figures out what information it needs, asks the user, and produces a complete Pydantic object. Python validates the result.

### Old Repo Name "couch2food" Conventions

The repository was renamed from couch2food to chat-workflow-prototype. Do not use "couch2food" naming conventions for new code. Key changes:

- Package names use chat-workflow namespacing (`chat_workflow/`, `chat_workflow_cli/`).
- Workflow directories live under `workflows/`, not `components/`.
- Workflow functions use `@atomic_workflow` / `@composite_workflow` decorators, not playbook file conventions.

## LLM as Orchestrator

### Why LLM-Driven Conversation Is Superior to Code-Driven

The most important architectural insight from building this system is this: **the LLM should drive the conversation, and the code should handle everything else.**

In the old (code-driven) approach, the Python code controlled the conversation flow:

1. Code asks: "What would you like to name your workflow?"
2. User answers.
3. Code asks: "What fields do you want?"
4. User answers.
5. Code asks: "What validation rules?"
6. User answers.
7. Code generates code from these answers.

This feels logical but it fails because the code has no understanding of the domain. It cannot ask intelligent follow-up questions, detect contradictions, or suggest improvements. The LLM is reduced to a dictation machine, filling in preset templates.

In the LLM-orchestrated approach:

1. The LLM receives a system prompt (the workflow function's docstring) describing its role and constraints.
2. The LLM conducts a natural conversation with the user, asking questions it determines are relevant.
3. The LLM produces a complete Pydantic object when it has enough information.
4. Python validates the object deterministically (types, ranges, cross-field rules).
5. The `LLMValidated` mixin checks natural-language rules via a secondary LLM call.
6. The composite workflow handles looping, file I/O, and orchestration.

### The Distinction

| Aspect | Code-Driven (Anti-Pattern) | LLM-Orchestrated (Correct) |
|---|---|---|
| Who controls the conversation? | Python code asks predetermined questions | LLM decides what to ask and when |
| LLM's role | Form-filler, template-filler | Analyst, coach, facilitator |
| Code's role | Conversation controller | Orchestrator, validator, I/O handler |
| Validation | Post-hoc, code writes rules | Pydantic enforces types, LLM validates rules |
| Error recovery | Code handles specific known errors | LLM can recover conversationally |
| User experience | Robotic, field-by-field interrogration | Natural dialogue with a knowledgeable assistant |

### How @atomic_workflow Enables This

The `@atomic_workflow` decorator (in `chat_workflow/decorators.py`) creates an `AtomicWorkflow` instance (in `chat_workflow/atomic_workflow.py`) that manages the LLM conversation loop. The decorator:

1. Reads the function's docstring as the system prompt (with `{param}` interpolation).
2. Resolves the return type (must be a Pydantic model).
3. Configures the LLM provider, model, max turns, and error handlers.
4. Runs the conversation loop, passing each user input through `process_turn()`.
5. The LLM returns an `AgentResponse` with an `intent` (continue / success / failure).
6. On success, the result is validated through Pydantic (types, ranges, `@model_validator`, `LLMValidated`).

The code handles orchestration (turns, retries, timeouts, errors). The LLM handles conversation (questions, analysis, artifact generation). They each do what they are good at.

### Anti-Pattern: Code Asking Specific Questions

The canonical anti-pattern is the old `flows.py` approach where Python code asked the user for each field individually. This is wrong because:

- The code has no semantic understanding. It cannot detect that a proposed field name conflicts with an existing model.
- The LLM cannot provide value. It is reduced to reformatting user input.
- The conversation is rigid. If the user wants to provide information in a different order, the code cannot adapt.
- The LLM cannot suggest improvements. It cannot say: "I notice you mentioned X, which suggests you might also need a field for Y."

The canonical example of the correct pattern is `generate_reviewed_criteria` in `workflows/evaluation_criteria/generate_reviewed_criteria.py`. It uses an LLM-driven conversation to generate criteria, then enters a refinement loop where the LLM presents the criteria to the user and asks if they want changes. The equality check (`refined.model_dump() == criteria.model_dump()`) is the termination signal: if the LLM returned the same object unchanged, the user is satisfied. The code handles the loop; the LLM handles the conversation.