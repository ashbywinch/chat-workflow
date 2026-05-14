# InteractiveEntity Guide: Creating Workflows Through Conversation

This guide explains the `InteractiveEntity` system — a framework for creating new workflows interactively through conversation, without writing code by hand. It covers the `create-workflow` CLI command, how generated workflows work, and the validation rule system.

**Audience:** Workflow authors and users who want to create new workflows dynamically.

## Overview of the InteractiveEntity System

The `InteractiveEntity` system lets you create new chat-workflow workflows through a guided conversation. Instead of manually writing Pydantic models and workflow functions, you describe what you want and the system generates the code for you.

### How It Works

1. You run the `create-workflow` CLI command
2. A guided wizard asks you for:
   - The workflow name
   - Field definitions (name, type, description)
   - Validation rules
3. The system generates a complete `InteractiveEntity` subclass with:
   - Pydantic fields with `Field(...)` descriptors
   - A `@model_validator` for business rules
   - An `@atomic_workflow` / `@classmethod` `generate_from_chat()` method
4. The generated code is verified with `ruff` (formatting + linting)
5. Files are saved to `workflows/{name}/` and loaded automatically

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `InteractiveEntity` | `chat_workflow/interactive_entity.py` | Base class for generated entities |
| `generate_class()` | `chat_workflow/code_generator.py` | Generates complete Python source for a subclass |
| `verify_code()` | `chat_workflow/code_generator.py` | Runs `ruff` to ensure code quality |
| `create_workflow` | `workflows/create_workflow/flows.py` | Composite workflow that orchestrates the wizard |

### Comparison with Hand-Written Workflows

The [evaluation_criteria workflow](example-evaluation-criteria.md) is a hand-written workflow where the author manually defines models, prompts, and orchestration. The `InteractiveEntity` system automates this process:

| Aspect | Hand-Written (e.g., evaluation_criteria) | Generated (InteractiveEntity) |
|--------|-------------------------------------------|-------------------------------|
| Model definition | Manual Pydantic `BaseModel` subclass | Auto-generated `InteractiveEntity` subclass |
| Validation | Manual `@model_validator` | Auto-generated from validation rules |
| Workflow method | Manual `@atomic_workflow` function | Auto-generated `generate_from_chat()` |
| Prompt | Custom docstring | Auto-generated from class name |
| File structure | Manual creation | Auto-generated files |

## Using the `create-workflow` CLI Command

### Prerequisites

- Chat Workflow installed and configured (see [User Guide](user-guide.md))
- API key set up for your LLM provider

### Running the Wizard

```bash
# Activate the virtual environment
source .venv/bin/activate

# Start the create-workflow wizard
chat-workflow create-workflow create-workflow
```

The wizard guides you through each step with prompts:

```
=== Create Workflow Wizard ===
I'll help you create a new workflow step by step.

What would you like to name your workflow?
> task tracker
Creating workflow 'TaskTracker' in workflows/task_tracker/

Now let's define the fields for your workflow.
Enter one field at a time. Leave the name empty to finish.

  Field name (or empty to finish)
> title
  Type for 'title' (str, int, float, list, etc.)
> str
  Description for 'title'
> The task title
  Added: title: str

  Field name (or empty to finish)
> priority
  Type for 'priority' (str, int, float, list, etc.)
> int
  Description for 'priority'
> Priority level (1-5)
  Added: priority: int

  Field name (or empty to finish)
> (empty)

Validation rules help ensure data quality.
Example: 'name must be non-empty' or 'count must be between 1 and 100'
  Validation rules (empty for none)
> title must be non-empty
> priority must be between 1 and 5

Generating code...
Verifying code quality...
Saving workflow files...
Loading new workflow...

Workflow 'TaskTracker' created successfully!
Location: workflows/task_tracker/
To run: chat-workflow task-tracker --context 'your context'
```

### Command Options

The `create-workflow` command takes no additional CLI options — all interaction happens through the conversation wizard.

## How Generated Workflows Work

### Generated File Structure

When you create a workflow named `task_tracker`, the system generates:

```
workflows/task_tracker/
├── __init__.py     # Exports the model class
└── models.py       # The InteractiveEntity subclass
```

### Generated Code

For the `TaskTracker` example above, `models.py` contains:

```python
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from chat_workflow import InteractiveEntity, ValidationError, atomic_workflow

class TaskTracker(InteractiveEntity):
    _validation_rules: str = "title must be non-empty\npriority must be between 1 and 5"

    title: str = Field(..., description="The task title")
    priority: int = Field(..., description="Priority level (1-5)")

    @model_validator(mode="after")
    def validate_business_rules(self):
        if not self.title:
            raise ValidationError("title must be non-empty")
        return self

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        context: str,
        max_turns: int = 10,
        session = None,
    ):
        """TaskTracker workflow."""
        ...
```

### How `generate_from_chat()` Works

The `generate_from_chat()` classmethod is decorated with `@atomic_workflow`, which means:

1. The LLM receives the class's JSON schema (fields, types, descriptions, validation rules)
2. The LLM conducts a multi-turn conversation with the user to populate the fields
3. The `@model_validator` enforces business rules on the LLM's output
4. The `InteractiveEntity._enforce_validation_rules()` method enforces the `_validation_rules` string at runtime
5. Returns a validated instance of the class

### CLI Auto-Discovery

Generated workflows are automatically discovered by the CLI. The `create-workflow` function is decorated with `@composite_workflow`, which sets `_is_workflow = True` on the function. The CLI discovers all workflow modules in the `workflows/` directory and registers their `@composite_workflow` functions as subcommands.

The function name `create_workflow` becomes the CLI command `create-workflow` (kebab-case conversion).

## Validation Rules Explained

The `InteractiveEntity` base class supports natural-language validation rules through its `_validation_rules` class attribute. These rules are parsed and enforced at runtime.

### Supported Rule Patterns

| Pattern | Example | What It Does |
|---------|---------|--------------|
| `{field} must be non-empty` | `title must be non-empty` | Ensures the field is truthy (non-empty string, non-zero number, etc.) |
| `{field} is required` | `email is required` | Same as non-empty — field must be truthy |
| `{field} must be between {min} and {max}` | `priority must be between 1 and 5` | Ensures numeric field is within inclusive range |
| `{field} must be after {other_field}` | `end_date must be after start_date` | Ensures one numeric field is greater than another |
| `if {field} > {n}, {field} is required` | `if priority > 3, description is required` | Conditional requirement — one field requires another when above a threshold |

### How Rules Are Enforced

Validation happens in two layers:

1. **`@model_validator`** — Generated in the class code. Checks the first field from the validation rules and raises `ValidationError` if violated. This runs during Pydantic model initialization.

2. **`InteractiveEntity._enforce_validation_rules()`** — Called in `model_post_init()`. Parses the `_validation_rules` string line by line and enforces each rule using regex-based pattern matching. This catches all rules, not just the first one.

### Writing Good Validation Rules

- Use field names that match your model fields exactly
- One rule per line
- Rules are case-sensitive for field name matching (but the system does fuzzy matching as a fallback)
- Keep rules simple — complex conditional logic is better handled in a custom `@model_validator`

## Example: Creating a Simple Workflow

Let's walk through creating a "Project Idea" workflow that captures project ideas with validation.

### Step 1: Run the Wizard

```bash
chat-workflow create-workflow create-workflow
```

### Step 2: Answer the Prompts

```
What would you like to name your workflow?
> project idea

Field name (or empty to finish)
> name
  Type for 'name' (str, int, float, list, etc.)
> str
  Description for 'name'
> The project name

Field name (or empty to finish)
> description
  Type for 'description' (str, int, float, list, etc.)
> str
  Description for 'description'
> A brief description of the project

Field name (or empty to finish)
> priority
  Type for 'priority' (str, int, float, list, etc.)
> int
  Description for 'priority'
> Priority from 1 (low) to 5 (high)

Field name (or empty to finish)
> (empty)

Validation rules (empty for none)
> name must be non-empty
> description must be non-empty
> priority must be between 1 and 5
```

### Step 3: Verify the Output

The wizard confirms success:

```
Workflow 'ProjectIdea' created successfully!
Location: workflows/project_idea/
To run: chat-workflow project-idea --context 'your context'
```

### Step 4: Inspect the Generated Files

```bash
cat workflows/project_idea/models.py
```

```python
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from chat_workflow import InteractiveEntity, ValidationError, atomic_workflow

class ProjectIdea(InteractiveEntity):
    _validation_rules: str = "name must be non-empty\ndescription must be non-empty\npriority must be between 1 and 5"

    name: str = Field(..., description="The project name")
    description: str = Field(..., description="A brief description of the project")
    priority: int = Field(..., description="Priority from 1 (low) to 5 (high)")

    @model_validator(mode="after")
    def validate_business_rules(self):
        if not self.name:
            raise ValidationError("name must be non-empty")
        return self

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        context: str,
        max_turns: int = 10,
        session = None,
    ):
        """ProjectIdea workflow."""
        ...
```

## Example: Running a Generated Workflow

### Via CLI

```bash
# Run the project idea workflow
chat-workflow project-idea --context "brainstorming weekend projects"

# Customize max turns
chat-workflow project-idea --context "startup ideas" --max-turns 15
```

The LLM will guide you through a conversation to fill in the `name`, `description`, and `priority` fields, then return a validated `ProjectIdea` object.

### Via Python API

```python
from workflows.project_idea.models import ProjectIdea
from chat_workflow import Session, SessionLog, UserIO

class MyIO(UserIO):
    def echo(self, message: str) -> None:
        print(message)

    def prompt(self, label: str) -> str:
        return input(label + ": ")

idea = ProjectIdea.generate_from_chat(
    context="weekend coding projects",
    max_turns=10,
    session=Session(io=MyIO(), state=SessionLog()),
)

print(f"Project: {idea.name}")
print(f"Priority: {idea.priority}")
```

### Expected Output Format

When the workflow completes successfully, the generated object is returned. The exact conversation flow depends on the LLM, but the result is always a validated `ProjectIdea` instance with all fields populated according to the validation rules.

## Error Handling

Generated workflows handle these error cases:

- **Turn limit exceeded**: Conversation reaches `max_turns` without success
- **Validation failure**: The LLM returns data that violates validation rules (the system retries automatically)
- **Code generation failure**: If `verify_code()` detects unfixable issues, the wizard offers to restart
- **Import failure**: If the generated module can't be loaded, the wizard reports the error

## Extending Generated Workflows

After creating a workflow with the wizard, you can customize the generated code:

1. **Edit the prompt**: Replace the auto-generated docstring in `generate_from_chat()` with a more detailed behavioral prompt
2. **Add custom validators**: Add additional `@model_validator` methods for complex business rules
3. **Add convenience methods**: Add methods to the model class (see [Workflow Author Guide](workflow-author-guide.md#solid-dry-principles-for-workflow-authors))
4. **Compose with other workflows**: Create a `@composite_workflow` function that uses the generated workflow as a step

For more on workflow patterns, see the [Workflow Author Guide](workflow-author-guide.md) and the [Evaluation Criteria example](example-evaluation-criteria.md).