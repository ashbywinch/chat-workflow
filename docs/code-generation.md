# Code Generation

## TL;DR

The chat-workflow library does not use templates or AST-based code generation for new components. Instead, the LLM designs the component and emits raw Python code as a string. We feed that string through `verify_code()` which runs `ruff format` and `ruff check`, auto-fixing issues, and raises `RuntimeError` if the code can't be cleaned up after 3 attempts. The result is clean, lint-free Python written to disk.

The `chat_workflow/code_generator.py` module provides helpers for this process:

- `verify_code(code)` -- the core pipeline for turning raw LLM output into clean Python
- `generate_code()` -- optional pass-through with libcst normalization
- `generate_field()` -- generates Pydantic field definition strings
- `generate_workflow_method()` -- generates `@atomic_workflow` / `@classmethod` boilerplate

A companion convention ("one class per file") keeps the generated codebase organized: each Pydantic model class lives in its own file named after the class.

---

## LLM-Generated Code

### The Approach

The `Component.create()` flow is the primary consumer of code generation. It works like this:

1. The LLM receives the full context of the component requirements (name, purpose, inputs, outputs, field types, validation rules, method signatures).
2. The LLM designs the component and emits Python source code as a string.
3. We feed that string through `verify_code()` which formats and lints it.
4. Any remaining quality constraints are checked via `Validation` rules on the output.
5. The cleaned code is written to disk at the component's `code_path`.

This approach works because the LLM has all the context it needs for a one-shot correct output. The code does the mechanical cleanup that LLMs are bad at (consistent indentation, import ordering, trailing whitespace). The LLM does the semantic work that code is bad at (designing sensible field types, writing good docstrings, choosing the right validation patterns).

```
LLM designs component         raw Python string
         |
         v
verify_code(code)             runs ruff format --check + ruff check
         |                    auto-fixes, loops up to 3 times
         v
clean, lint-free code         written to disk
```

### Example: LLM Output Before Verification

The LLM might emit something like this:

```python
from __future__ import annotations
from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow


class Criterion(BaseModel):
    name: str = Field(..., description="Name of this criterion")
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    description:str=Field(...,description="What this criterion measures")

    @atomic_workflow
    @classmethod
    def generate_from_chat(cls, context:str, max_turns:int=10, session=None):
        """Generate criterion from chat conversation."""
        ...
```

There are subtle issues: inconsistent spacing around `:` in annotations, missing whitespace in keyword arguments. `verify_code()` catches and fixes these.

### When to Use LLM-Generated Code

This approach is for generating new workflow component classes at runtime. It is not for:

- Writing framework code (that's done by hand, reviewed in PRs).
- Generating configuration files or data (those use Pydantic models directly).
- Producing one-off scripts (use the LLM interactively through `@atomic_workflow`).

---

## Code Verification with verify_code()

`verify_code()` is the gatekeeper between raw LLM output and clean disk files. It lives in `chat_workflow/code_generator.py`.

### How It Works

```python
from chat_workflow.code_generator import verify_code

raw_code = '''
from __future__ import annotations
from pydantic import BaseModel,Field


class Task(BaseModel):
    title:str=Field(...,description="Task title")
    done: bool = Field(default=False, description="Completion status")
'''

cleaned = verify_code(raw_code)
print(cleaned)
# Output: properly formatted, lint-free Python
```

The function does the following:

1. Writes the code to a temporary file.
2. Runs `ruff format --check` on the file. If formatting fails, runs `ruff format` to auto-fix.
3. Runs `ruff check` on the file. If linting fails, runs `ruff check --fix` to auto-fix.
4. If either check fails after auto-fix, it loops back to step 2 for another attempt.
5. After `max_attempts` (default: 3), if the code still does not pass, raises `RuntimeError` with the combined stdout and stderr from the failed checks.
6. On success, reads the cleaned file content and returns it.

### Signature

```python
def verify_code(code: str, max_attempts: int = 3) -> str:
    """Verify generated code passes ruff linting and formatting checks.

    Parameters
    ----------
    code:
        The Python source code to verify.
    max_attempts:
        Maximum number of fix attempts (default: 3).

    Returns
    -------
    str
        The cleaned, lint-free Python source code.

    Raises
    ------
    RuntimeError
        If the code cannot be fixed after max_attempts.
    """
```

### Error Handling

When `verify_code()` raises, the error message includes both the `ruff format --check` output and the `ruff check` output. This tells you exactly what is wrong so you can fix the prompt or the LLM's generation pattern.

```python
try:
    cleaned = verify_code(bad_code)
except RuntimeError as e:
    print(e)
    # "Code failed ruff checks after 3 attempts."
    # "--- format --check ---"
    # "error: Bad indentation at line 12"
    # "--- check ---"
    # "error: F401 `os` imported but unused"
```

### How Ruff Is Located

The function searches for the `ruff` executable in this order:

1. Alongside the current Python interpreter (the venv `bin/` directory).
2. On the system `PATH` via `shutil.which()`.

If ruff is not found, it raises `RuntimeError` with instructions to install it.

---

## One Class Per File Convention

Every Pydantic model class that participates in a workflow lives in its own file. The file is named after the class.

```
workflows/
  evaluation_criteria/
    __init__.py
    flows.py
    EvaluationCriteria.py       # class EvaluationCriteria(...)
    Criterion.py                # class Criterion(...)
```

This convention makes the codebase easy to navigate. If you are looking for the `Criterion` model, you open `Criterion.py`. If you are looking for the `Workflow` model, you open `Workflow.py`.

It also keeps each file small and focused. A model file typically contains:

- The Pydantic model class with fields, validators, and docstrings.
- Workflow classmethods (`generate_from_chat`) that produce instances of that model.
- Any helper functions that are specific to that model.

### Example

```python
# workflows/workflow/Workflow.py
from __future__ import annotations
from typing import Annotated, ClassVar
from pydantic import BaseModel, Field, model_validator
from chat_workflow.annotations import Blob, Validation
from chat_workflow.mixins import BlobSyncMixin, LLMValidated
from chat_workflow import atomic_workflow


class Workflow(BlobSyncMixin, LLMValidated):
    """Complete workflow specification with diagram."""

    name: str = Field(..., description="Workflow name ending in 'Workflow'")

    diagram: Annotated[
        str,
        Blob(".mmd"),
        Validation("Must have at least 3 participants"),
    ] = Field(..., description="Mermaid sequenceDiagram")

    _validation_rules: ClassVar[list[str]] = [
        "All component names must be descriptive of their function",
    ]

    @model_validator(mode="after")
    def validate_diagram_structure(self):
        if self.diagram and "participant" not in self.diagram:
            raise ValueError("Diagram must declare participants")
        return self

    @atomic_workflow
    @classmethod
    def generate_from_chat(cls, context: str, max_turns: int = 10, session=None):
        """Generate a Workflow from a process description."""
        ...
```

### Package Structure With One Class Per File

```
workflows/workflow/
  __init__.py              # exports workflow functions for CLI discovery
  Workflow.py              # class Workflow(...)
  Component.py             # class Component(...)
  ComponentRequirement.py  # class ComponentRequirement(...)
  GapAnalysis.py           # class GapAnalysis(...)
  Input.py                 # class Input(...)
  Output.py                # class Output(...)
  flows.py                 # composite/atomic workflow orchestration functions
```

The `__init__.py` re-exports the workflow functions for CLI discovery:

```python
# workflows/workflow/__init__.py
from workflows.workflow.flows import create_workflow
```

---

## Code Generation Helpers

The `chat_workflow/code_generator.py` module provides several helpers for the generation pipeline. These are convenience functions used by the `Component.create()` flow and are available for any code generation task.

### generate_code()

Two calling conventions:

```python
from chat_workflow.code_generator import generate_code

# Pass through an existing source string (normalizes via libcst)
code = generate_code(source="class Foo:\n    pass\n")

# Generate a model class from field definitions
code = generate_code(
    class_name="Task",
    fields=[
        ("title", "str"),
        ("done", "bool"),
        ("priority", "int"),
    ],
)
```

When called with `source`, the string is parsed by libcst and re-emitted, normalizing whitespace and indentation. When called with `class_name` and `fields`, it builds a class definition with annotated assignments.

Note: the fields-only form creates a bare class with no methods, no imports, and no Pydantic `Field()` calls. It is a minimal starting point. Most real usage goes through the LLM-based approach instead.

### generate_field()

Generates a Pydantic field definition string:

```python
from chat_workflow.code_generator import generate_field

# With description
field_str = generate_field("name", "str", "User display name")
# Returns: 'name: str = Field(..., description="User display name")'

# Without description
field_str = generate_field("count", "int")
# Returns: 'count: int = Field(...)'
```

Useful when building field definitions programmatically before passing them to the LLM or to `generate_code()`.

### generate_workflow_method()

Generates the boilerplate for an `@atomic_workflow` classmethod:

```python
from chat_workflow.code_generator import generate_workflow_method

method = generate_workflow_method("Task")
print(method)
```

Output:

```python
@atomic_workflow
@classmethod
def generate_from_chat(
    cls,
    context: str,
    max_turns: int = 10,
    session = None,
):
    """Workflow method for Task."""
    ...
```

The method body is a placeholder (`...`). The LLM fills in the actual implementation when it generates the component.

### generate_model_validator()

Generates a `@model_validator` method with a simple truthiness check:

```python
from chat_workflow.code_generator import generate_model_validator

validator = generate_model_validator("name must be non-empty")
print(validator)
```

Output:

```python
from chat_workflow import ValidationError

@model_validator(mode="after")
def validate_business_rules(self):
    if not self.name:
        raise ValidationError("name must be non-empty")
    return self
```

If the field name differs from the first word of the rule, pass it explicitly:

```python
validator = generate_model_validator(
    "title must be non-empty",
    field_name="title",
)
```

### Complete Example: LLM + Helpers Pipeline

```python
from pathlib import Path

from chat_workflow.code_generator import verify_code

# Step 1: LLM emits raw Python code (simulated here)
llm_output = '''from __future__ import annotations
from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow


class Task(BaseModel):
    title: str = Field(..., description="Task title")
    done: bool = Field(default=False, description="Completion status")

    @atomic_workflow
    @classmethod
    def generate_from_chat(
        cls,
        context: str,
        max_turns: int = 10,
        session = None,
    ):
        """Generate a Task from chat conversation."""
        ...
'''

# Step 2: Verify and clean
cleaned_code = verify_code(llm_output)

# Step 3: Write to disk
output_path = Path("workflows/tasks/Task.py")
output_path.write_text(cleaned_code)
```

---

## Reference

| Function | File | Purpose |
|----------|------|---------|
| `verify_code(code, max_attempts)` | `chat_workflow/code_generator.py` | Main pipeline: ruff format + lint, auto-fix, raise on failure |
| `generate_code(source=...)` | `chat_workflow/code_generator.py` | Pass-through with libcst normalization |
| `generate_code(class_name=..., fields=...)` | `chat_workflow/code_generator.py` | Generate bare model class from field tuples |
| `generate_field(name, type_, description)` | `chat_workflow/code_generator.py` | Generate Pydantic field definition string |
| `generate_workflow_method(class_name)` | `chat_workflow/code_generator.py` | Generate `@atomic_workflow` / `@classmethod` boilerplate |
| `generate_model_validator(rule, field_name)` | `chat_workflow/code_generator.py` | Generate `@model_validator` with truthiness check |
| `import_module(path)` | `chat_workflow/code_generator.py` | Dynamic module import |
| `reload_module(path)` | `chat_workflow/code_generator.py` | Reload already-imported module |

For the design decisions behind LLM-generated code, see the [Workflow Author Guide](workflow-author-guide.md).