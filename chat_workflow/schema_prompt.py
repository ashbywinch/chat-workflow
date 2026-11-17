"""A model's output schema as the prompt section that defines it.

The framework already sends a step's result schema to the model: the decorators
put ``model_json_schema()`` in the system prompt and describe it as the
contract for the answer. This module exposes that one piece so a caller that
builds its own prompt — a single annotated call rather than a workflow step —
uses the same rendering instead of copying it.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

INDENT = 2


def schema_section(model: type[BaseModel]) -> str:
    """The model's JSON schema, as the prompt text that defines the answer.

    Members keep their descriptions and enum values, so a value the model may
    return is described beside the member that takes it.
    """
    return json.dumps(model.model_json_schema(), indent=INDENT)
