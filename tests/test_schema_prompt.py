"""The prompt section a model's schema renders to."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from chat_workflow.schema_prompt import schema_section


class Answer(BaseModel):
    confidence: Literal["yes", "no"] = Field(description="the reviewer's certainty")


def test_the_section_is_the_models_schema_with_its_descriptions() -> None:
    section = schema_section(Answer)
    schema = json.loads(section)
    assert schema["properties"]["confidence"]["enum"] == ["yes", "no"]
    assert schema["properties"]["confidence"]["description"] == "the reviewer's certainty"
    assert section == json.dumps(Answer.model_json_schema(), indent=2)
