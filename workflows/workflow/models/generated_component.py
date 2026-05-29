from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedComponent(BaseModel):
    """Wrapper model for LLM-generated Python code.

    Required because @atomic_workflow demands a Pydantic return type.
    """

    code: str = Field(..., description="Generated Python code for the component")