"""Annotation classes for chat-workflow fields.

Provides metadata annotations that can be applied to Pydantic model fields
using ``typing.Annotated`` to control behavior like file materialization
(``Blob``) and natural-language validation rules (``Validation``).
"""

from __future__ import annotations


class Blob:
    """Annotation: field content should be materialized to a file on disk.

    Usage::

        class MyModel(BaseModel):
            diagram: Annotated[str, Blob(".mmd")]

    The ``extension`` controls the file suffix used when :class:`BlobSyncMixin`
    writes the field value to disk.
    """

    def __init__(self, extension: str = ".txt"):
        self.extension = extension


class Validation:
    """Annotation: a natural-language validation rule for this field.

    Usage::

        class MyModel(BaseModel):
            participants: Annotated[str, Validation("Must have at least 3")]

    Rules are collected by :class:`LLMValidated` and verified via LLM call
    during model validation.
    """

    def __init__(self, rule: str):
        self.rule = rule
