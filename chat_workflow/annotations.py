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


class UserLabel:
    """Annotation: the phrase to use with the user instead of this field's name.

    The result schema is internal vocabulary. This label tells the model what
    to say in ``message`` instead, and which internal forms must not leak.
    The user's own word always wins over the phrase.

    Usage::

        class Resource(BaseModel):
            source: Annotated[
                str,
                UserLabel("where it comes from", "source"),
            ]

    A common English word (``name``, ``format``, ``description``) can set
    ``guard=False``: the prompt still suggests the phrase, but the guard does
    not reject the word when it is ordinary language rather than schema
    leakage.
    """

    def __init__(self, phrase: str, *aliases: str, guard: bool = True):
        self.phrase = phrase
        self.aliases = aliases
        self.guard = guard


def user_labels(model: object) -> dict[str, UserLabel]:
    """Return ``{field_name: UserLabel}`` for a Pydantic model."""
    labels: dict[str, UserLabel] = {}
    for name, field in getattr(model, "model_fields", {}).items():
        for metadata in field.metadata:
            if isinstance(metadata, UserLabel):
                labels[name] = metadata
                break
    return labels
