"""Annotation classes for chat-workflow fields.

Provides metadata annotations that can be applied to Pydantic model fields
using ``typing.Annotated`` to control behavior like file materialization
(``Blob``) and natural-language validation rules (``Validation``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

__all__ = ["Blob", "UserLabel", "UserLabels", "Validation", "user_labels"]


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

    def __init__(
        self,
        phrase: str,
        aliases: str | tuple[str, ...] = (),
        *,
        guard: bool = True,
    ):
        self.phrase = phrase
        self.aliases = (aliases,) if isinstance(aliases, str) else tuple(aliases)
        self.guard = guard

    def __get_pydantic_json_schema__(self, schema, handler):
        """Give the field the user's label as its schema title and description.

        The model reads the tool schema, so the title is where a field's name
        would otherwise become the model's word for it. The description says
        that the title is a user facing label, so the model does not have to
        infer it from a separate prompt section.
        """
        json_schema = handler(schema)
        resolved = handler.resolve_ref_schema(json_schema)
        resolved["title"] = self.phrase
        label = f'User facing label: "{self.phrase}".'
        description = resolved.get("description")
        resolved["description"] = (
            f"{label} {description}" if description else label
        )
        return json_schema


class UserLabels(Mapping[str, "UserLabel"]):
    """The user-facing label for each field of a Pydantic model.

    The map is a domain object: callers read it by field name and iterate the
    pairs, and the build rule lives here rather than in every call site.
    """

    def __init__(self, model: object):
        labels: dict[str, UserLabel] = {}
        for name, field in getattr(model, "model_fields", {}).items():
            for metadata in field.metadata:
                if isinstance(metadata, UserLabel):
                    labels[name] = metadata
                    break
        self._labels = labels

    def __getitem__(self, name: str) -> UserLabel:
        return self._labels[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)


def user_labels(model: object) -> UserLabels:
    """Return the label for each field of a Pydantic model."""
    return UserLabels(model)
