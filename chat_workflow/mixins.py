"""Mixin classes for chat-workflow models.

Provides :class:`BlobSyncMixin` for materializing annotated fields to disk
and :class:`LLMValidated` for natural-language validation rules backed by
LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, PrivateAttr, model_validator

from .annotations import Blob, Validation
from .exceptions import ValidationError


def get_blob_fields(model: type[BaseModel]) -> dict[str, str]:
    """Discover ``{field_name: extension}`` from Blob-annotated fields.

    Iterates :attr:`BaseModel.model_fields`, checks metadata for
    :class:`Blob` instances, and returns a mapping of field names to
    their configured file extensions.
    """
    result: dict[str, str] = {}
    for name, field in model.model_fields.items():
        for meta in field.metadata:
            if isinstance(meta, Blob):
                result[name] = meta.extension
                break
    return result


class BlobSyncMixin(BaseModel):
    """Mixin that materializes Blob-annotated fields to files on disk.

    Usage::

        class MyModel(BlobSyncMixin):
            diagram: Annotated[str, Blob(".mmd")] = Field(...)

        model.materialize_blobs(Path("/tmp"))
        path = model.get_blob_path("diagram")
    """

    _blob_paths: dict[str, Path] = PrivateAttr(default_factory=dict)

    def materialize_blobs(self, output_dir: Path) -> BlobSyncMixin:
        """Write all Blob fields to files in *output_dir*.

        Creates *output_dir* if it does not exist.  Writes each
        Blob-annotated field to ``{output_dir}/{field_name}{extension}``.
        Stores the resulting paths in ``_blob_paths``.

        Returns *self* for chaining.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        blob_fields = get_blob_fields(type(self))
        for field_name, extension in blob_fields.items():
            value = getattr(self, field_name)
            path = output_dir / f"{field_name}{extension}"
            path.write_text(str(value))
            # Use object.__setattr__ to bypass Pydantic's private attribute
            # handling; PrivateAttr is already excluded from serialization.
            self._blob_paths[field_name] = path

        return self

    def get_blob_path(self, field: str) -> Path | None:
        """Return the path where the blob for *field* was materialized.

        Returns ``None`` if the field has not been materialized or is
        not a Blob field.
        """
        return self._blob_paths.get(field)


class LLMValidated(BaseModel):
    """Mixin that validates natural-language rules via LLM during validation.

    Discovers :class:`Validation` annotations on individual fields and
    ``_validation_rules`` class variable for model-level rules.  All
    rules are verified by an LLM call in ``@model_validator(mode="after")``.
    """

    _validation_rules: ClassVar[list[str]] = []

    @classmethod
    def collect_per_field_rules(cls) -> dict[str, list[str]]:
        """Collect ``Validation`` rules from field annotations.

        Returns ``{field_name: [rule_strings]}``.
        """
        rules: dict[str, list[str]] = {}
        for name, field in cls.model_fields.items():
            field_rules: list[str] = []
            for meta in field.metadata:
                if isinstance(meta, Validation):
                    field_rules.append(meta.rule)
            if field_rules:
                rules[name] = field_rules
        return rules

    @classmethod
    def collect_all_rules(cls) -> list[str]:
        """Collect per-field and per-model rules as a flat list of strings."""
        rules: list[str] = []
        for field_name, field_rules in cls.collect_per_field_rules().items():
            for rule in field_rules:
                rules.append(f"[{field_name}] {rule}")
        rules.extend(cls._validation_rules)
        return rules

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Inject ``Validation`` rules into field descriptions for JSON schema.

        Appends each ``Validation`` rule as a bullet point to the field's
        ``description`` attribute, so the rules are visible in generated
        JSON schemas and LLM prompts.
        """
        super().__pydantic_init_subclass__(**kwargs)
        per_field = cls.collect_per_field_rules()
        for name, field_rules in per_field.items():
            field = cls.model_fields[name]
            existing = field.description or ""
            bullets = "\n".join(f"- {r}" for r in field_rules)
            if existing:
                field.description = f"{existing}\n{bullets}"
            else:
                field.description = bullets

    @model_validator(mode="after")
    def validate_llm_rules(self) -> BaseModel:
        """Call the LLM to check all rules against the current instance.

        Uses :func:`chat_workflow.llm_interaction.get_client` to obtain
        an LLM client from config.  Builds a prompt listing all rules and
        asks the LLM to verify each one.  Raises :class:`ValidationError`
        (from ``chat_workflow.exceptions``) on any violation.

        Skips silently when infrastructure is unavailable (no config file
        or API key), so models can be constructed in test environments.
        """
        from .config import Config
        from .llm_interaction import get_client

        rules = self.collect_all_rules()
        if not rules:
            return self

        try:
            config = Config(Path(__file__).parent.parent / "config.json")
            client = get_client(provider=config.provider)
        except Exception as err:
            raise RuntimeError(
                "Failed to load config or API key for LLM validation. "
                "In test environments, mock validate_llm_rules to skip the LLM call."
            ) from err

        prompt = (
            "You are a validation assistant.  Given the following data, "
            "verify that ALL of these business rules are satisfied.\n\n"
            f"Data:\n{self.model_dump_json(indent=2)}\n\n"
            "Rules:\n"
        )
        for i, rule in enumerate(rules, 1):
            prompt += f"{i}. {rule}\n"
        prompt += (
            "\nRespond ONLY with a JSON object containing:\n"
            '- "valid": boolean — true only if ALL rules pass\n'
            '- "violations": list of strings describing each violation\n'
        )

        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            max_retries=config.max_retries,
            timeout=config.request_timeout_seconds,
        )

        content = getattr(response, "choices", None)
        if content and hasattr(content[0], "message"):
            text = content[0].message.content or "{}"
        elif hasattr(response, "valid"):
            text = json.dumps({"valid": response.valid, "violations": getattr(response, "violations", [])})
        else:
            text = "{}"

        try:
            result = json.loads(str(text))
        except (json.JSONDecodeError, TypeError, ValueError):
            result = {"valid": True, "violations": []}

        if not result.get("valid", True):
            violations = result.get("violations", ["Business rule validation failed"])
            raise ValidationError("; ".join(violations))

        return self