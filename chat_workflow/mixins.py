"""Mixin classes for chat-workflow models.

Provides :class:`BlobSyncMixin` for materializing annotated fields to disk
and :class:`LLMValidated` for natural-language validation rules backed by
LLM calls.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, override

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .annotations import Blob, Validation, user_labels
from .config import Config
from .conversation import Conversation
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


class RulePassed(BaseModel):
    """The judge's verdict that the data satisfies one rule."""

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(..., min_length=1, description="The rule text, copied exactly")
    reasoning: str = Field(
        ...,
        min_length=1,
        description=(
            "Why the evidence supports the verdict. This is the judge's own "
            "argument, kept apart from the quote."
        ),
    )
    verdict: Literal["pass"] = Field(description="The data satisfies the rule")
    evidence: str = Field(
        ...,
        min_length=1,
        description="The exact quote from the data that shows the rule holds",
    )


class RuleFailed(BaseModel):
    """The judge's verdict that the data does not satisfy one rule."""

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(..., min_length=1, description="The rule text, copied exactly")
    reasoning: str = Field(
        ...,
        min_length=1,
        description=(
            "Why the evidence supports the verdict. This is the judge's own "
            "argument, kept apart from the quote."
        ),
    )
    verdict: Literal["fail"] = Field(description="The data does not satisfy the rule")
    evidence: str = Field(
        ...,
        min_length=1,
        description="The exact quote from the data that shows it violates the rule",
    )


type RuleVerdict = Annotated[RulePassed | RuleFailed, Field(discriminator="verdict")]


class RuleVerdicts(BaseModel):
    """The judge's verdicts, one per rule, in the order the rules were given."""

    verdicts: list[RuleVerdict] = Field(
        ..., description="One verdict per rule, in order"
    )


def _is_actionable_failure(verdict: RulePassed | RuleFailed) -> bool:
    """A failure is a typed verdict; nothing is inferred from its prose."""
    return isinstance(verdict, RuleFailed)


class LLMValidated(BaseModel):
    """Mixin that validates natural-language rules via LLM during validation.

    Discovers :class:`Validation` annotations on individual fields and
    ``_validation_rules`` class variable for model-level rules.  All
    rules are verified by an LLM call in ``@model_validator(mode="after")``.
    """

    _validation_rules: ClassVar[list[str]] = []
    _validation_rules_heading: ClassVar[str] = "Object validation rules"

    # True → validate_llm_rules returns early (no LLM call).
    # Set to ``False`` on a model class to exercise real LLM validation.
    _skip_llm_validation: ClassVar[bool] = False

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

    @override
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Put the user-facing phrase next to the field in the JSON schema.

        ``UserLabel`` is the one place a field's phrase is declared. The
        schema is where the model sees the field while it writes the result,
        so the phrase becomes the schema title; a key and its phrase then sit
        together at the point of generation instead of in a separate list the
        model can drift away from. ``Validation`` rules are appended to the
        description for the same reason.
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, label in user_labels(cls).items():
            cls.model_fields[name].title = label.phrase
        per_field = cls.collect_per_field_rules()
        for name, field_rules in per_field.items():
            field = cls.model_fields[name]
            existing = field.description or ""
            bullets = "\n".join(f"- {r}" for r in field_rules)
            if existing:
                field.description = f"{existing}\n{bullets}"
            else:
                field.description = bullets
        # Pydantic builds its core schema before this hook runs. Without a
        # rebuild, model_json_schema() keeps the pre-mutation title and
        # description and neither reaches the prompt.
        cls.model_rebuild(force=True)

    def validate_deterministic_rules(self) -> None:
        """Run structural checks that do not need an LLM.

        Subclasses override this to validate facts a machine can decide:
        duplicates, required constraints, import resolution. It runs before
        the LLM judge so a plainly invalid object never costs a judge call.
        Raise :class:`ValidationError` on violation.
        """

    @model_validator(mode="after")
    def _reject_placeholders(self) -> Self:
        """A field cannot be filled with a placeholder.

        A model that cannot supply a value sometimes writes "...", "TBD", or
        "N/A". That passes the field's type and length checks, then reaches
        the user as an empty answer. Reject it here so the model retries with
        a real value, or leaves an optional field empty.
        """
        placeholders = {
            "...",
            "…",
            "tbd",
            "tba",
            "n/a",
            "na",
            "???",
            "none provided",
            "not provided",
        }
        for name in type(self).model_fields:
            value = getattr(self, name, None)
            if isinstance(value, str) and value.strip().casefold() in placeholders:
                raise ValueError(
                    f"The field '{name}' contains the placeholder {value!r}. "
                    "Supply the real value; if the field is optional and the "
                    "value is unknown, leave it empty."
                )
        return self

    @model_validator(mode="after")
    def validate_llm_rules(self) -> Self:
        """Call the model to check all rules against the current instance.

        The rule judge is a :class:`~chat_workflow.conversation.Conversation`:
        one model path, one transcript, and the schema carried by the request.
        Raises :class:`ValidationError` (from ``chat_workflow.exceptions``) on
        any violation.

        Skips silently when infrastructure is unavailable (no config file or
        API key), so models can be constructed in test environments.
        """
        # Deterministic first: cheap, precise, and no judge tokens wasted on
        # an object that is structurally invalid.
        self.validate_deterministic_rules()

        rules = self.collect_all_rules()
        if not rules:
            return self

        if self._skip_llm_validation:
            return self

        try:
            config = Config(Path(__file__).parent.parent / "config.json")
        except Exception:
            return self

        prompt = (
            f"Data:\n{self.model_dump_json(indent=2)}\n\n"
            "Rules:\n"
        )
        for i, rule in enumerate(rules, 1):
            prompt += f"{i}. {rule}\n"
        prompt += (
            "\nReturn one verdict per rule, in order. Each verdict has:\n"
            '- "rule": the rule text, copied exactly\n'
            '- "reasoning": why the evidence supports your verdict, in one or '
            "two sentences; decide the verdict from this reasoning\n"
            '- "verdict": "pass" if the data satisfies the rule, "fail" if it '
            "does not\n"
            '- "evidence": the exact quote from the data that supports the '
            "verdict; a fail MUST quote the field or phrase that violates the "
            "rule\n"
            "Reasoning explains the judgment; evidence quotes the data. Do not "
            "put an argument in evidence or a quote in reasoning. Do not "
            "propose a correction: the caller decides what to do with a fail."
        )

        try:
            conversation = Conversation(
                model=config.model,
                provider=config.provider,
                api_base=config.api_base,
                api_key_env=config.api_key_env,
                instructions=(
                    "You are a validation assistant. Evaluate the data against "
                    "each rule."
                ),
                output_type=RuleVerdicts,
                output_retries=config.max_retries,
                transport_retries=config.max_retries,
                timeout_seconds=config.request_timeout_seconds,
                temperature=0.0,
                thinking=True,
                extra_headers=config.extra_headers,
            )
            # This validator runs inside the agent's own event loop, where
            # `Conversation.send` cannot start one. A thread gives the judge
            # its own loop; the validator stays synchronous, as Pydantic
            # requires.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                turn = pool.submit(conversation.send, prompt).result()
            verdicts = turn.output
        except Exception:
            return self

        failures = [
            verdict
            for verdict in verdicts.verdicts
            if _is_actionable_failure(verdict)
        ]
        if failures:
            raise ValidationError(
                "; ".join(
                    f"[{verdict.rule}] {verdict.reasoning} "
                    f"Evidence: {verdict.evidence}"
                    for verdict in failures
                )
            )

        return self
