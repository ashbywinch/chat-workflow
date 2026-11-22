"""Mixin classes for chat-workflow models.

Provides :class:`BlobSyncMixin` for materializing annotated fields to disk
and :class:`LLMValidated` for natural-language validation rules backed by
LLM calls.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .annotations import Blob, Validation, user_labels
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


class RuleVerdict(BaseModel):
    """One judge verdict for one validation rule."""

    rule: str = Field(..., description="The rule text, copied exactly from the list")
    passed: bool = Field(..., description="Whether the data satisfies this rule")
    explanation: str | None = Field(None, description="Why the rule passed or failed")
    evidence: str | None = Field(
        None,
        description=(
            "Exact quote from the data that supports the verdict. A failed "
            "verdict MUST quote the code line, field, or phrase that violates "
            "the rule; a failed verdict without evidence is discarded."
        ),
    )
    fix: str | None = Field(
        None,
        description=(
            "For a failed verdict, the concrete correction: the exact value or "
            "change to make. Name the field and give the corrected form (for "
            "example: replace name 'Approval' with 'Approval Record'). Empty "
            "for a passing verdict."
        ),
    )


class RuleVerdicts(BaseModel):
    """The judge's verdicts, one per rule, in the order the rules were given."""

    verdicts: list[RuleVerdict] = Field(
        ..., description="One verdict per rule, in order"
    )


class _VerdictLike(Protocol):
    """The two fields the coherence check reads, from either verdict type."""

    explanation: str | None
    fix: str | None


# A judge sometimes narrates its own change of mind after the sentence that
# reads the data ("I need to correct my verdict"). That sentence is about the
# verdict, not the rule, so the conclusion is the last sentence that is not.
_VERDICT_META = (
    "i need to correct",
    "i should correct",
    "correct my verdict",
    "let me revise",
    "revising my verdict",
    "re-evaluat",
    "i was wrong",
)

_PASS_PHRASES = (
    "passes",
    " pass",
    "satisfied",
    "is valid",
    "meets the rule",
    "not violated",
    "does not violate",
    "no violation",
    "not a violation",
    "is unambiguous",
    "no fix needed",
    "is acceptable",
)

# A negated pass ("does not pass", "fails") is not a pass; a negated
# violation ("does not violate") is. Match the negated *pass* forms only,
# so a conclusion of "does not violate the rule" is read as the pass it is.
_NEGATED_PASS_PHRASES = (
    "does not pass",
    "doesn't pass",
    "does not satisfy",
    "doesn't satisfy",
    "is not satisfied",
    "fails",
    "failure",
    "is not valid",
)


def _concludes_a_pass(text: str | None) -> bool:
    """True when the last sentence about the rule concludes that it passes.

    Read the last sentence, so a hedge ("could be considered a failure,
    but...") does not hide the conclusion; skip the sentences that narrate the
    judge's own verdict, which say nothing about the data.
    """
    text = (text or "").strip().lower()
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if part]
    about_the_rule = [
        sentence
        for sentence in sentences
        if not any(marker in sentence for marker in _VERDICT_META)
    ]
    conclusion = (about_the_rule or sentences or [""])[-1]
    return any(phrase in conclusion for phrase in _PASS_PHRASES) and not any(
        phrase in conclusion for phrase in _NEGATED_PASS_PHRASES
    )


def _verdict_claims_the_rule_passes(verdict: _VerdictLike) -> bool:
    """True when a failed verdict's own words say the rule passed.

    A self-contradictory verdict is a malformed judge response, not a
    violation: its conclusion says the data satisfies the rule while the
    ``passed`` flag says it does not.
    """
    if _concludes_a_pass(verdict.explanation):
        return True
    # The judge may write the withdrawn verdict into the fix it was asked for;
    # a fix that concludes the data already passes asks for no change either.
    if _concludes_a_pass(verdict.fix):
        return True
    # A fix that says there is nothing to fix is a pass in disguise, whatever
    # the explanation's last sentence says.
    fix = (verdict.fix or "").strip().lower()
    return fix.startswith(
        ("no fix", "no change", "nothing to fix", "none needed", "not needed", "n/a")
    )


def _is_actionable_failure(verdict: RuleVerdict) -> bool:
    """A failure must be substantive and fixable, not a malformed verdict.

    The judge is told to attach the violating evidence and a concrete fix to
    every failed verdict. A failure without both is discarded, as is one whose
    conclusion says the rule passes.
    """
    return (
        not verdict.passed
        and bool((verdict.evidence or "").strip())
        and bool((verdict.fix or "").strip())
        and not _verdict_claims_the_rule_passes(verdict)
    )


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
    def validate_llm_rules(self) -> BaseModel:
        """Call the model to check all rules against the current instance.

        The rule judge is a :class:`~chat_workflow.conversation.Conversation`:
        one model path, one transcript, and the schema carried by the request.
        Raises :class:`ValidationError` (from ``chat_workflow.exceptions``) on
        any violation.

        Skips silently when infrastructure is unavailable (no config file or
        API key), so models can be constructed in test environments.
        """
        from .config import Config
        from .conversation import Conversation

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
            '- "passed": true only if the data satisfies the rule\n'
            '- "explanation": a brief reason\n'
            '- "evidence": the exact quote from the data that supports the verdict. '
            "A failed verdict MUST quote the code line, field, or phrase that "
            "violates the rule. A failed verdict without evidence and a concrete "
            "fix is discarded.\n"
            '- "fix": for a failed verdict, the concrete correction — name the '
            "field and give the exact corrected value or form (for example: "
            "replace name 'Approval' with 'Approval Record'). Empty on a pass.\n"
            "If your explanation concludes that the data passes or satisfies a "
            "rule, set passed to true. A failed verdict must state what is wrong "
            "and quote the violating text, not the valid value."
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
                extra_headers=config.extra_headers,
            )
            # This validator runs inside the agent's own event loop, where
            # `Conversation.send` cannot start one. A thread gives the judge
            # its own loop; the validator stays synchronous, as Pydantic
            # requires.
            import concurrent.futures

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
                    f"[{verdict.rule}] {verdict.explanation or 'rule failed'}"
                    + (f" Evidence: {verdict.evidence}" if verdict.evidence else "")
                    + (f" Fix: {verdict.fix}" if verdict.fix else "")
                    for verdict in failures
                )
            )

        return self
