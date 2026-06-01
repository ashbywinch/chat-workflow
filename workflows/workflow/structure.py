"""ComponentStructure model — Pydantic-level structural design of a generated component.

ComponentStructure captures the Pydantic model structure that will be generated
for a workflow component. It describes fields, validators, and imports needed
to produce the final Python code — without any LLM or conversation concepts.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow

from .domain_spec import ComponentDomainSpec


class StructField(BaseModel):
    """A single field in the generated Pydantic model.

    Describes the field's name, Python type expression, and any Pydantic
    Field() kwargs that should be applied. The type_expr and field_def_kwargs
    use string values because they will be serialized into generated Python code.
    """

    name: str = Field(
        ...,
        description="Field name (e.g. 'action_owner', 'description')",
        min_length=1,
    )
    type_expr: str = Field(
        ...,
        description=("Python type expression as a string (e.g. 'str', 'list[str]', 'list[ActionItem]')"),
        min_length=1,
    )
    field_def_kwargs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Pydantic Field kwargs as string key-value pairs (e.g. {'description': '...', 'min_length': '1'})"
        ),
    )


class StructValidator(BaseModel):
    """A validation rule for the generated Pydantic model.

    Captures a business rule or quality criterion that should be enforced
    as a @model_validator or @field_validator in the generated code.
    """

    rule: str = Field(
        ...,
        description=("Validation rule text (e.g. 'description must not exceed 3 sentences')"),
        min_length=1,
    )
    domain_origin: str = Field(
        ...,
        description=("The 'what good looks like' statement or business rule this validation came from"),
        min_length=1,
    )


class ComponentStructure(BaseModel):
    """Pydantic-level structural design for a generated component.

    Represents the complete Pydantic model structure that will be code-generated:
    the artifact description, base class, fields, validators, and extra imports.
    This is a pure data model — no LLM functionality.
    """

    description: str = Field(
        ...,
        description="Refined artifact description for the generated model",
        min_length=1,
    )
    base_class: str = Field(
        "BaseModel",
        description=("Base class for the generated model: 'BaseModel' (default) or 'LLMValidated'"),
    )
    fields: list[StructField] = Field(
        default_factory=list,
        description="Fields that make up the generated Pydantic model",
    )
    model_validators: list[StructValidator] = Field(
        default_factory=list,
        description="Validation rules for the generated model",
    )
    extra_imports: list[str] = Field(
        default_factory=list,
        description="Additional Python imports needed by the generated code",
    )

    @atomic_workflow
    @classmethod
    def design(
        cls,
        domain_spec: Annotated[
            ComponentDomainSpec,
            (
                "The domain specification — what the artifact means in the user's"
                " world, its fields, and what makes it excellent"
            ),
        ],
        max_turns: Annotated[int, "Maximum conversation turns"] = 10,
    ) -> ComponentStructure:
        """You are a data design expert helping someone define the structure and rules
        for a business artifact they've already described.

        The user has already told you what their artifact is, what fields it has, and
        what makes it excellent. Your job is to translate that domain understanding
        into concrete structural decisions — what kind of information each field
        holds, what constraints it should follow, and what overall rules the artifact
        must satisfy.

        This is not a technical conversation. The user understands their business but
        not data structures or programming. When you propose a constraint, frame it
        in terms they care about. For example, instead of asking "should I add
        max_length=200?", ask "should the description have a maximum length?" Instead
        of asking "should this be a list type?", ask "could there be more than one of
        these?" Instead of asking "should I add a @model_validator?", ask "should we
        have a rule that every action item must have an owner?"

        Start by proposing what you think the structure should look like based on the
        domain spec you received. For each field, suggest what kind of information it
        holds and what constraints make sense. For example: "The meeting date sounds
        like a specific date — should it be required, or could a meeting not have a
        set date yet? The attendees field could have multiple people — is there a
        minimum number of attendees for a valid meeting? The key decisions should
        probably have some context so someone who missed the meeting can understand
        them — should each decision have a description of a few sentences?"

        Once you've covered the fields, move on to the overall quality rules the
        artifact should enforce. The user has already told you what makes this
        artifact excellent — translate those criteria into concrete rules. For
        example, if the user said "someone who missed the meeting can catch up in two
        minutes", you might propose: "Since the minutes should let someone catch up
        quickly, should we require that every decision includes enough context to
        stand on its own, and that the overall minutes stay concise — say, no more
        than a few paragraphs per topic?" If the user said "every action item has a
        clear owner and a due date", you might propose: "Should we make sure every
        action item has both an owner and a due date, and flag any that are missing
        either one?"

        Present your proposals as a complete picture for the user to react to, not as
        a checklist of individual questions. Let them confirm, adjust, or add to what
        you've suggested. If they confirm something, move on. If they adjust
        something, update your understanding and reflect the change. Do not re-ask
        about things already settled.

        Stay entirely in the user's domain language throughout. Talk about what the
        information means in their work, what rules make their artifact trustworthy,
        what constraints protect its quality. Never mention data types, fields,
        validators, Pydantic, Python, or any technical implementation concepts.

        Do not re-ask or re-confirm what was already settled. If the user confirms
        your proposal, move on to the next topic. If they correct something, update
        your understanding and propose the revised picture — don't ask a follow-up
        question about each correction separately.

        Never put fabricated values in the final output. Only include what the user
        has confirmed. But you can propose ideas in conversation — that's how you
        help them think through what they need.

        When you need to ask the user a question or propose ideas for discussion,
        use "continue" intent with a message only — do not include a result. Only
        use "success" intent when the user has confirmed the complete structure and
        you are ready to return the final result. Never include a result with
        "continue" intent.
        """
        ...  # type: ignore[reportReturnType]
