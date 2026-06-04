"""ComponentDesignSpec — assembled complete design type.

ComponentDesignSpec bundles the three design outputs (domain spec, structure,
and interaction context) into a single parameter for GeneratedComponent.generate().
"""

from pydantic import BaseModel

from .domain_spec import ComponentDomainSpec
from .interaction_context import ComponentInteractionContext
from .structure import ComponentStructure


class ComponentDesignSpec(BaseModel):
    """Assembled complete design for a generated component.

    Combines the domain-level specification, Pydantic structure, and
    interaction context into one composite type. This is the single
    parameter passed to GeneratedComponent.generate().
    """

    domain_spec: ComponentDomainSpec
    """Domain-level understanding of the business artifact."""

    structure: ComponentStructure
    """Pydantic model structure for the generated component."""

    interaction_context: ComponentInteractionContext
    """Domain considerations for assistant-user interaction."""
