from .deliverable import Deliverable
from .gap_analysis import GapAnalysis
from .process_definition import ProcessDefinition, generate_from_chat
from .resource import Resource

__all__ = [
    "Deliverable",
    "GapAnalysis",
    "ProcessDefinition",
    "Resource",
    "generate_from_chat",
    "ComponentRequirement",
]


def __getattr__(name: str):
    """Lazy import ComponentRequirement to avoid circular imports.

    ``workflows/workflow/component_responsibilities.py`` imports from
    ``.models.resource`` etc., which triggers this ``__init__.py``. If we
    imported ``ComponentRequirement`` at module level, the import of
    ``component_responsibilities`` would be half-finished → circular import.
    """
    if name == "ComponentRequirement":
        from ..component_responsibilities import ComponentRequirement

        return ComponentRequirement
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
