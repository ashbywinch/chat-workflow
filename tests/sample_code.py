"""Shared sample code snippets for testing GeneratedComponent validation.

These are realistic code samples used by multiple test files. Keeping them
in one place avoids duplicating multi-line strings across tests.
"""

VALID_COMPONENT_CODE = '''"""Meeting minutes component."""
from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow


class MinutesDraft(BaseModel):
    """Structured meeting minutes."""

    meeting_date: str = Field(..., description="When the meeting took place")
    attendees: list[str] = Field(..., description="People who attended", min_length=1)

    @atomic_workflow
    @classmethod
    def create(cls, context: str, max_turns: int = 10) -> MinutesDraft:
        """Create meeting minutes by discussing with the user."""
        ...
'''

INVALID_SYNTAX_CODE = """from pydantic import BaseModel

class Broken(BaseModel):
    name: str = Field(..., description="Name"
"""

NO_BASEMODEL_CODE = '''"""Just a utility module."""
from __future__ import annotations

from chat_workflow import atomic_workflow


def helper():
    pass


@atomic_workflow
@classmethod
def some_method(cls):
    ...
'''

MULTIPLE_BASEMODEL_CODE = '''from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow


class Invoice(BaseModel):
    """An invoice."""
    amount: float = Field(..., description="Invoice amount")

    @atomic_workflow
    @classmethod
    def create(cls, context: str):
        ...


class Receipt(BaseModel):
    """A receipt — different artifact type."""
    date: str = Field(..., description="Receipt date")
'''

NO_WORKFLOW_CODE = '''from __future__ import annotations

from pydantic import BaseModel, Field


class MinutesDraft(BaseModel):
    """Structured meeting minutes."""

    meeting_date: str = Field(..., description="When the meeting took place")
    attendees: list[str] = Field(..., description="People who attended", min_length=1)

    @classmethod
    def create(cls, context: str):
        """Create meeting minutes."""
        ...
'''

LOW_COHESION_CODE = '''"""Mixed-domain component."""
from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow


class MixedBag(BaseModel):
    """Some kind of business thing."""

    invoice_number: str = Field(..., description="Invoice number")
    employee_salary: float = Field(..., description="Employee salary")
    meeting_room: str = Field(..., description="Meeting room name")

    @atomic_workflow
    @classmethod
    def create(cls, context: str):
        ...
'''

INCOMPLETE_CODE = '''"""Minimal component."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Skeleton(BaseModel):
    """A bare-bones model."""

    name: str = Field(..., description="Name")
'''

VERBOSE_CODE = '''"""Generic component."""
from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow


class GenericThing(BaseModel):
    """Represents a business concept that captures important information
    about various aspects of organizational operations and processes
    across multiple domains and functional areas within the enterprise."""

    name: str = Field(..., description="Name")

    @atomic_workflow
    @classmethod
    def create(cls, context: str):
        """Creates instances of this business concept through conversation
        with the user about their needs and requirements and preferences."""
        ...
'''

MULTI_ARTIFACT_CODE = '''"""Multi-artifact component."""
from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow


class Invoice(BaseModel):
    """An invoice."""
    amount: float = Field(..., description="Amount")

    @atomic_workflow
    @classmethod
    def create(cls, context: str):
        ...


class Timesheet(BaseModel):
    """A timesheet — completely different artifact."""
    hours: float = Field(..., description="Hours worked")
'''
