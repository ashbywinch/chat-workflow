"""Shared sample code snippets for testing ComponentSourceCode validation.

These are realistic code samples used by multiple test files. Keeping them
in one place avoids duplicating multi-line strings across tests.
"""

VALID_COMPONENT_CODE = '''"""Meeting minutes component."""
from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class MinutesDraft(LLMValidated):
    """Structured meeting minutes."""

    _validation_rules: ClassVar[list[str]] = [
        "The meeting date is a real date the meeting took place, not a placeholder.",
    ]

    meeting_date: str = Field(
        ..., description="When the meeting took place", min_length=1
    )
    attendees: list[str] = Field(..., description="People who attended", min_length=1)

    @atomic_workflow_step
    @classmethod
    def create(cls, context: str, max_turns: int = 10) -> MinutesDraft:
        """You help someone turn sketchy notes into minutes. Start from what
        they already told you; ask only for what is missing. Suggest sensible
        defaults where the user is unsure, and draw out details they only
        implied.

        When you have enough, show the draft once and ask if it is right. On
        confirmation, return success.
        """
        ...
'''

INVALID_SYNTAX_CODE = """from pydantic import BaseModel

class Broken(BaseModel):
    name: str = Field(..., description="Name"
"""

NO_BASEMODEL_CODE = '''"""Just a utility module."""
from __future__ import annotations

from chat_workflow import atomic_workflow_step


def helper():
    pass


@atomic_workflow_step
@classmethod
def some_method(cls):
    ...
'''

MULTIPLE_BASEMODEL_CODE = '''from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow_step


class Invoice(BaseModel):
    """An invoice."""
    amount: float = Field(..., description="Invoice amount")

    @atomic_workflow_step
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

LOW_COHESION_CODE = '''"""Meeting minutes with cross-cutting concerns."""
from __future__ import annotations

from datetime import date
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class ActionItem(BaseModel):
    """A task assigned during the meeting."""

    description: str = Field(..., description="What needs to be done", min_length=1)
    owner: str = Field(..., description="Who owns it", min_length=1)
    due_date: str = Field(..., description="When it is due", min_length=1)


class Invoice(BaseModel):
    """An invoice that came up during the meeting."""

    number: str = Field(..., description="Invoice number", min_length=1)
    amount: float = Field(..., description="Invoice amount", ge=0)


class PhoneNumberParser:
    """Normalize attendee phone numbers."""

    @staticmethod
    def normalize(value: str) -> str:
        """Keep digits and a leading plus."""
        return "".join(ch for ch in value if ch.isdigit() or ch == "+")


class MeetingMinutes(LLMValidated):
    """Structured meeting minutes."""

    _validation_rules: ClassVar[list[str]] = [
        "Every decision names what was decided and why.",
    ]

    meeting_date: str = Field(..., description="When the meeting took place", min_length=1)
    attendees: list[str] = Field(..., description="People who attended", min_length=1)
    decisions: list[str] = Field(..., description="Decisions made", min_length=1)
    action_items: list[ActionItem] = Field(..., description="Action items", min_length=1)

    calendar_event_id: str = Field(..., description="Calendar event identifier", min_length=1)
    email_recipients: list[str] = Field(..., description="People to notify by email", min_length=1)
    pdf_storage_path: str = Field(..., description="Where the PDF is stored", min_length=1)
    tenant_id: str = Field(..., description="Tenant that owns this record", min_length=1)
    audit_log: list[str] = Field(..., description="Audit trail entries", min_length=1)
    retry_count: int = Field(0, description="Number of delivery retries")
    related_invoice: Invoice = Field(..., description="Invoice discussed in the meeting")
    attendee_phones: list[str] = Field(..., description="Attendee phone numbers", min_length=1)

    @model_validator(mode="after")
    def _meeting_date_is_iso(self) -> MeetingMinutes:
        try:
            date.fromisoformat(self.meeting_date)
        except ValueError as exc:
            raise ValueError("meeting_date must be an ISO date") from exc
        return self

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MeetingMinutes:
        """You help someone turn sketchy notes into minutes. Start from what
        they already told you; ask only for what is missing. When you have
        enough, show the draft once and ask if it is right. On confirmation,
        return success."""
        ...
'''

INCOMPLETE_CODE = '''"""Minimal component."""
from __future__ import annotations

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step


class Skeleton(BaseModel):
    """A bare-bones model."""

    name: str = Field(..., description="Name")

    @atomic_workflow_step
    @classmethod
    def create(cls) -> Skeleton:
        """You help someone name a thing. Ask what it is called, then return
        success."""
        ...
'''

MULTI_ARTIFACT_CODE = '''"""Multi-artifact component."""
from __future__ import annotations

from pydantic import BaseModel, Field
from chat_workflow import atomic_workflow_step


class Invoice(BaseModel):
    """An invoice."""
    amount: float = Field(..., description="Amount")

    @atomic_workflow_step
    @classmethod
    def create(cls, context: str):
        ...


class Timesheet(BaseModel):
    """A timesheet — completely different artifact."""
    hours: float = Field(..., description="Hours worked")
'''

NESTED_BASEMODEL_CODE = '''"""Meeting minutes with a nested action item."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class ActionItem(BaseModel):
    """A single action item."""

    description: str = Field(..., description="What needs to be done", min_length=1)
    owner: str = Field(..., description="Who owns it", min_length=1)


class MinutesDraft(LLMValidated):
    """Structured meeting minutes."""

    _validation_rules: ClassVar[list[str]] = [
        "Every action item names what needs to be done and who owns it.",
    ]

    meeting_date: str = Field(..., description="When the meeting took place", min_length=1)
    action_items: list[ActionItem] = Field(..., description="Action items", min_length=1)

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MinutesDraft:
        """Help someone create meeting minutes."""
        ...
'''

SECOND_ARTIFACT_CODE = '''"""Meeting minutes plus an unrelated invoice."""
from __future__ import annotations

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step


class Invoice(BaseModel):
    """An invoice."""

    amount: float = Field(..., description="Invoice amount")


class MinutesDraft(BaseModel):
    """Structured meeting minutes."""

    meeting_date: str = Field(..., description="When the meeting took place", min_length=1)

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MinutesDraft:
        """Help someone create meeting minutes."""
        ...
'''

MULTIPLE_RETURN_TYPES_CODE = '''"""Two independent artifacts with workflows."""
from __future__ import annotations

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step


class Invoice(BaseModel):
    """An invoice."""

    amount: float = Field(..., description="Invoice amount")

    @atomic_workflow_step
    @classmethod
    def create(cls) -> Invoice:
        """Create an invoice."""
        ...


class Receipt(BaseModel):
    """A receipt."""

    total: float = Field(..., description="Receipt total")

    @atomic_workflow_step
    @classmethod
    def create(cls) -> Receipt:
        """Create a receipt."""
        ...
'''

CREATE_UPDATE_CODE = '''"""Meeting minutes with create and update methods."""
from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class MeetingMinutes(LLMValidated):
    """Structured meeting minutes."""

    _validation_rules: ClassVar[list[str]] = [
        "Every decision names what was decided and why.",
    ]

    meeting_date: str = Field(
        ..., description="When the meeting took place", min_length=1
    )
    decisions: list[str] = Field(..., description="Decisions made", min_length=1)

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MeetingMinutes:
        """You help someone draft minutes. Ask what was decided, show the
        draft once, and return success on confirmation."""
        ...

    @atomic_workflow_step
    @classmethod
    def update(cls, minutes: MeetingMinutes) -> MeetingMinutes:
        """You help someone correct existing minutes. Ask what should change,
        show the result once, and return success on confirmation."""
        ...
'''

AUXILIARY_RETURN_CODE = '''"""Meeting minutes whose methods also return a helper type."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class RevisionSummary(BaseModel):
    """A short report of what a revision changed."""

    summary: str = Field(..., description="What changed", min_length=1)


class MeetingMinutes(LLMValidated):
    """Structured meeting minutes."""

    _validation_rules: ClassVar[list[str]] = [
        "Every decision names what was decided and why.",
    ]

    meeting_date: str = Field(
        ..., description="When the meeting took place", min_length=1
    )
    decisions: list[str] = Field(..., description="Decisions made", min_length=1)

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MeetingMinutes:
        """You help someone draft minutes. Ask what was decided, show the
        draft once, and return success on confirmation."""
        ...

    @atomic_workflow_step
    @classmethod
    def update(cls, minutes: MeetingMinutes) -> MeetingMinutes:
        """You help someone correct existing minutes. Ask what should change,
        show the result once, and return success on confirmation."""
        ...

    @atomic_workflow_step
    @classmethod
    def summarize_revision(cls) -> RevisionSummary:
        """You help someone describe what a revision changed. Ask what
        changed, then return success."""
        ...
'''


NOOP_VALIDATOR_CODE = '''"""Meeting minutes with a rule that is never enforced."""
from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator

from chat_workflow import atomic_workflow_step
from chat_workflow.mixins import LLMValidated


class MinutesDraft(LLMValidated):
    """Structured meeting minutes."""

    _validation_rules: ClassVar[list[str]] = [
        "Approval notes must be filled if any conditions were discussed.",
    ]

    meeting_date: str = Field(
        ..., description="When the meeting took place", min_length=1
    )
    approval_notes: str = Field(
        "", description="Notes about the approval", min_length=0
    )

    @model_validator(mode="after")
    def _require_notes(self) -> MinutesDraft:
        if not self.approval_notes:
            pass
        return self

    @atomic_workflow_step
    @classmethod
    def create(cls) -> MinutesDraft:
        """You help someone draft minutes. Ask what was decided, show the
        draft once, and return success on confirmation."""
        ...
'''
