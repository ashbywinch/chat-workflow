"""Component.create orchestrates the phases and writes the artifact.

The phase steps are model conversations; this test replaces them and checks
the wiring: the record, the output file, and the legacy path.
"""

# lucidlint: ignore-file monkeypatch Component.create constructs ComponentDomainSpec, ComponentStructure,
# ComponentInteractionContext, ComponentSourceCode and calls verify_code in its own body — there is no
# injectable seam for that pipeline, and changing production is out of scope here

import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from pyfakefs.fake_filesystem_unittest import TestCase

from chat_workflow import Session, SessionLog
from chat_workflow.session import UserIO
from tests.conftest import FakeConfig
from workflows.workflow.component import Component
from workflows.workflow.component_spec import ComponentSpec
from workflows.workflow.domain_spec import ComponentDomainField, ComponentDomainSpec
from workflows.workflow.interaction_context import ComponentInteractionContext
from workflows.workflow.structure import ComponentStructure


class _IO(UserIO):
    def __init__(self) -> None:
        self.echoes: list[str] = []

    @override
    def echo(self, message: str) -> None:
        self.echoes.append(message)

    @override
    def prompt(self, label: str) -> str:
        return ""


def _session() -> Session:
    return Session(io=_IO(), state=SessionLog(), config=FakeConfig())


def _requirements() -> ComponentSpec:
    return ComponentSpec(
        name="MeetingMinutes",
        purpose="Write up the meeting.",
        scope_description="Not the agenda.",
        owner_role="You",
        required_inputs=["Raw notes"],
        component_type="artifact_producing",
        incidental_notes="",
    )


def _domain() -> ComponentDomainSpec:
    return ComponentDomainSpec(
        name="MeetingMinutes",
        description="The record of a meeting.",
        fields=[
            ComponentDomainField(
                name="decision",
                domain_description="What was agreed",
                field_type_hint="text",
            )
        ],
        what_good_looks_like=["Skimmable"],
        expert_role="Note Taker",
    )


class _Generated:
    code = "class MeetingMinutes:\n    pass\n"


@patch("chat_workflow.mixins.LLMValidated.collect_all_rules", return_value=[])
class TestComponentCreate(TestCase):
    @override
    def setUp(self) -> None:
        self.setUpPyfakefs()

    @patch("workflows.workflow.component.verify_code", return_value="clean code")
    @patch("workflows.workflow.component.ComponentSourceCode")
    @patch("workflows.workflow.component.ComponentInteractionContext")
    @patch("workflows.workflow.component.ComponentStructure")
    @patch("workflows.workflow.component.ComponentDomainSpec")
    def test_the_new_path_writes_the_component(
        self, domain, structure, interaction, source, _verify, _rules
    ):
        domain.explore.return_value = _domain()
        structure.design.return_value = ComponentStructure(description="The artifact")
        interaction.gather.return_value = ComponentInteractionContext(
            must_prioritize=[],
            auto_suggest=[],
            user_pain_points=[],
        )
        source.generate.return_value = _Generated()

        with tempfile.TemporaryDirectory() as directory:
            component = Component.create(
                _requirements(),
                session=_session(),
                output_dir=Path(directory),
            )

            self.assertEqual(component.name, "MeetingMinutes")
            self.assertEqual(component.code_path.read_text(), "clean code")

    @patch("workflows.workflow.component.verify_code")
    @patch("workflows.workflow.component.ComponentSourceCode")
    @patch("workflows.workflow.component.ComponentInteractionContext")
    @patch("workflows.workflow.component.ComponentStructure")
    @patch("workflows.workflow.component.ComponentDomainSpec")
    def test_a_verification_failure_is_raised(
        self, domain, structure, interaction, source, verify, _rules
    ):
        domain.explore.return_value = _domain()
        structure.design.return_value = ComponentStructure(description="The artifact")
        interaction.gather.return_value = ComponentInteractionContext(
            must_prioritize=[],
            auto_suggest=[],
            user_pain_points=[],
        )
        source.generate.return_value = _Generated()
        verify.side_effect = RuntimeError("bad code")

        with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
            Component.create(
                _requirements(),
                session=_session(),
                output_dir=Path(directory),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
