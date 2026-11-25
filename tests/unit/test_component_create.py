"""Component.create orchestrates the phases and writes the artifact.

The phase steps are model conversations; this test injects object fakes and
checks the wiring: the record, the output file, and the failure path.
"""

import unittest
from pathlib import Path
from typing import override

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


class _FakeDomainSpec:
    @staticmethod
    def explore(**_kwargs) -> ComponentDomainSpec:
        return _domain()


class _FakeStructure:
    @staticmethod
    def design(**_kwargs) -> ComponentStructure:
        return ComponentStructure(description="The artifact")


class _FakeInteraction:
    @staticmethod
    def gather(**_kwargs) -> ComponentInteractionContext:
        return ComponentInteractionContext(
            must_prioritize=[],
            auto_suggest=[],
            user_pain_points=[],
        )


class _FakeSource:
    @staticmethod
    def generate(**_kwargs):
        return _Generated()


def _clean(_code: str) -> str:
    return "clean code"


def _fails(_code: str) -> str:
    raise RuntimeError("bad code")


class _NoRuleJudge(TestCase):
    """Construct real records without the LLM judge, using the documented skip flag."""

    _MODELS = (ComponentSpec, ComponentDomainSpec, Component)

    @override
    def setUp(self) -> None:
        self.setUpPyfakefs()
        self._saved_skip = {model: model._skip_llm_validation for model in self._MODELS}
        for model in self._MODELS:
            model._skip_llm_validation = True

    @override
    def tearDown(self) -> None:
        for model, value in self._saved_skip.items():
            model._skip_llm_validation = value


class TestComponentCreate(_NoRuleJudge):
    def _create(self, verifier):
        return Component.create(
            _requirements(),
            session=_session(),
            output_dir=Path("/out"),
            domain_spec=_FakeDomainSpec,
            structure=_FakeStructure,
            interaction=_FakeInteraction,
            source=_FakeSource,
            verifier=verifier,
        )

    def test_the_new_path_writes_the_component(self):
        component = self._create(_clean)

        self.assertEqual(component.name, "MeetingMinutes")
        self.assertEqual(component.code_path.read_text(), "clean code")

    def test_a_verification_failure_is_raised(self):
        with self.assertRaises(RuntimeError):
            self._create(_fails)


if __name__ == "__main__":
    unittest.main(verbosity=2)
