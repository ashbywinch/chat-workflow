"""A changed draft is shown again; an unchanged draft closes."""

import unittest

from chat_workflow.confirmation import confirm_draft


class _Draft:
    """A minimal Renderable: what the user sees is its text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, context) -> str:
        return self.text


class _Session:
    def __init__(self) -> None:
        self.proposals: list[_Draft] = []
        self.said: list[str] = []

    def propose(
        self,
        result: _Draft,
        *,
        changed: bool = False,
        note: str | None = None,
    ) -> None:
        self.proposals.append(result)

    def say(self, text: str) -> None:
        self.said.append(text)


class TestConfirmDraft(unittest.TestCase):
    def test_an_unchanged_draft_is_not_presented_again(self):
        session = _Session()

        def step(*, draft, session):
            return draft

        result = confirm_draft(session=session, draft=_Draft("one"), step=step)

        self.assertEqual(result.text, "one")
        self.assertEqual(session.proposals, [])
        self.assertEqual(session.said, [])

    def test_a_changed_draft_is_presented_then_confirmed(self):
        session = _Session()
        replies = [_Draft("two"), _Draft("two")]

        def step(*, draft, session):
            return replies.pop(0)

        result = confirm_draft(session=session, draft=_Draft("one"), step=step)

        self.assertEqual(result.text, "two")
        self.assertEqual([p.text for p in session.proposals], ["two"])
        self.assertEqual(session.said, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
