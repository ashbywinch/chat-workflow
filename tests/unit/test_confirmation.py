"""A changed draft is shown again; an unchanged draft closes."""

import unittest

from chat_workflow.confirmation import confirm_draft


class _Session:
    def __init__(self) -> None:
        self.proposals: list[object] = []
        self.said: list[str] = []

    def propose(self, result: object) -> None:
        self.proposals.append(result)

    def say(self, text: str) -> None:
        self.said.append(text)


class TestConfirmDraft(unittest.TestCase):
    def test_an_unchanged_draft_is_not_presented_again(self):
        session = _Session()

        def step(*, draft, session, max_turns):
            return draft

        result = confirm_draft(session=session, draft={"a": 1}, step=step)

        self.assertEqual(result, {"a": 1})
        self.assertEqual(session.proposals, [])
        self.assertEqual(session.said, [])

    def test_a_changed_draft_is_presented_then_confirmed(self):
        session = _Session()
        replies = [{"a": 2}, {"a": 2}]

        def step(*, draft, session, max_turns):
            return replies.pop(0)

        result = confirm_draft(session=session, draft={"a": 1}, step=step)

        self.assertEqual(result, {"a": 2})
        self.assertEqual(session.proposals, [{"a": 2}])
        self.assertEqual(session.said, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
