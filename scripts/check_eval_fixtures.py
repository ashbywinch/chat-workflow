#!/usr/bin/env python3
"""Construct every eval fixture and report which ones fail their own rules.

Fixtures are inputs to a step, but they are built from the same models the
judge validates, so an invalid fixture breaks an eval before the step under
test runs. Run this after changing a validation rule or a fixture:

    make fixtures-check

A failure here means fix the fixture. Passing invalid input to a step tests the
wrong thing, and disabling validation on fixtures would hide the mistake.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Running a script puts its own directory on sys.path, not the repository root.
# The fixtures import `tests` and `workflows`, so make the checkout importable
# the same way `python -m unittest` does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.evals import fixtures  # noqa: E402 — follows the sys.path insert above


class FixtureCatalogue:
    """The fixture factories whose output every eval input must satisfy."""

    NAMES = [
        "meeting_minutes_deliverable",
        "meeting_notes_resource",
        "family_dietary_preferences",
        "weekly_meal_plan_deliverable",
        "weekly_menu_deliverable",
        "meal_planning_analysis",
        "meeting_analysis",
        "weekly_meal_plan_spec",
        "minutes_draft_spec",
        "meeting_minutes_spec",
    ]


def main() -> int:
    failures = 0
    for name in FixtureCatalogue.NAMES:
        factory = getattr(fixtures, name)
        try:
            factory()
        # lucidlint: ignore broad-except each factory validates a different model; report every failure, keep going
        except Exception as exc:  # noqa: BLE001 — report, do not stop
            failures += 1
            print(f"FAIL {name}: {exc}\n")
            if "--traceback" in sys.argv:
                traceback.print_exc()
        else:
            print(f"OK   {name}")
    print(f"\n{len(FixtureCatalogue.NAMES) - failures}/{len(FixtureCatalogue.NAMES)} fixtures valid")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
