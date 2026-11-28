"""Resume component generation from a saved ComponentSpec checkpoint.

This is not the end-to-end gate. It re-runs only the component phase from
the specs the pipeline test captured, so a component failure can be iterated
on in minutes instead of re-running the whole workflow pipeline.

Environment:
    RESUME_SPECS  Path to the checkpoint JSON (required; without it the
                  test skips so the normal eval suite does not run it)
    RESUME_NAMES  Optional comma-separated component names to resume
"""

# lucidlint: ignore-file fakefs Component.create's write of generated Python is the behaviour under test

import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.conftest import timeout


class TestResumeComponentGeneration(unittest.TestCase):
    """Generate only the components named in a checkpoint."""

    @timeout(900)
    def test_resume_from_checkpoint(self):
        from tests.evals.helpers import AgentIO, make_config, make_tools
        from tests.evals.personas import MEETING_MINUTES_PERSONA
        from workflows.workflow.component import Component
        from workflows.workflow.component_spec import ComponentSpec

        checkpoint_path = os.environ.get("RESUME_SPECS")
        if not checkpoint_path:
            self.skipTest("Set RESUME_SPECS to a checkpoint to resume")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            self.fail(f"Checkpoint not found: {checkpoint}")

        specs = [
            ComponentSpec.model_validate(item)
            for item in json.loads(checkpoint.read_text())
        ]
        names = {
            name.strip()
            for name in os.environ.get("RESUME_NAMES", "").split(",")
            if name.strip()
        }
        if names:
            specs = [spec for spec in specs if spec.name in names]
        self.assertTrue(specs, f"No component specs matched {checkpoint}")

        config = make_config()
        user_bot = AgentIO(persona_prompt=MEETING_MINUTES_PERSONA, config=config)
        session = make_tools(user_bot)

        failures: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for spec in specs:
                try:
                    component = Component.create(
                        requirements=spec,
                        session=session,
                        output_dir=Path(tmpdir),
                    )
                except Exception as exc:  # noqa: BLE001 — report every failure
                    failures.append(f"{spec.name}: {exc}")
                    continue
                self.assertEqual(component.name, spec.name)

        self.assertFalse(
            failures,
            f"{len(failures)}/{len(specs)} components failed:\n"
            + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
