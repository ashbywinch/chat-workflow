"""Every relative markdown link in the docs set resolves.

The check itself lives in `scripts/check_docs_links.py` — the same code
`make docs-check` runs. Deriving the repo root from the script's location keeps
the check working when this test module moves between packages.
"""

import unittest

from scripts.check_docs_links import find_broken_links


class TestDocLinks(unittest.TestCase):
    def test_all_relative_links_resolve(self):
        failures = find_broken_links()
        self.assertEqual(failures, [], f"\n{chr(10).join(failures)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
