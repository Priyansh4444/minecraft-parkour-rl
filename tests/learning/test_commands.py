"""The learner command must work before a server is started."""

import subprocess
import sys
import unittest
from pathlib import Path


class LearningCommandTests(unittest.TestCase):
    def test_q_learning_help_works_without_a_server(self):
        result = subprocess.run(
            [sys.executable, "-m", "parkour.learning.q_learning", "--help"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--course", result.stdout)
        self.assertIn("--eval", result.stdout)
        self.assertIn("--scripted", result.stdout)


if __name__ == "__main__":
    unittest.main()
