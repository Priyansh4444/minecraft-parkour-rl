"""The experiment runner must apply one untuned script to every course."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.experiments import run_tabular_experiment as experiment


class BaselineDispatchTests(unittest.TestCase):
    def test_all_seeds_and_courses_use_constant_script_without_editing_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = dict(
                start=[0.5, 1, 0.5],
                yaw=0,
                platforms=[[0.5, 1, 0.5], [0.5, 1, 2.5]],
                fall_y=-1,
                max_steps=80,
                script=[],
            )
            train, heldout = root / "train.json", root / "heldout.json"
            train.write_text(json.dumps(profile))
            heldout.write_text(
                json.dumps(
                    dict(profile, platforms=[[0.5, 1, 0.5], [0.5, 2, 3.5]], script=[1, 2, 3])
                )
            )
            before = {path: path.read_bytes() for path in (train, heldout)}
            commands = []

            class Process:
                returncode = 0

                def __init__(self, command, **kwargs):
                    commands.append(command)

                def wait(self, **kwargs):
                    return 0

                def poll(self):
                    return 0

            output = root / "results"
            argv = [
                "experiment",
                "--course",
                str(train),
                "--held-out",
                str(heldout),
                "--output",
                str(output),
                "--train-episodes",
                "1",
                "--eval-episodes",
                "1",
            ]
            with patch("sys.argv", argv), patch.object(experiment.subprocess, "Popen", Process):
                experiment.main()
            scripts = [command for command in commands if "--scripted" in command]
            self.assertEqual(len(scripts), 6)
            for command in scripts:
                actual = json.loads(Path(command[command.index("--course") + 1]).read_text())
                self.assertEqual(actual["script"], [4] * 80)
                self.assertIn(
                    actual["platforms"], [json.loads(p.read_text())["platforms"] for p in before]
                )
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("simple-v1", manifest["scripted_baseline"])
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "completed")


if __name__ == "__main__":
    unittest.main()
