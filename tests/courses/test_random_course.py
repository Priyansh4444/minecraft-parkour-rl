import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from parkour.learning.q_learning import Profile
from tools.courses.prepare_random_course import generate, main, simple_script


class RandomCourseTests(unittest.TestCase):
    def test_reproducible_valid_profiles_and_geometry(self):
        self.assertEqual(generate(), generate(17501, 3))
        self.assertNotEqual(generate(1), generate(2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "course.json"
            for seed in range(40):
                for jumps in (1, 3, 5):
                    profile, provenance, commands = generate(seed, jumps)
                    path.write_text(json.dumps(profile))
                    loaded = Profile.load(path)
                    self.assertEqual(len(loaded.platforms), jumps + 1)
                    for choice in provenance["choices"]:
                        self.assertIn(choice["center_distance"], (2, 3, 4))
                        self.assertIn(choice["height_change"], (0, 1))
                        if choice["center_distance"] == 4:
                            self.assertEqual(choice["height_change"], 0)
                    self.assertEqual(len(commands), jumps + 3)

    def test_build_commands_are_bounded_and_support_all_platforms(self):
        profile, provenance, commands = generate()
        low, high = provenance["clear_bounds"]
        self.assertTrue(
            all(
                970 <= p[0] <= 1003 and 77 <= p[1] <= 89 and 997 <= p[2] <= 1003
                for p in (low, high)
            )
        )
        volume = (high[0] - low[0] + 1) * (high[1] - low[1] + 1) * (high[2] - low[2] + 1)
        self.assertLess(volume, 32768)
        self.assertEqual(commands[0], f"forceload add {low[0]} {low[2]} {high[0]} {high[2]}")
        self.assertEqual(
            commands[1], f"fill {' '.join(map(str, low))} {' '.join(map(str, high))} minecraft:air"
        )
        for platform, command in zip(profile["platforms"], commands[2:], strict=True):
            _, x, y, z, block = command.split()
            point = list(map(int, (x, y, z)))
            self.assertTrue(all(a <= p <= b for a, p, b in zip(low, point, high, strict=True)))
            self.assertEqual([int(x) + 0.5, int(y) + 1, int(z) + 0.5], platform)
            self.assertEqual(block, "minecraft:green_wool")

    def test_baseline_is_constant_and_does_not_mutate_profile(self):
        profile = dict(start=[0, 0, 0], platforms=[[0, 0, 0]], max_steps=80, script=[1, 2, 3])
        original = copy.deepcopy(profile)
        result = simple_script(profile)
        self.assertEqual(profile, original)
        result["start"][0] = 100
        self.assertEqual(profile, original)
        self.assertEqual(result["script"], [4] * 80)
        self.assertEqual(generate(1)[0]["script"], generate(2)[0]["script"])

    def test_cli_freezes_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frozen"
            main(["--output", str(output)])
            manifest = json.loads((output / "manifest.json").read_text())
            before = {path: path.read_bytes() for path in output.iterdir()}
            for name, value in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), value)
            with self.assertRaises(FileExistsError):
                main(["--output", str(output), "--seed", "2"])
            self.assertEqual(before, {path: path.read_bytes() for path in output.iterdir()})

    def test_invalid_generator_arguments(self):
        for seed, jumps in ((-1, 3), (True, 3), (1, 0), (1, 6), (1, True)):
            with self.assertRaises(ValueError):
                generate(seed, jumps)


if __name__ == "__main__":
    unittest.main()
