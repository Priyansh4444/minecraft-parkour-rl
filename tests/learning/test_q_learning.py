"""Fixed-proposal checks: a literal table, exact rewards and read-only evaluation."""

import copy
import hashlib
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parkour.learning import q_learning as tabular


def frame(position=(0.5, 1.0, 0.5), **changes):
    result = {
        "position": list(position),
        "velocity": [0.0, -0.0784, 0.0],
        "on_ground": True,
        "health": 20,
        "tick": 10,
        "start_tick": 10,
        "elapsed_ticks": 0,
        "correction_count": 1,
    }
    result.update(changes)
    return result


class Backend:
    def __init__(self, frames):
        self.frames = copy.deepcopy(frames)
        self.requests = []
        self.closed = False

    def _request(self, operation, **fields):
        self.requests.append((operation, fields))
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class TabularTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.course = self.root / "course.json"
        self.course.write_text(
            json.dumps(
                {
                    "start": [0.5, 1, 0.5],
                    "yaw": 0,
                    "platforms": [[0.5, 1, 0.5], [0.5, 1, 2.5]],
                    "fall_y": -1,
                    "max_steps": 3,
                    "script": [3, 1, 1],
                }
            )
        )
        self.profile = tabular.Profile.load(self.course)

    def test_exact_sparse_rewards_and_no_height_or_proximity_shaping(self):
        for shaped in (False, True):
            self.assertEqual(tabular.reward_for("goal", 2, 0, 2, shaped), 1)
            self.assertEqual(tabular.reward_for("fall", None, 0, 2, shaped), -1)
            self.assertEqual(tabular.reward_for("timeout", None, 0, 2, shaped), -0.01)
        self.assertEqual(tabular.reward_for("running", 1, 0, 2, False), -0.01)
        self.assertEqual(tabular.reward_for("running", None, 0, 2, True), -0.01)

    def test_platform_bonus_is_only_for_a_new_grounded_intermediate_platform(self):
        self.assertAlmostEqual(tabular.reward_for("running", 1, 0, 3, True), 0.09)
        self.assertEqual(tabular.reward_for("running", 1, 1, 3, True), -0.01)
        self.assertEqual(tabular.reward_for("running", 1, 2, 3, True), -0.01)
        self.assertIsNone(
            tabular.landed_platform(self.profile, frame((0.5, 1, 2.5), on_ground=False))
        )
        self.assertIsNone(tabular.landed_platform(self.profile, frame((0.5, 2, 2.5))))

    def test_literal_table_update_changes_only_one_full_state_and_action(self):
        a = (1, 2, 3, 4, 5, 1, 0)
        b = (1, 2, 3, 4, 6, 1, 0)
        q = {a: [0.0] * 5, b: [2.0] * 5}
        tabular.update_q(q, a, 3, 1, b, False)
        self.assertAlmostEqual(q[a][3], 0.1 * (1 + 0.99 * 2))
        self.assertEqual(q[a][:3] + q[a][4:], [0.0] * 4)
        self.assertEqual(q[b], [2.0] * 5)
        self.assertEqual(len(q), 2)
        for outcome_reward in (1.0, -1.0):
            terminal = {b: [2.0] * 5}
            tabular.update_q(terminal, a, 0, outcome_reward, b, True)
            self.assertEqual(terminal[a][0], 0.1 * outcome_reward)
            self.assertEqual(terminal[b], [2.0] * 5)

    def test_state_distinguishes_velocity_direction_and_contact(self):
        standing = frame(velocity=[0, 0, 0.15])
        backwards = frame(velocity=[0, 0, -0.15])
        airborne = frame(velocity=[0, 0, 0.15], on_ground=False)
        first = tabular.state_key(self.profile, standing, 0)
        self.assertEqual(len(first), 7)
        self.assertTrue(all(type(x) is int for x in first))
        self.assertNotEqual(first, tabular.state_key(self.profile, backwards, 0))
        self.assertNotEqual(first, tabular.state_key(self.profile, airborne, 0))

    def test_cardinal_lane_has_no_roundoff_sideways_bin(self):
        for yaw, delta in ((0, (0, 2)), (90, (-2, 0)), (180, (0, -2)), (-90, (2, 0))):
            start = [19.5, -6, 13.5]
            target = [start[0] + delta[0], start[1], start[2] + delta[1]]
            profile = tabular.Profile(tuple(start), yaw, (tuple(start), tuple(target)), -8, 80, ())
            state = tabular.state_key(profile, frame(start), 0)
            self.assertEqual(state[0], 8)
            self.assertEqual(state[1], 0)

    def test_episode_resets_to_start_and_goal_requires_ground_contact(self):
        backend = Backend(
            [
                frame(),
                frame((0.5, 1, 2.5), on_ground=False, tick=12, elapsed_ticks=2),
                frame((0.5, 1, 2.5), tick=14, start_tick=12, elapsed_ticks=2),
            ]
        )
        rows = []
        result = tabular.TabularRunner(self.profile, backend).episode(
            {}, random.Random(1), 0, "scripted", lambda **r: rows.append(r)
        )
        self.assertEqual(result["outcome"], "goal")
        self.assertEqual(result["return"], 0.99)
        self.assertEqual(result["max_platform"], 1)
        self.assertEqual(backend.requests[0][1]["start"], [0.5, 1, 0.5])
        self.assertEqual(rows[0]["reward"], -0.01)
        self.assertFalse(rows[0]["terminal"])
        self.assertTrue(rows[1]["terminal"])
        self.assertTrue(rows[1]["terminated"])
        self.assertFalse(rows[1]["truncated"])
        self.assertTrue(rows[1]["episode_end"])
        self.assertEqual(backend.requests[1][1]["action"], [1, 0, 1, 0, 0])

    def test_fall_and_timeout_end_without_extra_penalties(self):
        for frames, outcome, total in (
            ([frame(), frame((0.5, -2, 0.5), tick=12, elapsed_ticks=2)], "fall", -1),
            (
                [
                    frame(),
                    frame(tick=12, elapsed_ticks=2),
                    frame(tick=14, start_tick=12, elapsed_ticks=2),
                    frame(tick=16, start_tick=14, elapsed_ticks=2),
                ],
                "timeout",
                -0.03,
            ),
        ):
            with self.subTest(outcome=outcome):
                rows = []
                result = tabular.TabularRunner(self.profile, Backend(frames)).episode(
                    {}, random.Random(0), 0, "scripted", lambda rows=rows, **r: rows.append(r)
                )
                self.assertEqual(result["outcome"], outcome)
                self.assertAlmostEqual(result["return"], total)
                self.assertTrue(rows[-1]["episode_end"])
                self.assertEqual(rows[-1]["terminal"], outcome == "fall")
                self.assertEqual(rows[-1]["terminated"], outcome == "fall")
                self.assertEqual(rows[-1]["truncated"], outcome == "timeout")

    def test_timeout_keeps_the_nonterminal_next_state_bootstrap(self):
        profile = tabular.Profile((0.5, 1, 0.5), 0, ((0.5, 1, 0.5), (0.5, 1, 2.5)), -1, 1, ())
        initial = frame()
        next_frame = frame((0.5, 1, 1), tick=12, elapsed_ticks=2)
        state = tabular.state_key(profile, initial, 0)
        next_state = tabular.state_key(profile, next_frame, 0)
        table = {next_state: [2.0] * 5}
        rows = []
        result = tabular.TabularRunner(profile, Backend([initial, next_frame])).episode(
            table, random.Random(0), 0, "train", lambda **r: rows.append(r)
        )
        action = tabular.ACTIONS.index(rows[0]["action"])
        self.assertEqual(result["outcome"], "timeout")
        self.assertAlmostEqual(table[state][action], 0.197)
        self.assertEqual(rows[0]["reward"], -0.01)
        self.assertFalse(rows[0]["terminal"])
        self.assertFalse(rows[0]["terminated"])
        self.assertTrue(rows[0]["truncated"])
        self.assertTrue(rows[0]["episode_end"])
        self.assertEqual(table[next_state], [2.0] * 5)

    def test_timeout_with_unvisited_next_state_uses_zero_future_value(self):
        profile = tabular.Profile((0.5, 1, 0.5), 0, ((0.5, 1, 0.5), (0.5, 1, 2.5)), -1, 1, ())
        initial = frame()
        next_frame = frame((0.5, 1, 1), tick=12, elapsed_ticks=2)
        state = tabular.state_key(profile, initial, 0)
        next_state = tabular.state_key(profile, next_frame, 0)
        table, rows = {}, []
        tabular.TabularRunner(profile, Backend([initial, next_frame])).episode(
            table, random.Random(0), 0, "train", lambda **r: rows.append(r)
        )
        action = tabular.ACTIONS.index(rows[0]["action"])
        self.assertAlmostEqual(table[state][action], -0.001)
        self.assertNotIn(next_state, table)
        self.assertTrue(all(math.isfinite(value) for value in table[state]))

    def test_bad_duration_correction_and_reset_rejected_without_learning(self):
        cases = (
            [frame((0.5, 1, 2.5))],
            [frame(), frame(tick=13, elapsed_ticks=3)],
            [frame(), frame(tick=12, elapsed_ticks=2, correction_count=2)],
        )
        for frames in cases:
            q = {}
            with self.assertRaises(RuntimeError):
                tabular.TabularRunner(self.profile, Backend(frames)).episode(
                    q, random.Random(0), 1, "train", lambda **r: None
                )
            self.assertEqual(q, {})

    def test_checkpoint_schema_and_reward_mode_are_strict(self):
        path = self.root / "q.json"
        q = {(1, 2, 3, 4, 5, 1, 0): [0.1] * 5}
        tabular.save_table(path, q, False)
        self.assertEqual(tabular.load_table(path, False), q)
        with self.assertRaises(ValueError):
            tabular.load_table(path, True)
        with self.assertRaises(ValueError):
            tabular.load_table(path, False, ticks=4)
        old_contract = json.loads(path.read_text())
        old_contract["schema"]["version"] = 2
        path.write_text(json.dumps(old_contract))
        with self.assertRaises(ValueError):
            tabular.load_table(path, False)
        path.write_text(json.dumps({"algorithm": "linear-q", "q": {"bias": [1] * 5}}))
        with self.assertRaises(ValueError):
            tabular.load_table(path, False)

    def test_train_then_eval_random_scripted_never_change_checkpoint(self):
        prefix = self.root / "result"
        args = ["--course", str(self.course), "--output", str(prefix), "--episodes", "1"]

        def backend(*args):
            return Backend([frame(), frame((0.5, 1, 2.5), tick=12, elapsed_ticks=2)])

        with (
            patch.object(tabular, "MineflayerBackend", side_effect=backend),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            tabular.main(args)
            checkpoint = prefix.with_suffix(".json")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for mode in ("--eval", "--random", "--scripted"):
                tabular.main(args + [mode])
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                tabular.main(args)

    def test_invalid_course_and_missing_script_rejected_before_connection(self):
        data = json.loads(self.course.read_text())
        for changes in (
            {"yaw": 45},
            {"start": [True, 1, 0.5]},
            {"fall_y": 1},
            {"platforms": [[0.5, 1, 0.5], [1.5, 1, 2.5]]},
            {"max_steps": True},
        ):
            invalid = self.root / "invalid.json"
            invalid.write_text(json.dumps({**data, **changes}))
            with self.assertRaises(ValueError):
                tabular.Profile.load(invalid)
        self.course.write_text(json.dumps({**data, "script": []}))
        with (
            patch.object(tabular, "MineflayerBackend") as connection,
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            with self.assertRaises(SystemExit):
                tabular.main(
                    [
                        "--course",
                        str(self.course),
                        "--output",
                        str(self.root / "empty"),
                        "--scripted",
                    ]
                )
            connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
