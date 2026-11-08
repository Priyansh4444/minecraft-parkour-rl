"""Compact PPO has the same information, actions and episode contract as Q-learning."""

import copy
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from parkour.learning import ppo
from parkour.learning import q_learning as tabular


def frame(position=(0.5, 1.0, 0.5), **changes):
    result = dict(
        position=list(position),
        velocity=[0.0, -0.0784, 0.0],
        on_ground=True,
        health=20,
        tick=10,
        start_tick=10,
        elapsed_ticks=0,
        correction_count=1,
    )
    result.update(changes)
    return result


def step_frame(position=(0.5, 1.0, 0.5), **changes):
    return frame(position, **(dict(tick=12, elapsed_ticks=2) | changes))


class Backend:
    def __init__(self, frames=None):
        self.frames = copy.deepcopy(frames)
        self.requests, self.closed = [], False

    def _request(self, operation, **fields):
        self.requests.append((operation, fields))
        if self.frames is not None:
            return self.frames.pop(0)
        return frame() if operation == "parkour_reset" else step_frame()

    def close(self):
        self.closed = True


class CompactPPOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.course = self.root / "course.json"
        self.course.write_text(
            json.dumps(
                dict(
                    start=[0.5, 1, 0.5],
                    yaw=0,
                    platforms=[[0.5, 1, 0.5], [0.5, 1, 2.5]],
                    fall_y=-1,
                    max_steps=1,
                    script=[3],
                )
            )
        )
        self.profile = tabular.Profile.load(self.course)
        torch.set_num_threads(1)

    def test_all_bin_values_round_trip_including_negative_bounds(self):
        for dimension in range(7):
            for value in range(int(ppo.LOW[dimension]), int(ppo.HIGH[dimension]) + 1):
                state = ppo.LOW.copy()
                state[dimension] = value
                observation = ppo.encode(state)
                self.assertEqual(ppo.decode(observation), tuple(state))
                self.assertTrue(np.isfinite(observation).all())
                self.assertTrue(np.all(np.abs(observation) <= 1))
        extreme = frame((1e6, 1e6, 1e6), velocity=[-1e6, 1e6, 1e6])
        self.assertEqual(
            ppo.decode(ppo.encode(tabular.state_key(self.profile, extreme, 0))),
            tabular.state_key(self.profile, extreme, 0),
        )
        with self.assertRaises(ValueError):
            ppo.encode([float("nan")] * 7)
        with self.assertRaises(ValueError):
            ppo.encode([0.5] * 7)

    def parity(self, profile, frames, shaped=False):
        q_backend, backend = Backend(frames), Backend(frames)
        q_rows, rows = [], []
        expected = tabular.TabularRunner(profile, q_backend, shaped=shaped).episode(
            {}, random.Random(11), 0, "scripted", lambda **row: q_rows.append(row)
        )
        env = ppo.CompactParkourEnv(
            profile, backend=backend, shaped=shaped, log=lambda **row: rows.append(row)
        )
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=11)
        for step in range(profile.max_steps):
            action = profile.script[step] if step < len(profile.script) else 0
            self.assertEqual(ppo.decode(observation), tuple(q_rows[step]["state"]))
            observation, reward, terminated, truncated, info = env.step(action)
            row = q_rows[step]
            self.assertEqual(ppo.decode(observation), tuple(row["next_state"]))
            for name, value in (
                ("reward", reward),
                ("terminated", terminated),
                ("truncated", truncated),
            ):
                self.assertEqual(value, row[name])
            self.assertEqual(info["highest_platform"], row["highest_platform"])
            self.assertEqual(
                info["progress"], row["highest_platform"] / (len(profile.platforms) - 1)
            )
            self.assertEqual(info["outcome"], row["outcome"])
            if terminated or truncated:
                break
        for key in (
            "outcome",
            "success",
            "return",
            "max_platform",
            "progress",
            "steps",
            "client_ticks",
        ):
            self.assertEqual(info["summary"][key], expected[key])
        self.assertEqual(q_backend.requests, backend.requests)
        transitions = [r for r in rows if r["event"] == "transition"]
        for q, p in zip(q_rows, transitions, strict=True):
            for key in q:
                self.assertEqual(p[key], q[key])
        return transitions

    def test_same_observation_and_episode_contract_for_goal_fall_and_timeout(self):
        for after in (
            step_frame((0.5, 1, 2.5)),
            step_frame((0.5, -2, 0.5)),
            step_frame(),
            step_frame((0.5, 1, 2.5), on_ground=False),
            step_frame(health=0),
        ):
            with self.subTest(after=after):
                self.parity(self.profile, [frame(), after])

    def test_intermediate_bonus_only_once_and_highest_does_not_regress(self):
        profile = tabular.Profile(
            (0.5, 1, 0.5), 0, ((0.5, 1, 0.5), (0.5, 1, 2.5), (0.5, 1, 4.5)), -1, 4, (3, 1, 1, 4)
        )
        frames = [
            frame(),
            step_frame((0.5, 1, 2.5)),
            step_frame((0.5, 1, 2.5)),
            step_frame(),
            step_frame((0.5, 1, 4.5)),
        ]
        for shaped in (False, True):
            rows = self.parity(profile, frames, shaped)
            self.assertEqual([row["highest_platform"] for row in rows], [1, 1, 1, 2])
            self.assertEqual(
                [row["reward"] for row in rows],
                [(-0.01 + 0.1) if shaped else -0.01, -0.01, -0.01, 1.0],
            )

    def test_bad_duration_correction_and_frame_abort_without_transition(self):
        invalid = [
            step_frame(elapsed_ticks=1),
            step_frame(correction_count=2),
            step_frame(velocity=[float("nan"), 0, 0]),
        ]
        for after in invalid:
            backend, rows = Backend([frame(), after]), []
            env = ppo.CompactParkourEnv(
                self.profile, backend=backend, log=lambda rows=rows, **r: rows.append(r)
            )
            env.reset()
            with self.assertRaises((ValueError, RuntimeError)):
                env.step(0)
            self.assertTrue(backend.closed)
            self.assertEqual(rows, [])

    def test_reset_and_constructor_errors_close_backend(self):
        for reset in (frame((0.5, 1, 2.5)), frame(on_ground=False)):
            backend = Backend([reset])
            env = ppo.CompactParkourEnv(self.profile, backend=backend)
            with self.assertRaises(RuntimeError):
                env.reset()
            self.assertTrue(backend.closed)
        backend = Backend()
        with self.assertRaises(ValueError):
            ppo.CompactParkourEnv(self.profile, backend=backend, ticks=0)
        self.assertTrue(backend.closed)
        backend = Backend()
        too_long = tabular.Profile(
            (0.5, 1, 0.5), 0, tuple((0.5, 1, z) for z in range(7)), -1, 1, ()
        )
        with self.assertRaises(ValueError):
            ppo.CompactParkourEnv(too_long, backend=backend)
        self.assertTrue(backend.closed)

    def test_real_optimizer_timeout_bootstrap_save_reload_and_read_only_evaluation(self):
        backend = Backend()
        env = ppo.CompactParkourEnv(self.profile, backend=backend)
        self.addCleanup(env.close)
        check_env(env, warn=True)
        model = PPO("MlpPolicy", env, seed=11, **ppo.PPO_OPTIONS)
        before = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
        # SB3 must add gamma*V(terminal_observation) for timeouts, not fall/goal.
        with patch.object(
            model.policy, "predict_values", side_effect=lambda obs: torch.full((len(obs), 1), 2.0)
        ):
            model.learn(128)
        self.assertTrue(np.allclose(model.rollout_buffer.rewards, -0.01 + tabular.GAMMA * 2))
        self.assertTrue(
            any(
                not torch.equal(before[name], tensor)
                for name, tensor in model.policy.state_dict().items()
            )
        )
        self.assertTrue(any(state["step"] > 0 for state in model.policy.optimizer.state.values()))
        prefix = self.root / "compact"
        meta = ppo.metadata(self.profile, self.course)
        ppo.save_checkpoint(model, prefix, meta)
        ppo.validate_checkpoint(prefix, meta)
        restored = PPO.load(prefix.with_suffix(".zip"), env=env, device="cpu")
        for name, tensor in model.policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored.policy.state_dict()[name]))
        files = list(self.root.iterdir())
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        with (
            patch.object(ppo, "MineflayerBackend", return_value=Backend()),
            patch("sys.stdout", new=io.StringIO()),
        ):
            ppo.main(
                ["--course", str(self.course), "--output", str(prefix), "--eval", "--episodes", "2"]
            )
        self.assertEqual(set(files), set(self.root.iterdir()))
        self.assertEqual(
            hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        )

    def test_incompatible_metadata_is_rejected_before_connecting(self):
        prefix = self.root / "old"
        prefix.with_suffix(".zip").write_bytes(b"historical model")
        prefix.with_suffix(".json").write_text(json.dumps(dict(algorithm="ppo", grid_radius=3)))
        with patch.object(ppo, "MineflayerBackend") as constructor:
            with self.assertRaises(ValueError):
                ppo.main(["--course", str(self.course), "--output", str(prefix), "--eval"])
            constructor.assert_not_called()
        expected = ppo.metadata(self.profile, self.course)
        for field, value in (
            ("ticks", 3),
            ("reward_mode", "platform-bonus"),
            ("raw_high", [999] * 7),
        ):
            wrong = dict(expected, **{field: value})
            prefix.with_suffix(".json").write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):
                ppo.validate_checkpoint(prefix, expected)
            with self.assertRaises(ValueError):
                ppo.validate_checkpoint(prefix, expected, evaluation=True)
        prefix.with_suffix(".json").write_text(json.dumps(expected))
        variation = dict(expected, profile={"held_out": True}, course_sha256="different")
        self.assertEqual(ppo.validate_checkpoint(prefix, variation, evaluation=True), expected)
        with self.assertRaises(ValueError):
            ppo.validate_checkpoint(prefix, variation)


if __name__ == "__main__":
    unittest.main()
