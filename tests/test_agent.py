"""Fast checks for the learning math without launching Minecraft."""
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

fake = types.SimpleNamespace(
    player_press_forward=lambda value: None,
    player_press_sprint=lambda value: None,
    player_press_jump=lambda value: None,
)
sys.modules['minescript'] = fake
path = Path(__file__).resolve().parents[1] / 'parkour' / 'agent.py'
spec = importlib.util.spec_from_file_location('agent', path)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class AgentTests(unittest.TestCase):
    def test_air_and_stairs_are_distinct(self):
        self.assertFalse(agent.solid('minecraft:air'))
        self.assertFalse(agent.solid('minecraft:cave_air'))
        self.assertFalse(agent.solid('minecraft:void_air'))
        self.assertTrue(agent.solid('minecraft:stone_stairs'))

    def test_shared_features_transfer_and_different_action_stays_zero(self):
        weights = {}
        first = ('bias', 'speed:1', 'lip:2')
        similar = ('bias', 'speed:1', 'lip:3')
        agent.update_weights(weights, first, 2, 4.0)
        self.assertGreater(agent.values(weights, similar)[2], 0)
        self.assertEqual(agent.values(weights, similar)[1], 0)

    def test_goal_reward_and_fall_do_not_bootstrap_potential(self):
        self.assertEqual(agent.reward_for_step(5, 0, 'goal'), 10.5)
        self.assertEqual(agent.reward_for_step(5, 30, 'fall'), -9.5)

    def test_distance_shaping_uses_discount(self):
        expected = -agent.STEP_PENALTY + agent.GAMMA * agent.potential(4) - agent.potential(5)
        self.assertAlmostEqual(agent.reward_for_step(5, 4, None), expected)

    def test_observation_uses_velocity_orientation_and_goal(self):
        player = types.SimpleNamespace(position=(0.2, -60.0, 0.5), velocity=(-0.15, -0.078, 0.0), yaw=90.0)
        with patch.object(fake, 'getblock', return_value='minecraft:stone', create=True), patch.object(agent, 'scan', return_value=(1.0, 2, None)):
            features, info = agent.observe(0.12, player)
        self.assertIn('height:unknown', features)
        self.assertIn('ground:1', features)
        self.assertIn('yaw:0', features)
        self.assertGreater(info['forward_velocity'], 0)
        self.assertGreater(info['goal_distance'], 20)

    def test_random_baseline_never_learns(self):
        with patch.object(sys, 'argv', ['agent.py', '3', '--random', '--seed', '7']), \
             patch.object(agent, 'run_batch', return_value={'goal': 0}) as batch, \
             patch.object(agent, 'log'), patch.object(fake, 'echo', create=True):
            agent.main()
        batch.assert_called_once_with(3, 1.0, 0.05, learn=False)

    def test_eval_requires_weights(self):
        with patch.object(sys, 'argv', ['agent.py', '3', '--eval']), \
             patch.object(agent, 'Q_FILE', Path('/nonexistent-cs175/linear-q.json')):
            with self.assertRaises(FileNotFoundError):
                agent.main()

    def test_saved_course_matches_goal(self):
        self.assertEqual(agent.GOAL, [-20.5, -59.0, 0.5])


if __name__ == '__main__':
    unittest.main()
