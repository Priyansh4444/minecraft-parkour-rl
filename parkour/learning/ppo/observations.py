"""Optional numerical context for PPO on straight, full-block courses."""

import numpy as np
from numpy.typing import NDArray

from ...minecraft.connection import PhysicsFrame
from ..q_learning import DIRECTIONS, Profile

CONTEXT_FIELDS = [
    "target_forward",
    "target_lateral",
    "target_height",
    "forward_velocity",
    "vertical_velocity",
    "grounded",
    "highest_platform",
    "current_platform_forward",
    "current_platform_height",
    "lateral_velocity",
    "jump_cooldown",
    "following_gap",
    "following_rise",
    "target_is_goal",
    "time_remaining",
]


def context_observation(
    profile: Profile, frame: PhysicsFrame, highest: int, steps: int
) -> NDArray[np.float32]:
    x, y, z = frame["position"]
    velocity_x, velocity_y, velocity_z = frame["velocity"]
    cooldown = frame.get("jump_ticks")
    if cooldown is None:
        raise ValueError("Context observations require the bot's jump cooldown")
    forward_x, forward_z = DIRECTIONS[int(profile.yaw) % 360]

    target_index = min(highest + 1, len(profile.platforms) - 1)
    target = profile.platforms[target_index]
    current = profile.platforms[highest]
    following = profile.platforms[min(target_index + 1, len(profile.platforms) - 1)]

    forward = (target[0] - x) * forward_x + (target[2] - z) * forward_z
    lateral = -(target[0] - x) * forward_z + (target[2] - z) * forward_x

    # scale distances, velocities and flags into the policy's [-1, 1] range
    values = {
        "target_forward": (forward + 2) / 5 - 1,
        "target_lateral": lateral / 2,
        "target_height": (target[1] - y) / 4,
        "forward_velocity": (velocity_x * forward_x + velocity_z * forward_z) / 0.4,
        "vertical_velocity": (velocity_y + 0.3) / 0.9,
        "grounded": 2 * int(frame["on_ground"]) - 1,
        "highest_platform": 2 * highest / 5 - 1,
        "current_platform_forward": ((current[0] - x) * forward_x + (current[2] - z) * forward_z)
        / 4,
        "current_platform_height": (current[1] - y) / 4,
        "lateral_velocity": (-velocity_x * forward_z + velocity_z * forward_x) / 0.4,
        "jump_cooldown": 2 * cooldown / 10 - 1,
        "following_gap": (
            (following[0] - target[0]) * forward_x + (following[2] - target[2]) * forward_z
        )
        / 4,
        "following_rise": (following[1] - target[1]) / 2,
        "target_is_goal": 1 if target_index == len(profile.platforms) - 1 else -1,
        "time_remaining": 2 * (profile.max_steps - steps) / profile.max_steps - 1,
    }
    observation = np.asarray([values[name] for name in CONTEXT_FIELDS], dtype=np.float32)
    if not np.isfinite(observation).all():
        raise ValueError("Context observations must contain finite numbers")
    # bound the network inputs; this does not limit the bot's movement or rewards
    return np.clip(observation, -1, 1)
