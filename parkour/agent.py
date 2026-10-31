"""A straight-line parkour prototype that runs inside Minecraft via Minescript.

The observation scans ahead toward the next gap:

  lip     distance to the edge the agent must jump from
  gap     width of the missing span
  height  landing surface relative to the feet
  speed   measured horizontal displacement per tick
  vy      vertical velocity
  ground  standing on a block

Actions are short key presses. In Minecraft chat, ``\\agent 40 0.5 0.05``
runs 40 episodes with epsilon decreasing from 0.5 to 0.05.
Edit ``course.json`` for a different start, goal, or fall height.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import minescript

HERE = Path(__file__).resolve().parent
COURSE_FILE = HERE / "course.json"
Q_FILE = HERE / "linear-q.json"
LOG_FILE = HERE / "agent-log.jsonl"

ACTIONS = ("idle", "walk", "run", "walk_jump", "run_jump")
KEYS = {
    "idle": (False, False, False),
    "walk": (True, False, False),
    "run": (True, True, False),
    "walk_jump": (True, False, True),
    "run_jump": (True, True, True),
}
TICKS = 2
ALPHA = 0.25
GAMMA = 0.95
STEP_PENALTY = 0.01
GOAL_REWARD = 10.0
FALL_REWARD = -10.0
MAX_STEPS = 80
LOOKAHEAD = 8

NON_SOLID = (
    "air",
    "cave_air",
    "void_air",
    "water",
    "lava",
    "short_grass",
    "tall_grass",
    "fern",
    "flower",
    "torch",
    "sapling",
    "vine",
    "snow_layer",
    "carpet",
    "button",
    "pressure_plate",
    "rail",
)

HELD = {"forward": False, "sprint": False, "jump": False}
PRESS = {
    "forward": minescript.player_press_forward,
    "sprint": minescript.player_press_sprint,
    "jump": minescript.player_press_jump,
}


def load_course() -> dict:
    if not COURSE_FILE.exists():
        raise FileNotFoundError("Copy course.json into the minescript folder with agent.py")
    return json.loads(COURSE_FILE.read_text())


COURSE = load_course()
START = COURSE["start"]
YAW = float(COURSE["yaw"])
GOAL = COURSE["goal"]
FALL_Y = float(COURSE["fall_y"])
GOAL_RADIUS = float(COURSE["goal_radius"])

ANGLE = math.radians(YAW)
DX, DZ = -math.sin(ANGLE), math.cos(ANGLE)


def log(**fields: object) -> None:
    with LOG_FILE.open("a") as handle:
        handle.write(json.dumps(fields) + "\n")


def load_q() -> dict[str, list[float]]:
    if Q_FILE.exists():
        return json.loads(Q_FILE.read_text())
    return {}


def save_q(q: dict[str, list[float]]) -> None:
    Q_FILE.write_text(json.dumps(q, separators=(",", ":")) + "\n")


def force_release() -> None:
    for name, press in PRESS.items():
        press(False)
        HELD[name] = False


def set_keys(forward: bool, sprint: bool, jump: bool) -> None:
    for name, value in (("forward", forward), ("sprint", sprint), ("jump", jump)):
        if HELD[name] != value:
            PRESS[name](value)
            HELD[name] = value


def do_action(name: str) -> None:
    set_keys(*KEYS[name])


def solid(block: str) -> bool:
    """Treat only explicitly listed block IDs as non-solid (stairs are solid)."""
    return bool(block) and block.split(":")[-1].split("[")[0] not in NON_SOLID


def bin_distance(distance: float | None) -> int:
    """0 past the lip, then 0.5, 1, 2, 3.5, farther, no gap."""
    if distance is None:
        return 6
    if distance < 0:
        return 0
    if distance < 0.5:
        return 1
    if distance < 1.0:
        return 2
    if distance < 2.0:
        return 3
    if distance < 3.5:
        return 4
    return 5


def bin_speed(speed: float) -> int:
    if speed < 0.03:
        return 0
    if speed < 0.15:
        return 1
    if speed < 0.24:
        return 2
    return 3


def scan(x: float, y: float, z: float) -> tuple[float | None, int, float | None]:
    """Return distance to gap, gap width and estimated landing height."""
    base = math.floor(y - 0.01)
    columns: list[tuple[int, int]] = []
    positions: list[list[int]] = []
    for distance in range(0, LOOKAHEAD + 1):
        px = math.floor(x + DX * distance)
        pz = math.floor(z + DZ * distance)
        columns.append((px, pz))
        for dy in (1, 0, -1, -2, -3):
            positions.append([px, base + dy, pz])
    blocks = minescript.get_block_list(positions)

    surfaces: list[float | None] = []
    for index in range(len(columns)):
        surface = None
        for offset in range(5):
            # Ignore the safety floor below the fall boundary: it catches the
            # player but is not a traversable landing platform.
            block_y = base + 1 - offset
            if block_y > FALL_Y - 2 and solid(blocks[index * 5 + offset]):
                surface = base + 2 - offset
                break
        surfaces.append(surface)

    first_open = next((i for i, surface in enumerate(surfaces) if surface is None), None)
    if first_open is None:
        return None, 0, None

    px, pz = columns[first_open]
    if abs(DX) >= abs(DZ):
        lip = px + 1 if DX < 0 else px
        distance = (x - lip) if DX < 0 else (lip - x)
    else:
        lip = pz + 1 if DZ < 0 else pz
        distance = (z - lip) if DZ < 0 else (lip - z)

    gap = 0
    while first_open + gap < len(surfaces) and surfaces[first_open + gap] is None:
        gap += 1
    landing = first_open + gap
    height = None if landing >= len(surfaces) else surfaces[landing] - y
    return distance, min(9, gap), height


def observe(speed: float, player=None) -> tuple[tuple[str, ...], dict]:
    player = player or minescript.player()
    x, y, z = player.position
    vy = player.velocity[1]
    base = math.floor(y - 0.01)
    support = solid(minescript.getblock(math.floor(x), base, math.floor(z)))
    if not support:
        # The 0.6-wide box still touches a block just behind the feet.
        support = solid(
            minescript.getblock(math.floor(x - DX * 0.3), base, math.floor(z - DZ * 0.3))
        )
    snapped = abs(y * 2 - round(y * 2)) < 0.06
    grounded = support and snapped and vy <= 0.05

    lip, gap, height = scan(x, y, z)
    vx, vy, vz = player.velocity
    forward_velocity = vx * DX + vz * DZ
    lateral_velocity = vx * (-DZ) + vz * DX
    distance_to_goal = math.dist((x, y, z), GOAL)

    # These sparse, shared features let similar states share learned weights.
    # Paired features allow the jump decision to depend on both edge and speed.
    lip_bin = bin_distance(lip)
    speed_bin = bin_speed(speed) if forward_velocity >= 0 else -1
    height_bin = "unknown" if height is None else "down" if height < -0.5 else "up" if height > 0.5 else "level"
    vertical_bin = "up" if vy > 0.1 else "down" if vy < -0.4 else "level"
    goal_bin = min(8, int(distance_to_goal / 4))
    side_bin = -1 if lateral_velocity < -0.03 else 1 if lateral_velocity > 0.03 else 0
    lateral_offset = (x - START[0]) * (-DZ) + (z - START[2]) * DX
    offset_bin = -1 if lateral_offset < -0.35 else 1 if lateral_offset > 0.35 else 0
    yaw_error = (player.yaw - YAW + 180) % 360 - 180
    yaw_bin = -1 if yaw_error < -10 else 1 if yaw_error > 10 else 0
    state = (
        "bias", f"lip:{lip_bin}", f"gap:{gap}", f"height:{height_bin}",
        f"speed:{speed_bin}", f"vertical:{vertical_bin}",
        f"side:{side_bin}", f"offset:{offset_bin}", f"yaw:{yaw_bin}",
        f"ground:{int(grounded)}", f"goal:{goal_bin}",
        f"takeoff:{lip_bin},{speed_bin},{int(grounded)}",
        f"jump:{gap},{height_bin},{speed_bin}",
    )
    return state, {
        "x": x, "y": y, "z": z, "lip": lip, "gap": gap,
        "height": height, "speed": speed, "vx": vx, "vy": vy, "vz": vz,
        "forward_velocity": forward_velocity, "lateral_velocity": lateral_velocity,
        "goal_distance": distance_to_goal, "yaw_error": yaw_error,
        "lateral_offset": lateral_offset, "ground": grounded,
    }


def values(q: dict[str, list[float]], state: tuple[str, ...]) -> list[float]:
    """Linear value approximation: sum weights of the active features."""
    result = [0.0] * len(ACTIONS)
    for feature in state:
        weights = q.get(feature)
        if weights is not None:
            for action, weight in enumerate(weights):
                result[action] += weight
    return result


def choose(q: dict[str, list[float]], state: tuple[str, ...], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    row = values(q, state)
    best = max(row)
    return random.choice([i for i, value in enumerate(row) if value == best])


def potential(distance: float) -> float:
    # Bounded distance shaping; all true terminal states have potential zero.
    return -0.1 * min(distance, 40.0)


def reward_for_step(old_distance: float, new_distance: float, outcome: str | None) -> float:
    """Goal-focused reward with discounted potential-based shaping."""
    base = GOAL_REWARD if outcome == "goal" else FALL_REWARD if outcome == "fall" else -STEP_PENALTY
    next_potential = 0.0 if outcome is not None else potential(new_distance)
    return base + GAMMA * next_potential - potential(old_distance)


def update_weights(q: dict[str, list[float]], state: tuple[str, ...], action: int, target: float) -> None:
    """Move the chosen action value toward the target using active features."""
    current = values(q, state)[action]
    adjustment = ALPHA * (target - current) / len(state)
    for feature in state:
        q.setdefault(feature, [0.0] * len(ACTIONS))[action] += adjustment


def wait_alive(timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if minescript.player_health() > 0:
                return True
        except Exception:
            pass
        time.sleep(0.05)
    return False


def reset() -> None:
    force_release()
    if not wait_alive(5.0):
        raise RuntimeError(
            "player is dead and did not respawn; set /gamerule immediate_respawn true"
        )
    for _ in range(2):
        minescript.execute(f"tp @s {START[0]} {START[1]} {START[2]} {YAW} 0")
        deadline = time.time() + 2.0
        stable = 0
        while time.time() < deadline:
            time.sleep(0.05)
            if minescript.player_health() <= 0:
                if not wait_alive(5.0):
                    raise RuntimeError("player died during reset")
                break
            player = minescript.player()
            x, y, z = player.position
            vx, _, vz = player.velocity
            if abs(x - START[0]) < 0.2 and abs(y - START[1]) < 0.2 and abs(z - START[2]) < 0.2 and math.hypot(vx, vz) < 0.03:
                stable += 1
            else:
                stable = 0
            if stable >= 2:
                minescript.player_set_orientation(YAW, 0.0)
                time.sleep(0.05)
                return
    raise RuntimeError("could not teleport to the course start; is the player an operator?")


def at_goal(x: float, y: float, z: float) -> bool:
    return math.hypot(x - GOAL[0], z - GOAL[2]) <= GOAL_RADIUS and abs(y - GOAL[1]) < 1.5


def wait_ticks(ticks: int) -> float:
    """Wait `ticks` game ticks and return the last measured blocks per tick."""
    tick = minescript.world_info().game_ticks
    pos = minescript.player_position()
    start = tick
    last_tick, last_pos = tick, pos
    speed = 0.0
    deadline = time.monotonic() + 5.0
    while tick - start < ticks:
        if time.monotonic() > deadline:
            force_release()
            raise RuntimeError("game ticks stopped; check the Minecraft window and server")
        tick = minescript.world_info().game_ticks
        pos = minescript.player_position()
        if tick > last_tick:
            speed = math.hypot(pos[0] - last_pos[0], pos[2] - last_pos[2]) / (tick - last_tick)
            last_tick, last_pos = tick, pos
        if pos[1] < FALL_Y:
            break
        time.sleep(0.01)
    return speed


def episode(q: dict[str, list[float]], number: int, epsilon: float, learn: bool = True) -> str:
    reset()
    player = minescript.player()
    state, info = observe(0.0, player)
    outcome = "timeout"
    started = time.monotonic()
    for step in range(MAX_STEPS):
        action = choose(q, state, epsilon)
        name = ACTIONS[action]
        do_action(name)
        try:
            speed = wait_ticks(TICKS)
        finally:
            set_keys(False, False, False)
        player = minescript.player()
        x1, y1, z1 = player.position

        terminal_outcome = None
        if y1 < FALL_Y:
            outcome = terminal_outcome = "fall"
        next_state, next_info = observe(speed, player)
        if terminal_outcome is None and next_info["ground"] and at_goal(x1, y1, z1):
            outcome = terminal_outcome = "goal"
        terminal = terminal_outcome is not None
        reward = reward_for_step(info["goal_distance"], next_info["goal_distance"], terminal_outcome)

        if learn:
            target = reward if terminal else reward + GAMMA * max(values(q, next_state))
            update_weights(q, state, action, target)
        log(
            episode=number,
            step=step,
            state=list(state),
            action=name,
            next=list(next_state),
            reward=round(reward, 3),
            x=round(x1, 3),
            y=round(y1, 3),
            z=round(z1, 3),
            lip=None if next_info["lip"] is None else round(next_info["lip"], 3),
            gap=next_info["gap"],
            height=None if next_info["height"] is None else round(next_info["height"], 2),
            speed=round(speed, 3),
            vx=round(next_info["vx"], 3),
            vy=round(next_info["vy"], 3),
            vz=round(next_info["vz"], 3),
            goal_distance=round(next_info["goal_distance"], 3),
            yaw_error=round(next_info["yaw_error"], 3),
            lateral_offset=round(next_info["lateral_offset"], 3),
            base_reward=GOAL_REWARD if terminal_outcome == "goal" else FALL_REWARD if terminal_outcome == "fall" else -STEP_PENALTY,
            ground=next_info["ground"],
        )
        if terminal:
            break
        state, info = next_state, next_info
    force_release()
    log(episode=number, outcome=outcome, epsilon=round(epsilon, 3), features=len(q), learn=learn, steps=step + 1, seconds=round(time.monotonic() - started, 3))
    return outcome


def run_batch(
    episodes: int, epsilon_start: float, epsilon_end: float, learn: bool = True
) -> dict[str, int]:
    q = load_q()
    LOG_FILE.touch()
    log(event="start", episodes=episodes, epsilon=epsilon_start, end=epsilon_end, features=len(q), learn=learn)
    outcomes = {"goal": 0, "fall": 0, "timeout": 0}
    for number in range(1, episodes + 1):
        span = max(1, episodes - 1)
        epsilon = epsilon_start if not learn else epsilon_start + (epsilon_end - epsilon_start) * (number - 1) / span
        outcome = episode(q, number, epsilon, learn)
        outcomes[outcome] += 1
        if learn:
            save_q(q)
        log(event="episode", episode=number, outcome=outcome, epsilon=round(epsilon, 3), **outcomes, features=len(q), learn=learn)
        if number % 5 == 0 or number == episodes or outcome == "goal":
            minescript.echo(
                f"{number}/{episodes} {outcome} | goals {outcomes['goal']} "
                f"falls {outcomes['fall']} features {len(q)}"
            )
    force_release()
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate a parkour agent")
    parser.add_argument("episodes", nargs="?", type=int, default=3)
    parser.add_argument("epsilon_start", nargs="?", type=float, default=0.5)
    parser.add_argument("epsilon_end", nargs="?", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", action="store_true", help="evaluate saved weights without learning")
    mode.add_argument("--random", action="store_true", help="evaluate random actions without learning")
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.epsilon_start <= 1 or not 0 <= args.epsilon_end <= 1:
        parser.error("episodes must be positive and epsilon values must be between 0 and 1")
    if args.eval and not Q_FILE.exists():
        raise FileNotFoundError("No saved weights found; train before using --eval")
    random.seed(args.seed)
    learning = not (args.eval or args.random)
    epsilon_start = 0.0 if args.eval else 1.0 if args.random else args.epsilon_start
    log(event="run_config", seed=args.seed, mode="train" if learning else "random" if args.random else "eval")
    try:
        outcomes = run_batch(args.episodes, epsilon_start, args.epsilon_end, learn=learning)
        minescript.echo(f"agent done: {outcomes} | weights {Q_FILE.name}")
    finally:
        force_release()


if __name__ == "__main__":
    main()
