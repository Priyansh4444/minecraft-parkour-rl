"""A straight-line parkour prototype that runs inside Minecraft via Minescript.

The observation scans ahead toward the next gap:

  lip     distance to the edge the agent must jump from
  gap     width of the missing span
  height  landing surface relative to the feet
  speed   horizontal blocks per tick
  vy      vertical velocity
  ground  standing on a block

Actions are short key presses. In Minecraft chat, ``\\agent 40 0.5 0.05``
runs 40 episodes with epsilon decreasing from 0.5 to 0.05.
Edit ``course.json`` for a different start, goal, or fall height.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import minescript

HERE = Path(__file__).resolve().parent
COURSE_FILE = HERE / "course.json"
Q_FILE = HERE / "q-table.json"
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

DEFAULT_COURSE = {
    "start": [0.2, -60.0, 0.5],
    "yaw": 90.0,
    "goal": [-8.5, -60.0, 0.5],
    "fall_y": -61.2,
    "goal_radius": 1.2,
}

NON_SOLID = (
    "air",
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
    if COURSE_FILE.exists():
        return {**DEFAULT_COURSE, **json.loads(COURSE_FILE.read_text())}
    COURSE_FILE.write_text(json.dumps(DEFAULT_COURSE, indent=2) + "\n")
    return dict(DEFAULT_COURSE)


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
    return bool(block) and not any(word in block for word in NON_SOLID)


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


def scan(x: float, y: float, z: float) -> tuple[int, int, float | None]:
    """Return (lip bin, gap bin, landing height) looking along the course yaw."""
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
        return 6, 0, 0.0

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
    return bin_distance(distance), min(9, gap), height


def observe(speed: float, player=None) -> tuple[str, dict]:
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
    if height is None or abs(height) < 0.5:
        height_bin = 1
    elif height < 0:
        height_bin = 0
    else:
        height_bin = 2
    vy_bin = 0 if vy > 0.1 else 1 if vy > -0.4 else 2
    state = f"{lip},{gap},{height_bin},{bin_speed(speed)},{vy_bin},{int(grounded)}"
    return state, {
        "x": x,
        "y": y,
        "z": z,
        "lip": lip,
        "gap": gap,
        "height": height,
        "speed": speed,
        "vy": vy,
        "ground": grounded,
    }


def choose(q: dict[str, list[float]], state: str, epsilon: float) -> int:
    if random.random() < epsilon or state not in q:
        return random.randrange(len(ACTIONS))
    row = q[state]
    best = max(row)
    return random.choice([i for i, value in enumerate(row) if value == best])


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
    crossed = False
    for step in range(MAX_STEPS):
        action = choose(q, state, epsilon)
        name = ACTIONS[action]
        x0, _, z0 = player.position
        do_action(name)
        try:
            speed = wait_ticks(TICKS)
        finally:
            force_release()
        player = minescript.player()
        x1, y1, z1 = player.position

        progress = (x1 - x0) * DX + (z1 - z0) * DZ
        reward = 2.0 * progress - STEP_PENALTY
        terminal = False
        if y1 < FALL_Y:
            outcome = "fall"
            reward = FALL_REWARD
            terminal = True

        next_state, next_info = observe(speed, player)
        if not terminal and next_info["ground"] and at_goal(x1, y1, z1):
            outcome = "goal"
            reward = GOAL_REWARD
            terminal = True
        landed = not crossed and info["gap"] > 0 and next_info["gap"] == 0 and next_info["ground"]
        if landed:
            reward += 3.0
            crossed = True

        if learn:
            q.setdefault(state, [0.0] * len(ACTIONS))
            # A terminal state has no future value to add to its reward.
            if terminal:
                target = reward
            else:
                next_values = q.setdefault(next_state, [0.0] * len(ACTIONS))
                target = reward + GAMMA * max(next_values)
            q[state][action] += ALPHA * (target - q[state][action])
        log(
            episode=number,
            step=step,
            state=state,
            action=name,
            next=next_state,
            reward=round(reward, 3),
            landed=landed,
            x=round(x1, 3),
            y=round(y1, 3),
            z=round(z1, 3),
            lip=next_info["lip"],
            gap=next_info["gap"],
            height=None if next_info["height"] is None else round(next_info["height"], 2),
            speed=round(speed, 3),
            vy=round(next_info["vy"], 3),
            ground=next_info["ground"],
        )
        if terminal:
            break
        state, info = next_state, next_info
    force_release()
    log(episode=number, outcome=outcome, epsilon=round(epsilon, 3), states=len(q), learn=learn)
    return outcome


def run_batch(
    episodes: int, epsilon_start: float, epsilon_end: float, learn: bool = True
) -> dict[str, int]:
    q = load_q()
    LOG_FILE.touch()
    log(event="start", episodes=episodes, epsilon=epsilon_start, end=epsilon_end, states=len(q), learn=learn)
    outcomes = {"goal": 0, "fall": 0, "timeout": 0}
    for number in range(1, episodes + 1):
        span = max(1, episodes - 1)
        epsilon = 0.0 if not learn else max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (number - 1) / span)
        outcome = episode(q, number, epsilon, learn)
        outcomes[outcome] += 1
        if learn:
            save_q(q)
        log(event="episode", episode=number, outcome=outcome, epsilon=round(epsilon, 3), **outcomes, states=len(q), learn=learn)
        if number % 5 == 0 or number == episodes or outcome == "goal":
            minescript.echo(
                f"{number}/{episodes} {outcome} | goals {outcomes['goal']} "
                f"falls {outcomes['fall']} states {len(q)}"
            )
    force_release()
    return outcomes


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    epsilon_start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    epsilon_end = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    try:
        outcomes = run_batch(episodes, epsilon_start, epsilon_end)
        minescript.echo(f"agent done: {outcomes} | q-table {Q_FILE.name}")
    finally:
        force_release()


if __name__ == "__main__":
    main()
