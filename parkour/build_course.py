"""Build the starter course in a NEW superflat world with cheats enabled.

In Minecraft chat, run ``\\build_course``. This changes blocks around the
world origin; do not run it in a world you want to keep unchanged.
"""

import time

import minescript

# These coordinates match course.json. Feet stand at y=-60, on blocks at -61.
COURSE_COMMANDS = (
    "fill -15 -64 -4 4 -64 4 minecraft:bedrock",  # catch falls
    "fill 4 -60 -1 -10 -56 1 minecraft:air",       # clear space above the course
    "fill 3 -61 0 -3 -61 0 minecraft:grass_block", # starting platform
    "fill -6 -61 0 -7 -61 0 minecraft:grass_block", # landing platform
    "setblock -8 -61 0 minecraft:gold_block",       # goal
    "fill -4 -63 0 -5 -61 0 minecraft:air",        # two-block gap
    "setworldspawn 0 -60 0 90 0",
    "gamerule immediate_respawn true",
    "gamerule advance_time false",
    "gamerule advance_weather false",
    "time set noon",
    "weather clear",
)


def main() -> None:
    # Going to the course loads its chunks before the fill commands are sent.
    minescript.execute("tp @s 0.2 -60 0.5 90 0")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        x, y, z = minescript.player_position()
        if abs(x - 0.2) < 1 and abs(y + 60) < 1 and abs(z - 0.5) < 1:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Teleport failed. Use a new superflat world with cheats enabled.")

    for command in COURSE_COMMANDS:
        minescript.execute(command)
        time.sleep(0.1)
    minescript.echo("Course commands sent. Check the two-block gap and gold goal before training.")


if __name__ == "__main__":
    main()
