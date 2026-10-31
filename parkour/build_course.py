"""Build the starter course in a NEW superflat world with cheats enabled.

In Minecraft chat, run ``\\build_course``. This changes blocks around the
world origin; do not run it in a world you want to keep unchanged.
"""

import time

import minescript

# Each platform is three blocks wide. The second jump rises one block.
# X ranges: start 3..-3, first landing -6..-9, raised -11..-14,
# goal platform -17..-21. The gaps are 2, 1, and 2 blocks wide.
COURSE_COMMANDS = (
    "fill -24 -64 -3 4 -64 3 minecraft:bedrock",  # catch falls
    "fill -23 -63 -2 4 -55 2 minecraft:air",       # clear the course volume
    "fill -3 -61 -1 3 -61 1 minecraft:grass_block",   # start, feet y=-60
    "fill -9 -61 -1 -6 -61 1 minecraft:stone",        # same height
    "fill -14 -60 -1 -11 -60 1 minecraft:stone",     # one block higher
    "fill -21 -60 -1 -17 -60 1 minecraft:stone",     # final landing
    "setblock -21 -60 0 minecraft:gold_block",       # goal
    "setworldspawn 0 -60 0 90 0",
    "gamerule immediate_respawn true",
    "gamerule advance_time false",
    "gamerule advance_weather false",
    "time set noon",
    "weather clear",
    "tp @s 0.2 -60 0.5 90 0",
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

    # The end is across a chunk boundary; force-load the small build area
    # before placing blocks, then let the world manage it normally again.
    minescript.execute("forceload add -32 -16 15 15")
    time.sleep(3)
    for command in COURSE_COMMANDS:
        minescript.execute(command)
        time.sleep(0.1)
    minescript.execute("forceload remove -32 -16 15 15")
    minescript.echo("Course commands sent. Check all three gaps and the gold goal before training.")


if __name__ == "__main__":
    main()
