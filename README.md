# Minecraft Parkour RL

A small course project exploring whether a Q-learning agent can learn to cross a small Minecraft parkour course. The agent reads game state and presses movement keys through the [Minescript](https://minescript.net/) mod. The course and the learner are Python files your team can read and change.

**Project status:** Early prototype. The course has three straight jumps, including a raised platform; it is not a general parkour agent. The official-launcher, single-player setup has not been tested end to end yet, including on macOS and Windows. Timing, resets, state representation, and learning still need validation. Do not treat a few trial runs as evidence of learning.

## Get started

You need Minecraft Java Edition, the official Minecraft Launcher, Python 3.12, Fabric, Fabric API and Minescript. Install compatible versions and configure them using the [complete setup guide](docs/setup.md). The guide covers macOS, Windows and Linux, including where to place the project files.

Download this repository or clone it:

```sh
git clone https://github.com/Priyansh4444/minecraft-parkour-rl.git
```

Once setup is complete, create a **new superflat single-player world with cheats enabled**, then enter these commands in Minecraft chat, one at a time:

```text
\build_course
\agent 3 1.0 1.0
```

The first command overwrites blocks around the world's origin to build the course; use a dedicated test world. The second runs three exploratory episodes. Copy the scripts into the Minecraft `minescript` folder as described in the setup guide first. A few episodes are a connection test, not evidence that the agent learned.

## Project files

- [`parkour/build_course.py`](parkour/build_course.py) creates the start, gap, catch floor and goal.
- [`parkour/course.json`](parkour/course.json) defines the start, goal, direction and fall boundary; keep it consistent with the builder.
- [`parkour/agent.py`](parkour/agent.py) observes the player, chooses actions, scores outcomes and updates learned weights.
- [`docs/setup.md`](docs/setup.md) gives installation, requirements, run instructions and troubleshooting.
- [`docs/code.md`](docs/code.md) explains the state, actions, rewards, learning update and limitations.

Logs and weights are created locally in the game's `minescript` folder. They are not shared automatically. This repository does not contain Minecraft, the mods, a world save, or an installer.
