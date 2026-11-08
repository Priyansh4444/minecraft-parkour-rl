# Minecraft Parkour RL — CS175

This CS175 project compares tabular Q-learning with random and scripted controllers on short Minecraft parkour courses. Compact PPO is an optional comparison using the same observations, actions and rewards.

## Folders

```text
parkour/
  learning/       Q-learning and compact PPO
  minecraft/      bot connection and server launcher
  courses/        fixed course definitions and reserved starts
bridge/           Mineflayer transport and physics checks
tools/
  courses/        map import and random-course generation
  experiments/    independent-seed experiment runner
docs/             setup, project, experiment and research guides
.local/           local servers, worlds, checkpoints and logs
```

Run commands from this folder. Install dependencies once:

```sh
uv sync --extra ppo --python 3.12
npm ci --prefix bridge --ignore-scripts
```

The server needs **Java 25** and Minecraft **26.1.2**. Follow [server setup](docs/setup/server.md) and [map import](docs/setup/maps.md) first.

## Headless

Start a prepared server without a game window:

```sh
uv run python -m parkour.minecraft.server --directory .local/mc2612/server --java /path/to/java25/bin/java
```

In another terminal, train Q-learning:

```sh
uv run python -m parkour.learning.q_learning --course parkour/courses/proposal-map1-single.json --output .local/runs/q/seed-11/q --episodes 300 --seed 11
```

## Watch in Minecraft

Start the same server with your Minecraft username:

```sh
uv run python -m parkour.minecraft.server --directory .local/mc2612/server --java /path/to/java25/bin/java --watch-player YourName
```

Open Minecraft Java 26.1.2 and connect to `127.0.0.1:25566`. Start the **same training command**. Your player enters spectator mode and follows `Watcher`. Closing your game window leaves training running. See [watching instructions](docs/setup/watching.md).

## Development

See [development checks](docs/development/checks.md) for formatting, linting and type checking.

- [How the code works](docs/project/guide.md)
- [Proposal scope and evaluation](docs/project/proposal-scope.md)
- [PPO comparison](docs/experiments/ppo.md)
- [Simple baseline and randomized course](docs/experiments/random-course.md)

```sh
./scripts/check.sh
```

Start with single-jump calibration, then the proposed three-jump course. Completing one fixed jump does not establish full-map navigation or generalization. After installation the training loop needs no internet; the server and computer must keep running.
