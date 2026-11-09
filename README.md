# Minecraft Parkour RL

Train and compare Q-learning, PPO, random actions and a simple scripted controller on short, straight Minecraft courses.

- [Setup](docs/setup.md)
- [Code](docs/code.md)
- [Training and evaluation](docs/running.md)

## Folders

```text
parkour/
  learning/       Q-learning and PPO
  minecraft/      server launcher and bot connection
  courses/        course definitions
bridge/           TypeScript controls and physics measurements
tools/
  courses/        map import and course generation
  experiments/    training and comparison runners
scripts/          formatting and type checking
docs/             setup, code and commands
.local/           local worlds, models and logs
```

Run commands from this folder. After [setup](docs/setup.md), start the server:

```sh
uv run python -m parkour.minecraft.server --directory .local/mc2612/server --java /path/to/java25/bin/java
```

In another terminal, start learning:

```sh
uv run python -m parkour.learning.q_learning --course parkour/courses/proposal-map1-single.json --output .local/runs/q --episodes 300
```
