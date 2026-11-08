# Development

Install the Python development tools and optional PPO dependencies:

```sh
uv sync --locked --extra ppo --python 3.12
npm ci --include=dev --prefix bridge --ignore-scripts
```

Run everything from the project root:

```sh
uv run --extra ppo python -m tools.development.check
```

To format files and apply safe lint fixes first:

```sh
uv run --extra ppo python -m tools.development.check --fix
```

The command stops at the first failing check and returns a nonzero exit code. These checks do not start Minecraft or change checkpoints.

## Tools

- [Ruff](https://docs.astral.sh/ruff/) formats Python, sorts imports and checks correctness, common bugs and outdated syntax.
- [ty](https://docs.astral.sh/ty/) checks Python source, commands and tests against the installed dependencies. PPO must be installed so its library types resolve.
- [Biome](https://biomejs.dev/) formats and lints the JavaScript bridge and its configuration.
- Python unittest and Node's test runner check learning behavior, saved models, world imports, course generation, stepping and spectator setup.

Run individual checks when working on one part:

```sh
uv run ruff format --check .
uv run ruff check .
uv run --extra ppo ty check --error-on-warning
npm run check --prefix bridge
uv run --extra ppo python -m unittest discover -s tests -v
npm test --prefix bridge
```

Configuration lives in `pyproject.toml`, `bridge/biome.json` and `.editorconfig`. Dependency versions are recorded in `uv.lock` and `bridge/package-lock.json`. Generated runs, server files, installed dependencies and caches stay outside the source checks.

Keep changes small, explain decisions in ordinary comments and update tests when behavior changes. Formatting or type fixes should preserve the course settings, learning rules and checkpoint contracts.
