# Local development

Install the tools once:

```sh
uv sync --locked --extra ppo --python 3.12
npm ci --include=dev --prefix bridge --ignore-scripts
```

Run these Bash scripts from the project root:

```sh
./scripts/format.sh
./scripts/check.sh
./scripts/typecheck.sh
```

`format.sh` applies Python formatting, import sorting and safe lint fixes, then formats the TypeScript bridge. `check.sh` checks formatting and linting and calls `typecheck.sh`. `typecheck.sh` runs ty for Python and TypeScript's compiler without emitting JavaScript.

The scripts stop on a failing command. They do not launch Minecraft, change checkpoints, run automated tests or use CI.

Configuration lives in `pyproject.toml`, `bridge/biome.json`, `bridge/tsconfig.json` and `.editorconfig`. Installed versions are recorded in `uv.lock` and `bridge/package-lock.json`. Generated runs and dependencies stay outside the source checks.

Use short functions, explicit types, descriptive names, lowercase comments and blank lines between logical steps. Inspect runtime behavior through Minecraft and the logs as described in [run observations](../experiments/verification.md).
