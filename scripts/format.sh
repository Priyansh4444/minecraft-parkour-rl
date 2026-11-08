#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

uv run --locked --extra ppo ruff check . --fix
uv run --locked --extra ppo ruff format .
npm run format --prefix bridge
