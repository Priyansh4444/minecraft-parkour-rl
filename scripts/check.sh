#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

uv run --locked --extra ppo ruff format --check .
uv run --locked --extra ppo ruff check .
npm run check --prefix bridge

./scripts/typecheck.sh
