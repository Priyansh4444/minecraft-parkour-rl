# Verification

Run from the repository root:

```sh
uv run --extra ppo python -m unittest discover -s tests -v
node --test bridge/parkour-state.test.ts
```

After the folder migration, all 38 Python tests and six bridge tests passed. All three saved Q-tables and three compact PPO models loaded without file changes. Local guide links and protected map/checkpoint/proposal hashes were checked as well.

## What the checks cover

- Exact Q-learning rewards, state/action-specific table updates and timeout bootstrapping.
- Q/PPO parity for observations, rewards, goals, falls and truncations.
- Actual PPO optimizer updates, saved-model reload and evaluation without checkpoint changes.
- Course generation reproducibility, safe imports and one geometry-blind scripted rule across layouts.
- Spectator names, join order, reconnects, separation from bot controls and server shutdown.

The folder reorganization preserves the executable Q/PPO definitions and backend operations apart from import and file-location changes. Saved checkpoints keep their existing observation/action/reward schemas.

Live artifacts under `.local/watch-mode-validation/` exercise the actual Java server launcher and a successful scripted single-jump episode in each mode. Visible setup uses a second real protocol client: the server accepts spectator mode, teleport and camera attachment, while the bot's paused state and exact action ticks remain unchanged. This verifies server-side setup, not graphical rendering in a Minecraft window.

Current experiment evidence remains separate under `.local/map1-tabular-v3-calibration-sparse/`, `.local/map1-compact-ppo-calibration-sparse/` and `.local/random-course-evaluation-17501/`. Source paths inside completed manifests describe their recorded run versions and are not rewritten by a folder move.

Single-jump calibration and exploratory training successes are distinct from no-update evaluation and the final three-jump experiment. Read the [scope](../project/proposal-scope.md) and [test protocol](random-course.md) before interpreting results. Linux is the tested platform; Windows/macOS and exact human-client physics equivalence are not established.
