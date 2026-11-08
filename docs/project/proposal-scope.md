# Project scope

The project trains a tabular Q-learning model on a short Minecraft parkour course, then compares it with random and simple scripted controllers. Experiments use compact numerical observations and timed actions.

## Algorithm and observations

`parkour/learning/q_learning.py` implements a literal Q-table. Each complete discretized state tuple indexes five action values: idle, walk, run, walk+jump and run+jump. The epsilon-greedy, one-step update is:

```text
Q[state][action] += 0.1 * (reward + 0.99 * max(Q[next_state]) - Q[state][action])
```

Goals and falls have no next-state bootstrap. A time-limit cutoff ends the attempt but retains the next-state bootstrap, following Lecture 07 slides 12 and 27. Only the selected state/action cell changes. There are no shared feature weights, neural policies, images or 3D terrain inputs in the main experiment.

The seven state entries are binned forward distance, lateral distance and height relative to the next configured platform, signed forward velocity, vertical velocity, ground contact and the highest platform index reached. Position bins are 0.25/0.5 blocks and velocity bins 0.05/0.1 blocks per client tick; `BINS` records the exact bounds. The dictionary stores visited states rather than allocating every possible combination. This representation is approximate, as acknowledged in the proposal.

## Course and reward

Use `proposal-map1-single.json` for initial jump calibration. `proposal-map1-three.json` describes the straight Map 1 platforms at x=19,17,13,10 along z=13, with the final landing one block higher. The target is this small three-jump course, not the entire 28-jump map. Both profiles have an 80-decision limit and reset to the configured start for every episode.

The initial rewards are exactly **-0.01 per ordinary decision, +1 for the final goal, -1 for a fall**. A timeout ends the episode with the ordinary decision reward and no added fall penalty. There is no distance shaping or repeated height bonus.

Only if sparse rewards impede learning, `--shaped` enables the proposed comparison: a 0.1 bonus once for a newly landed-on intermediate platform. Highest platform progress prevents repeated bonuses from revisiting a platform. Merely being high or passing over a platform gives nothing. The shaped condition needs a fresh table and separate results. On a single-jump task there are no intermediate platforms, so shaping has no effect; use the multi-jump course for that comparison.

No hunger, potions, neos or headhitters are part of the initial experiment. Start-platform resets are the default; individual-jump calibration is labeled separately from complete-course evaluation.

## Experiments

With the local Map 1 server running, one small training run is:

```sh
uv run python -m parkour.learning.q_learning --course parkour/courses/proposal-map1-single.json --output .local/tabular/seed-11/q --episodes 300 --seed 11
```

For the submitted evaluation plan, the thin experiment runner creates separate tables for seeds 11, 22 and 33, then runs 50 no-exploration evaluations and 50 matched random/scripted comparison episodes per seed:

```sh
uv run python -m tools.experiments.run_tabular_experiment --course parkour/courses/proposal-map1-single.json --output .local/map1-tabular-calibration-sparse --train-episodes 300 --eval-episodes 50
```

This first run is **single-jump calibration**. Move to the three-jump task after checking the interface and learning results. Earlier scripted results used a preset calibrated on the training layout. New experiment-runner comparisons use `simple-v1`: always forward+sprint+jump, with no observations or platform-specific timing. The runner creates separate baseline profiles without changing the source course. The low-level `--scripted` interface still replays its supplied profile's sequence. See [simple baseline and randomized tests](../experiments/random-course.md) for the frozen evaluation-only three-jump course.

Starting offsets `proposal-map1-heldout-start-a.json` and `proposal-map1-heldout-start-b.json` are predeclared reserves. Do not put them in training or tune on them. `--held-out PATHS...` adds evaluation and comparison-controller runs on reserved cases without learning or changing checkpoints. For the final multi-jump experiment, likewise predeclare its own layout/start variations before tuning. Once results inform tuning, those cases are validation data rather than untouched tests.

Every transition records state, action, reward, next state and raw player observations. Episode summaries record completion/fall/timeout, maximum platform, progress, decisions, elapsed client ticks and wall time. Run manifests preserve seeds, course definitions and source hashes. Each trained seed remains a separate result; repeated episodes are not independent training runs. Report failures as well as successes.

## Environment and code

The existing local Java server and Mineflayer transport provide observations and movement. The bridge applies fixed-duration actions and pauses client movement between decisions; it does not freeze server time or guarantee vanilla-exact player physics. The live reset/tick loop must be checked on the pinned server version. See [headless setup](../setup/server.md).

The primary learner needs only Python's standard library; the bot uses pinned Node/Mineflayer dependencies. Historical neural-library dependencies are excluded from the normal install. Earlier experimental code and checkpoints are kept separate for provenance and are not used by the submitted-proposal experiment.

PPO is an optional secondary comparison. PPO is covered in Lecture 13, but any comparison must use the same compact state, five actions, rewards and course so that it tests the learning method rather than a larger task. It remains secondary to the tabular experiment. See [the course mapping](course-alignment.md).
