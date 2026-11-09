# Code

An **episode** is one attempt at a course. A **decision** chooses an action, normally held for two physics ticks. The **policy** is the learner's rule for choosing actions.

## Execution flow

1. `parkour/minecraft/connection.py` sends a JSON request to `bridge/headless.ts`.
2. `bridge/protocol.ts` validates the request. `Action` describes its five button values; `PhysicsFrame` describes the measured position, velocity, ground contact and tick counters.
3. `bridge/parkour-state.ts` applies the buttons for the requested ticks, pauses bot movement and returns measurements. The server continues running between requests.
4. `parkour/learning/contacts.py` detects platform contacts and episode endings. `q_learning.py` builds the state and calculates reward; PPO uses the same task helpers through `ppo/environment.py`.
5. Training updates the Q-table or PPO policy from the observation, action, reward and next observation. Evaluation chooses actions without updating the model.

Both learners choose idle, walk, run, walking jump or running jump along the course's fixed heading.

## Q-learning

A state has seven entries: binned forward distance, sideways distance, target height, forward speed, vertical speed, grounded status and highest platform reached. `BINS` defines the bin widths and limits. Each state indexes five action values in a dictionary.

`choose()` selects a random action with probability epsilon, the exploration setting. Otherwise it selects a highest-valued action. `update_q()` applies:

```text
Q[state][action] += 0.1 * (reward + 0.99 * future_value - Q[state][action])
```

`future_value` is the best next-state action value, or zero after a goal or fall. A timeout ends the attempt but keeps that next-state estimate.

## PPO

PPO (Proximal Policy Optimization) learns a neural network that chooses actions. Stable-Baselines3 is the Python library that implements its training algorithm; this project supplies the Minecraft observations, controls and rewards.

`ppo/__main__.py` starts training or evaluation. `environment.py` runs episodes, `checkpoints.py` saves the model and settings, and `routes.py` selects training routes.

The default input is the same seven bins, scaled to -1 through 1. `--observations context` instead uses these fifteen numerical inputs from `observations.py`, in order:

| Input | Measurement before scaling | Purpose |
| --- | --- | --- |
| `target_forward` | Distance ahead, blocks | Jump timing |
| `target_lateral` | Sideways offset, blocks | Alignment |
| `target_height` | Target height minus feet height, blocks | Upward or downward jump |
| `forward_velocity` | Forward speed, blocks/tick | Momentum |
| `vertical_velocity` | Vertical speed, blocks/tick | Rising or falling |
| `grounded` | Ground contact, yes/no | Jump availability |
| `highest_platform` | Furthest platform index | Select the next target |
| `current_platform_forward` | Last reached platform relative to the bot, blocks | Departure location |
| `current_platform_height` | Last reached platform height minus feet height, blocks | Height above departure |
| `lateral_velocity` | Sideways speed, blocks/tick | Drift |
| `jump_cooldown` | Automatic-jump cooldown, ticks | When another jump can start |
| `following_gap` | Distance between the next two platform centers, blocks | Following jump distance |
| `following_rise` | Height change after the target, blocks | Following jump height |
| `target_is_goal` | Whether the target is the goal, yes/no | Finish or continue |
| `time_remaining` | Decisions remaining | Time budget |

Scaling keeps different units on comparable ranges. For example, dividing a two-block height difference by four gives `0.5`. `np.clip(..., -1, 1)` then bounds the network inputs: `1.4` becomes `1`, while `0.5` stays `0.5`. It changes the observation, not Minecraft movement; differences outside the range are lost.

## Reward and platform contact

An ordinary decision gives -0.01, the goal gives +1 and a fall gives -1. `--shaped` adds +0.1 once for each newly reached intermediate platform.

A platform counts when the grounded bot's center is within half a block horizontally and 0.15 blocks vertically of its configured center. Q-learning checks every physics tick. PPO enables this with `--track-contacts`. Both use `learning/contacts.py` and keep the first goal or fall within an action.

## Course files

`Profile.load()` reads JSON such as:

```json
{
  "start_position": {"x": 19.5, "y": -6.0, "z": 13.5},
  "facing_yaw_degrees": 90,
  "landing_positions": [
    {"x": 19.5, "y": -6.0, "z": 13.5},
    {"x": 17.5, "y": -6.0, "z": 13.5}
  ],
  "fall_below_y": -8.0,
  "max_decisions": 80
}
```

`y` is feet height. The landing list includes the start and ends at the goal; yaw 90 faces west. Falling below `fall_below_y` or reaching `max_decisions` ends the attempt. Optional `scripted_actions` lists action names for the scripted controller.

The JSON describes existing blocks; it does not build them. `proposal-map1-single.json` describes one jump and `proposal-map1-three.json` describes three. `tools/courses/` imports worlds and generates courses with matching construction commands.

## Saved files

Q-learning writes a `.json` Q-table with its course and contact settings; PPO writes `.zip` weights and a `.json` settings file. Both write `.jsonl` transitions and episode summaries containing completion, progress, falls, decisions and elapsed time. Resume checks the training course; evaluation can use another course with compatible settings.

## Connection errors

`parse_reply()` checks incoming JSON and returns a `BridgeReply`. `_request()` sends a command and waits for that reply. A failed command, timeout or broken connection raises `BridgeError` with the operation name and original cause. The connection closes after failure so a late reply cannot become another action's result. `close()` releases the bot process and its pipes.
