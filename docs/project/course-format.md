# Course configuration

JSON is a good fit here: both Python and TypeScript can read it directly, and a course is just structured data. Named fields make it understandable without changing formats or adding another parser.

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

- `start_position` is where the bot resets. `y` is its feet height, not the support block's height.
- `facing_yaw_degrees` gives the initial direction; 90 faces west.
- `landing_positions` describes the ordered route. The first position is the starting platform and the last is the goal.
- `fall_below_y` ends a failed attempt.
- `max_decisions` limits the number of timed actions in one attempt.

Landing positions belong in this file because the compact learner needs to know the next target. It does not infer platforms from images or discover the route automatically. These coordinates define the task; the actual blocks remain in the Minecraft world. Changing the coordinates does not build or move any blocks.

An optional `scripted_actions` list uses names such as `walk_jump` and `walk`. It is a reference sequence for inspecting the route, not instructions supplied to Q-learning or PPO. The simple comparison controller always uses `run_jump` on every course.

Recorded run files can still use the older coordinate-array format. The loader decodes both formats into the same course, preserving existing model compatibility. PPO checks the decoded course rather than rejecting a model merely because its JSON file was reformatted.

Freeze evaluation layouts before training. Training on one layout can specialize to it; successful training-course runs do not establish success on unseen layouts.
