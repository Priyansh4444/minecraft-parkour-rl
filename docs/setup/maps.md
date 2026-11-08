# Import the downloaded maps

The supplied `maps.zip` contains Java Edition worlds named `Map 1`, `map 2`, and `MAP 3`. Their `level.dat` files record **Minecraft 26.1.2**. Use that server version for these experiments; do not open the saves in the older 26.1 prototype version.

Keep the original ZIP. From the repository directory, run:

```sh
python -m tools.courses.import_maps ~/Downloads/maps.zip
```

The importer creates working copies in `.local/worlds/RL-Map-1`, `RL-Map-2`, and `RL-Map-3`. It validates archive paths, skips copied session locks and refuses to overwrite existing destinations. If these copies already exist, reuse them rather than rerunning the import over them. `--saves PATH` selects a fresh destination.

Copy `RL-Map-1` into a separate server folder as `world`, following [the headless server setup](server.md). Leave the original imported copies intact. No Fabric, Minescript mod or graphical game client is required for the current path.

The first calibration profile is `parkour/courses/proposal-map1-single.json`: feet start at `(19.5, -6, 13.5)` and the landing is `(17.5, -6, 13.5)`. The next training layout is `proposal-map1-three.json`, a straight three-jump section with a raised final landing. Neither profile describes the entire supplied map. Five forward/jump actions cannot navigate its turns.

Use [the proposal experiment plan](../project/proposal-scope.md) for training, independent seeds and evaluation. Keep [the frozen randomized course](../experiments/random-course.md) and reserved starts out of training and tuning. Training and baselines need identical server settings and episode limits; exploratory successes and no-update greedy evaluation are separate measurements.

Minecraft's server and the computer must remain running, but its graphical client can stay closed. Mineflayer client stepping is tested separately from the continuously running server world; it does not establish exact equivalence to a human client. See [test results](../experiments/verification.md).
