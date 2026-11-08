# Server setup and headless training

Both modes use a local Minecraft Java server and a Mineflayer bot. Headless mode has no game window. Visible mode adds your Minecraft client as a spectator and keeps the same learner and movement controls.

## Prepare once

Use Python 3.12, Node.js 22.18 or newer, **Java 25**, and the Minecraft Java **26.1.2** server JAR. Check `java --version`: Java 21 cannot run this server. Supply the Java 25 executable with `--java` if it is not your default.

```sh
uv sync --extra ppo --python 3.12
npm ci --prefix bridge --ignore-scripts
python -m tools.courses.import_maps ~/Downloads/maps.zip
```

The importer refuses to overwrite existing worlds. Keep the original ZIP and clean copies under `.local/worlds/`. Put `server.jar` in a separate server folder, such as `.local/mc2612/server/`. Run Java once to create its configuration, read Minecraft's EULA and set `eula=true` if you accept it. With that server stopped, copy `.local/worlds/RL-Map-1` into the folder as `world`. Move an existing generated world aside first; do not merge saves.

Set these values in `server.properties`:

```properties
server-ip=127.0.0.1
server-port=25566
online-mode=false
enforce-secure-profile=false
level-name=world
gamemode=adventure
force-gamemode=true
difficulty=peaceful
spawn-protection=0
pause-when-empty-seconds=0
allow-flight=true
```

For a new working world, start Java directly once and enter the course settings in its server console:

```text
gamerule fall_damage false
gamerule immediate_respawn true
stop
```

These settings persist in the working world. The project launcher grants operator permission to `Watcher` for resets but does not change gamerules or construct courses.

## Start headlessly

From the repository root:

```sh
uv run python -m parkour.minecraft.server --directory .local/mc2612/server --java /path/to/java25/bin/java
```

Wait for the ready message. In a second terminal:

```sh
uv run python -m parkour.learning.q_learning --course parkour/courses/proposal-map1-single.json --output .local/runs/q/seed-11/q --episodes 300 --seed 11
```

After training, evaluate without updating the saved table:

```sh
uv run python -m parkour.learning.q_learning --course parkour/courses/proposal-map1-single.json --output .local/runs/q/seed-11/q --episodes 50 --seed 11 --eval
```

Use a fresh prefix for an independent run. `--resume` deliberately continues a compatible table. The [experiment runner](../project/proposal-scope.md#experiments) supplies independent seeds and the constant forward+sprint+jump baseline. Low-level `--scripted` replays the sequence in its supplied profile.

Ctrl+C stops the learner and closes its bot. Ctrl+C in the server terminal saves and stops only that server. The launcher refuses to replace an occupied port. Independent experiments need separate working folders and ports; pass the same `--port` to server and learner. Do not run two bots named `Watcher` on one server.

For a game window, use [watching mode](watching.md). First-time installations require downloads; later offline operation uses the installed dependencies.

A connection refusal usually means the server is not ready or the port differs. A reset failure requires checking operator permission, course coordinates and support blocks. Missing chunks, corrections during an action and malformed observations fail the run rather than becoming training data.

Mineflayer advances its client physics for each action while the server world continues running. This tested loop is not a guarantee of exact human-client movement or authoritative server-side lockstep.
