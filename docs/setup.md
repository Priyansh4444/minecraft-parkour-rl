# Setup

Use Python 3.12, Node.js 22.18 or newer, Java 25 and Minecraft Java 26.1.2.

## Dependencies and maps

```sh
uv sync --extra ppo --python 3.12
npm ci --prefix bridge --ignore-scripts
uv run python -m tools.courses.import_maps ~/Downloads/maps.zip
```

The importer creates `.local/worlds/RL-Map-1`, `RL-Map-2` and `RL-Map-3`. It skips session locks and refuses to overwrite existing worlds. Reuse an existing import, or choose another destination with `--saves PATH`.

## Server configuration

Place the Minecraft 26.1.2 `server.jar` in `.local/mc2612/server/`. Run it with Java 25 to create the settings. Read Minecraft's EULA and set `eula=true` if you accept it.

With the server stopped, copy `RL-Map-1` into the server folder as `world`. Keep the imported copy separate. Set these values in `server.properties`:

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

Start Java in that folder:

```sh
/path/to/java25/bin/java -jar server.jar nogui
```

Enter these commands in the server console, then stop it:

```text
gamerule fall_damage false
gamerule immediate_respawn true
setworldspawn 19 -6 13
stop
```

Use the [README command](../README.md) to start the prepared server. The launcher grants the bot operator permission for resets. The server and learner must use the same `--port`; the default is 25566 for the server and Q-learning. Ctrl+C stops each process in its own terminal.

## Spectator mode

Add `--watch-player YourName` to the server command. Open Minecraft Java 26.1.2 and connect to `127.0.0.1:25566`. Start the learner in another terminal. The launcher puts your player in spectator mode and follows the bot, `Watcher`.

Omit `--watch-player` for headless operation. Closing the game window leaves training running; the server and computer must stay on. Installed dependencies allow training without internet.

If using another bot name, match server `--bot-name NAME` with learner `--username NAME`. A refused connection usually means the server is not ready or the port differs. A failed reset usually means the bot lacks operator permission or the starting platform is missing.
