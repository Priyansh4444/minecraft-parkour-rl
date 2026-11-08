# Watch training in Minecraft

Visible mode shows the **same bot, server, actions and learner** as headless mode. Your player is a separate spectator and does not supply the bot's controls.

1. Complete [server setup](server.md). Stop an existing server before starting the same working folder with the launcher.
2. Start it with your Minecraft username:

   ```sh
   uv run python -m parkour.minecraft.server --directory .local/mc2612/server --java /path/to/java25/bin/java --watch-player YourName
   ```

3. Open Minecraft Java **26.1.2** normally. Choose Multiplayer → Direct Connection → `127.0.0.1:25566`.
4. In another terminal start Q-learning, greedy evaluation or PPO using the ordinary command. The bot uses `Watcher` by default. Your username must be different.

The launcher waits until both players have joined, switches only your player to spectator mode, teleports you to the bot and attaches your camera. Either join order works. If either player reconnects, the launcher attaches the spectator again.

Closing the game window does not stop the learner. Ctrl+C in the learner terminal stops the learner; Ctrl+C in the server terminal saves and stops the server. Omit `--watch-player` for a headless start. The same server can also continue training with the observer disconnected.

If changing the bot name, use matching server `--bot-name NAME` and learner `--username NAME` options. `--port` must match the launcher, Minecraft connection address and learner. The launcher does not install or start your graphical client.

The viewer's spectator mode is a saved player setting in this working world. To play normally later, an operator can run `/gamemode adventure YourName`; use a separate play world to preserve experiment settings.

Server-side spectator setup and uninterrupted bot stepping are checked with two actual protocol clients. Rendering in a graphical Minecraft window is a manual join check, not something those protocol tests establish.
