# Installation and first run

The official-launcher and single-player workflow has **not yet been tested end to end on any operating system**, including macOS and Windows. These instructions set up a local test world. Report setup errors before relying on it for experiments.

## Requirements

- [Minecraft: Java Edition and the official Minecraft Launcher](https://www.minecraft.net/download), with a licensed account.
- [Python 3.12](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads) (or download the project files as a ZIP).
- Minecraft **26.1**, Fabric Loader **0.19.3**, Minescript **5.0 for Fabric 26.1**, and Fabric API **0.155.3+26.1.2**. Use these versions together; the official-launcher setup still needs an end-to-end test on each platform.

Fabric and Minescript are mods for Minecraft Java Edition. Neither a separately running server nor Node.js is required for this workflow. On macOS, the [Fabric installer may require Java](https://docs.fabricmc.net/players/installing-fabric/macos) to run.

## Install

1. Install the official launcher, sign in, and launch Minecraft Java Edition **26.1** once. Quit Minecraft and the launcher.
2. Download and run the [Fabric installer](https://fabricmc.net/use/installer/). Choose **Client**, Minecraft **26.1** and loader **0.19.3**, leave **Create profile** checked, and install.
3. Open the selected installation's game directory. Defaults are `%appdata%\.minecraft` on Windows, `~/Library/Application Support/minecraft` on macOS, and `~/.minecraft` on Linux. On macOS, use Finder → **Go → Go to Folder…**. If the launcher's installation has a custom **Game Directory**, use that path instead. Create a `mods` folder there if needed.
4. Place [Minescript Fabric 26.1 5.0](https://cdn.modrinth.com/data/KcpXWngB/versions/GHPXOXZB/minescript-fabric-26.1-5.0.jar) and [Fabric API 0.155.3+26.1.2](https://cdn.modrinth.com/data/P7dR8mSH/versions/3dM0X6ou/fabric-api-0.155.3%2B26.1.2.jar) in that `mods` folder. Launch the Fabric installation from the official launcher once, then quit. Minescript should create a `minescript` folder in the same game directory.
5. Obtain this repository. Copy `parkour/agent.py`, `parkour/build_course.py`, and `parkour/course.json` into the game directory's `minescript` folder. Repeat the copy after changing these files in your local checkout: editing the repository does not update the game folder automatically.
6. Set Minescript's Python interpreter. In `minescript/config.txt`, set `python` to your **own** Python 3.12 executable path; leave other settings intact. To find the path:

   - macOS/Linux terminal: `python3.12 -c 'import sys; print(sys.executable)'`
   - Windows PowerShell: `py -3.12 -c "import sys; print(sys.executable)"`

   Use that output in a line such as `python="/absolute/path/to/python3.12"`. On Windows, use the path printed on your machine; forward slashes also work. These scripts use standard Python; don't assume the integrated Pyjinn interpreter supports them unchanged. See [Minescript configuration](https://minescript.net/docs#configuration).

## Build and run

1. Start the Fabric installation. Create a **new Superflat single-player world with cheats enabled**. Use a dedicated world: the builder overwrites blocks at fixed coordinates near `(0, -60, 0)`.
2. Open Minecraft chat and enter `\build_course`. Wait for the completion message, then inspect the start platform, three gaps and gold goal. The message means commands were sent, not that every block was verified. Run the builder once in each new world.
3. In chat enter `\agent 3 1.0 1.0`. This runs three episodes with fully random exploration. Progress and final outcomes appear in chat. Try `\agent 20 0.5 0.05` for 20 training episodes with decreasing exploration. Run only one batch at a time. For a seeded random baseline, use `\agent 20 --random --seed 7`; after training, evaluate without updates using `\agent 20 --eval --seed 8`.
4. The game directory's `minescript/agent-log.jsonl` and `minescript/linear-q.json` are local generated outputs. Later training runs reuse the weights; move or delete it deliberately for a fresh run. They are not synchronized with the repository automatically.

A few completed episodes are a smoke test, not evidence that the agent learned or generalizes.

## Troubleshooting

- **No Fabric profile:** Re-run the installer for the client and Minecraft 26.1, then select the Fabric profile in the official launcher.
- **Commands not found:** Check the two mod JARs in the selected installation's `mods` folder, restart Minecraft, and try `\ls`. Check that the three project files are in the same installation's `minescript` folder. The commands use a **backslash**.
- **Python not found:** Verify Python 3.12 runs and check its exact path in `config.txt`; `/usr/bin/python3` may not be the version installed on macOS. Check in-game errors and `logs/latest.log`.
- **Teleport or course fails:** Confirm cheats are enabled, the world is Superflat, and `\build_course` finished. Check the game log and use a dedicated test world.
- **A script gets stuck:** Enter `\jobs` in chat and then `\killjob JOB_ID` with the displayed number. Inspect `minescript/agent-log.jsonl`. Timing and reset behavior still need validation.
