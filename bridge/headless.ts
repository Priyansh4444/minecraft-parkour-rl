// A JSON-lines transport. Python owns observations and learning.
import readline from "node:readline";
import mineflayer, { type Bot, type ControlState } from "mineflayer";
import { Vec3 } from "vec3";
import { SteppedParkour } from "./parkour-state.ts";
import { type BridgeRequest, type ConnectRequest, parseRequest } from "./protocol.ts";

// Keep dependency diagnostics out of the JSON response stream.
console.log = (...args) => console.error(...args);
let bot: Bot;
let tick = 0;
let disconnected: string | undefined;
let parkour: SteppedParkour;

async function connect(options: ConnectRequest) {
  if (bot) throw new Error("Already connected");
  bot = mineflayer.createBot({
    host: options.host,
    port: options.port,
    username: options.username,
    version: options.version,
    auth: "offline",
  });
  bot.on("physicsTick", () => {
    tick++;
  });
  parkour = new SteppedParkour(bot, () => tick);
  bot.on("error", (error) => {
    disconnected = error.message;
    console.error("Mineflayer:", disconnected);
  });
  bot.on("kicked", (reason) => {
    disconnected = `Kicked: ${JSON.stringify(reason)}`;
  });
  bot.on("end", (reason) => {
    disconnected ||= `Disconnected: ${reason}`;
  });
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error("Timed out joining the local server")), 18000);
    const onError = (error: Error) => finish(error);
    const onEnd = () => finish(new Error(disconnected || "Disconnected while joining"));
    const onSpawn = async () => {
      try {
        await bot.waitForChunksToLoad();
        await bot.waitForTicks(2);
        finish();
      } catch (error) {
        finish(error);
      }
    };
    function finish(error?: unknown) {
      clearTimeout(timer);
      bot.removeListener("error", onError);
      bot.removeListener("end", onEnd);
      bot.removeListener("spawn", onSpawn);
      if (error) reject(error);
      else resolve();
    }
    bot.once("error", onError);
    bot.once("end", onEnd);
    bot.once("spawn", onSpawn);
  });
  return { username: bot.username, tick };
}

function snapshot() {
  const entity = bot.entity;
  return {
    position: entity.position.toArray(),
    velocity: entity.velocity.toArray(),
    yaw: ((180 - (entity.yaw * 180) / Math.PI + 540) % 360) - 180,
    health: bot.health,
    tick,
  };
}

async function handle(request: BridgeRequest) {
  if (request.op === "connect") return connect(request);
  if (request.op === "close") {
    if (bot) {
      bot.clearControlStates();
      bot.quit();
    }
    return true;
  }
  if (!bot || disconnected) throw new Error(disconnected || "Not connected");
  if (!bot.entity) throw new Error("Player state is unavailable");
  switch (request.op) {
    case "parkour_reset":
      return parkour.reset(request);
    case "parkour_step":
      return parkour.step(request);
    case "parkour_observe":
      return parkour.observe(request.grid);
    case "snapshot":
      return snapshot();
    case "blocks":
      return request.positions.map((position) => {
        if (!Array.isArray(position) || position.length !== 3 || !position.every(Number.isFinite)) {
          throw new Error("Block position must contain three finite coordinates");
        }
        const block = bot.blockAt(new Vec3(...position));
        if (!block) throw new Error(`Chunk is not loaded at ${position.join(",")}`);
        return `minecraft:${block.name}`;
      });
    case "controls":
      for (const [name, pressed] of Object.entries(request.keys)) {
        if (
          !["forward", "back", "left", "right", "sprint", "jump"].includes(name) ||
          typeof pressed !== "boolean"
        ) {
          throw new Error("Unknown control or non-boolean control value");
        }
        bot.setControlState(name as ControlState, pressed);
      }
      return true;
    case "look":
      await bot.look(
        Math.PI - (request.yaw * Math.PI) / 180,
        (-request.pitch * Math.PI) / 180,
        true,
      );
      return true;
    case "command":
      if (typeof request.command !== "string" || /[\r\n]/.test(request.command)) {
        throw new Error("Expected one Minecraft command");
      }
      bot.chat(`/${request.command}`);
      return true;
    default:
      throw new Error(`Unknown operation: ${request.op}`);
  }
}

const input = readline.createInterface({ input: process.stdin });
let pending = Promise.resolve();
input.on("line", (line) => {
  pending = pending.then(async () => {
    let request: BridgeRequest | undefined;
    try {
      request = parseRequest(line);
      const result = await handle(request);
      process.stdout.write(`${JSON.stringify({ id: request.id, result })}\n`);
    } catch (error) {
      process.stdout.write(
        `${JSON.stringify({ id: request?.id, error: error instanceof Error ? error.message : String(error) })}\n`,
      );
    }
  });
});
input.on("close", () => {
  if (bot) {
    bot.clearControlStates();
    bot.quit();
  }
});
