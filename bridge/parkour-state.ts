// State and bounded client stepping for static parkour worlds.
import { Vec3 } from "vec3";
import type { Grid, ParkourBot, ResetRequest, StepRequest } from "./protocol.ts";

const CONTROL_NAMES = ["forward", "back", "left", "right", "jump", "sprint"] as const;
const HAZARDS = new Set([
  "water",
  "lava",
  "fire",
  "soul_fire",
  "magma_block",
  "cactus",
  "campfire",
  "soul_campfire",
  "sweet_berry_bush",
  "powder_snow",
]);

export function validateGrid(grid: Grid = { radius: 7, down: 9, up: 3 }) {
  if (
    !grid ||
    ![grid.radius, grid.down, grid.up].every(Number.isInteger) ||
    grid.radius < 0 ||
    grid.radius > 16 ||
    grid.down < 0 ||
    grid.up < 0 ||
    grid.down > 32 ||
    grid.up > 32
  ) {
    throw new Error("Grid requires integer radius 0..16 and down/up 0..32");
  }
  const width = 2 * grid.radius + 1;
  const height = grid.down + grid.up + 1;
  if (width * width * height > 15000) throw new Error("Grid exceeds 15000 cells");
  return { ...grid, width, height };
}

export function decodeAction(action: unknown) {
  const limits = [3, 3, 2, 2, 3];
  if (
    !Array.isArray(action) ||
    action.length !== limits.length ||
    action.some((value, i) => !Number.isInteger(value) || value < 0 || value >= limits[i])
  ) {
    throw new Error(
      "Action must contain [longitudinal 0..2, strafe 0..2, jump 0..1, sprint 0..1, turn 0..2]",
    );
  }
  return {
    keys: {
      forward: action[0] === 1,
      back: action[0] === 2,
      left: action[1] === 1,
      right: action[1] === 2,
      jump: action[2] === 1,
      sprint: action[3] === 1,
    },
    yaw_delta: action[4] === 1 ? -15 : action[4] === 2 ? 15 : 0,
  };
}

export function readTerrain(bot: ParkourBot, requestedGrid?: Grid) {
  const grid = validateGrid(requestedGrid);
  const origin = bot.entity.position.floored();
  const cells = grid.width * grid.width * grid.height;
  const data = new Array(cells * 3).fill(0);
  let index = 0;
  for (let y = -grid.down; y <= grid.up; y++) {
    for (let z = -grid.radius; z <= grid.radius; z++) {
      for (let x = -grid.radius; x <= grid.radius; x++, index++) {
        const block = bot.blockAt(new Vec3(origin.x + x, origin.y + y, origin.z + z), false);
        if (!block) {
          data[2 * cells + index] = 1;
          continue;
        }
        const liquid = block.name === "water" || block.name === "lava";
        data[index] = !liquid && Array.isArray(block.shapes) && block.shapes.length > 0 ? 1 : 0;
        data[cells + index] = HAZARDS.has(block.name) ? 1 : 0;
      }
    }
  }
  return { shape: [3, grid.height, grid.width, grid.width], data };
}

export function minecraftYaw(yaw: number) {
  return ((180 - (yaw * 180) / Math.PI + 540) % 360) - 180;
}

export class SteppedParkour {
  readonly bot: ParkourBot;
  readonly getTick: () => number;
  correctionCount: number;

  constructor(bot: ParkourBot, getTick: () => number) {
    this.bot = bot;
    this.getTick = getTick;
    this.correctionCount = 0;
    bot.on("forcedMove", () => {
      this.correctionCount++;
    });
  }

  snapshot(grid?: Grid, startTick = this.getTick()) {
    const bot = this.bot;
    const entity = bot.entity;
    // All reads here are synchronous: no physics event can split this snapshot.
    return {
      position: entity.position.toArray(),
      velocity: entity.velocity.toArray(),
      yaw: minecraftYaw(entity.yaw),
      health: bot.health,
      food: bot.food,
      tick: this.getTick(),
      start_tick: startTick,
      elapsed_ticks: this.getTick() - startTick,
      on_ground: Boolean(entity.onGround),
      collided_horizontal: Boolean(entity.isCollidedHorizontally),
      collided_vertical: Boolean(entity.isCollidedVertically),
      is_in_water: Boolean(entity.isInWater),
      is_in_lava: Boolean(entity.isInLava),
      is_in_web: Boolean(entity.isInWeb),
      jump_ticks: bot.jumpTicks ?? 0,
      jump_queued: Boolean(bot.jumpQueued),
      correction_count: this.correctionCount,
      controls: Object.fromEntries(
        CONTROL_NAMES.map((name) => [name, Boolean(bot.controlState[name])]),
      ),
      terrain: readTerrain(bot, grid),
    };
  }

  pause() {
    // Only this bot's client movement pauses; the dedicated server keeps ticking.
    this.bot.physicsEnabled = false;
    this.bot.clearControlStates();
  }

  observe(grid?: Grid) {
    validateGrid(grid);
    this.pause();
    return this.snapshot(grid);
  }

  waitForBoundary(check: () => boolean, timeout: number, operation: string) {
    const bot = this.bot;
    return new Promise<void>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout>;
      let completed = false;
      const cleanup = () => {
        clearTimeout(timer);
        bot.removeListener("physicsTick", onTick);
        bot.removeListener("end", onEnd);
        bot.removeListener("error", onError);
      };
      const finish = (error?: unknown) => {
        if (completed) return;
        completed = true;
        // Do this in the event listener, before a catch-up loop can advance again.
        this.pause();
        cleanup();
        if (error) reject(error);
        else resolve();
      };
      const onTick = () => {
        try {
          if (check()) finish();
        } catch (error) {
          finish(error);
        }
      };
      const onEnd = (reason: string) =>
        finish(new Error(`Disconnected during ${operation}: ${reason}`));
      const onError = (error: Error) => finish(error);
      timer = setTimeout(() => finish(new Error(`Timed out during ${operation}`)), timeout);
      bot.on("physicsTick", onTick);
      bot.once("end", onEnd);
      bot.once("error", onError);
    });
  }

  async reset(request: ResetRequest) {
    const { start, yaw, grid } = request;
    validateGrid(grid);
    if (
      !Array.isArray(start) ||
      start.length !== 3 ||
      !start.every(Number.isFinite) ||
      !Number.isFinite(yaw)
    ) {
      throw new Error("Reset requires three finite start coordinates and finite yaw");
    }
    this.pause();
    const startTick = this.getTick();
    const beforeCorrection = this.correctionCount;
    let stable = 0;
    const settled = this.waitForBoundary(
      () => {
        const entity = this.bot.entity;
        const close = entity.position.distanceTo(new Vec3(...start)) < 0.15;
        const stopped = Math.hypot(entity.velocity.x, entity.velocity.z) < 0.03;
        stable =
          this.correctionCount > beforeCorrection && close && stopped && entity.onGround
            ? stable + 1
            : 0;
        return stable >= 2;
      },
      5000,
      "parkour reset (check operator permissions and start platform)",
    );
    this.bot.chat(`/tp @s ${start.join(" ")} ${yaw} 0`);
    this.bot.physicsEnabled = true;
    await settled;
    return this.snapshot(grid, startTick);
  }

  async step(request: StepRequest) {
    const { ticks, grid } = request;
    validateGrid(grid);
    const { keys, yaw_delta: yawDelta } = decodeAction(request.action);
    if (!Number.isInteger(ticks) || ticks < 1 || ticks > 20)
      throw new Error("Step ticks must be an integer 1..20");
    if (this.bot.physicsEnabled) throw new Error("Call parkour_reset before stepped actions");
    const yaw = minecraftYaw(this.bot.entity.yaw) + yawDelta;
    await this.bot.look(Math.PI - (yaw * Math.PI) / 180, 0, true);
    for (const name of CONTROL_NAMES) this.bot.setControlState(name, keys[name]);
    const startTick = this.getTick();
    const advanced = this.waitForBoundary(
      () => this.getTick() - startTick >= ticks,
      ticks * 50 + 5000,
      "parkour step",
    );
    this.bot.physicsEnabled = true;
    await advanced;
    return this.snapshot(grid, startTick);
  }
}
