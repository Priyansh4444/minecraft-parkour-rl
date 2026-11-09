// state and bounded client stepping for static parkour worlds.
import { Vec3 } from "vec3";
import type { ParkourBot, PhysicsFrame, ResetRequest, StepRequest } from "./protocol.ts";

const CONTROL_NAMES = ["forward", "back", "left", "right", "jump", "sprint"] as const;

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

  snapshot(startTick = this.getTick()): PhysicsFrame {
    const bot = this.bot;
    const entity = bot.entity;
    // all reads are synchronous; a physics event cannot split this snapshot
    return {
      position: [entity.position.x, entity.position.y, entity.position.z],
      velocity: [entity.velocity.x, entity.velocity.y, entity.velocity.z],
      health: bot.health,
      tick: this.getTick(),
      start_tick: startTick,
      elapsed_ticks: this.getTick() - startTick,
      on_ground: Boolean(entity.onGround),
      jump_ticks: bot.jumpTicks ?? 0,
      correction_count: this.correctionCount,
    };
  }

  pause() {
    // only this bot pauses; the dedicated server keeps ticking
    this.bot.physicsEnabled = false;
    this.bot.clearControlStates();
  }

  observe(): PhysicsFrame {
    this.pause();
    return this.snapshot();
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
        // pause in the listener before another physics tick can advance
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
    const { start, yaw } = request;
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
    return this.snapshot(startTick);
  }

  async step(request: StepRequest) {
    const { ticks, action } = request;
    const [longitudinal, strafe, jump, sprint, turn] = action;
    const keys = {
      forward: longitudinal === 1,
      back: longitudinal === 2,
      left: strafe === 1,
      right: strafe === 2,
      jump: jump === 1,
      sprint: sprint === 1,
    };
    const yawDelta = turn === 1 ? -15 : turn === 2 ? 15 : 0;
    if (this.bot.physicsEnabled) throw new Error("Call parkour_reset before stepped actions");
    const yaw = minecraftYaw(this.bot.entity.yaw) + yawDelta;
    await this.bot.look(Math.PI - (yaw * Math.PI) / 180, 0, true);
    for (const name of CONTROL_NAMES) this.bot.setControlState(name, keys[name]);
    const startTick = this.getTick();
    const tickFrames: PhysicsFrame[] = [];
    const advanced = this.waitForBoundary(
      () => {
        // record contacts before the action boundary pauses movement
        tickFrames.push(this.snapshot(startTick));
        return this.getTick() - startTick >= ticks;
      },
      ticks * 50 + 5000,
      "parkour step",
    );
    this.bot.physicsEnabled = true;
    await advanced;
    if (tickFrames.length !== ticks) throw new Error("Missing intermediate physics frame");
    return { ...this.snapshot(startTick), tick_frames: tickFrames };
  }
}
