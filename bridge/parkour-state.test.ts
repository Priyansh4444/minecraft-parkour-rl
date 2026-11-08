import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import type { ControlState } from "mineflayer";
import { Vec3 } from "vec3";
import { SteppedParkour } from "./parkour-state.ts";
import type { BlockSample, ParkourBot } from "./protocol.ts";

const grid = { radius: 1, down: 0, up: 0 };

class TestBot extends EventEmitter implements ParkourBot {
  entity = {
    position: new Vec3(0.5, 0, 0.5),
    velocity: new Vec3(0, 0, 0),
    yaw: Math.PI / 2,
    onGround: true,
    isCollidedVertically: true,
  };
  controlState: Partial<Record<ControlState, boolean>> = {};
  health = 20;
  food = 20;
  physicsEnabled = true;
  setControlState(name: ControlState, pressed: boolean) {
    this.controlState[name] = pressed;
  }
  clearControlStates() {
    for (const name of Object.keys(this.controlState) as ControlState[])
      this.controlState[name] = false;
  }
  blockAt: (position: Vec3, extraInfo?: boolean) => BlockSample | null = () => ({
    name: "air",
    shapes: [],
  });
  look = async (yaw: number) => {
    this.entity.yaw = yaw;
  };
  chat = (_command: string) => undefined;
  advance(count: number) {
    for (let i = 0; i < count; i++) if (this.physicsEnabled) this.emit("physicsTick");
  }
}

function fixture() {
  const bot = new TestBot();
  let tick = 0;
  bot.on("physicsTick", () => tick++);
  return { bot, movement: new SteppedParkour(bot, () => tick) };
}

test("physics stops inside the final tick, even within one catch-up loop", async () => {
  const { bot, movement } = fixture();
  movement.observe(grid);
  const step = movement.step({ action: [1, 0, 1, 0, 1], ticks: 4, grid });
  await Promise.resolve(); // look() resolves before movement resumes
  bot.advance(10);
  const state = await step;
  assert.equal(state.start_tick, 0);
  assert.equal(state.elapsed_ticks, 4);
  assert.equal(state.tick, 4);
  assert.equal(bot.physicsEnabled, false);
  assert.ok(Object.values(state.controls).every((value) => value === false));
  assert.ok(Math.abs(state.yaw - 75) < 1e-8);
  bot.advance(10);
  assert.equal(movement.snapshot(grid).tick, 4);
});

test("reset waits for teleport acknowledgement and stable contact", async () => {
  const { bot, movement } = fixture();
  bot.chat = (command) => {
    assert.equal(command, "/tp @s 3.5 4 5.5 90 0");
    queueMicrotask(() => {
      bot.entity.position = new Vec3(3.5, 4, 5.5);
      bot.emit("forcedMove");
      bot.advance(10);
    });
  };
  const state = await movement.reset({ start: [3.5, 4, 5.5], yaw: 90, grid });
  assert.deepEqual(state.position, [3.5, 4, 5.5]);
  assert.equal(state.elapsed_ticks, 2);
  assert.equal(state.correction_count, 1);
  assert.equal(bot.physicsEnabled, false);
});

test("disconnect during a step rejects and releases controls", async () => {
  const { bot, movement } = fixture();
  movement.observe(grid);
  const step = movement.step({ action: [1, 0, 0, 1, 0], ticks: 4, grid });
  await Promise.resolve();
  bot.emit("end", "test disconnect");
  await assert.rejects(step, /Disconnected during parkour step/);
  assert.equal(bot.physicsEnabled, false);
  assert.ok(Object.values(bot.controlState).every((value) => value === false));
  assert.equal(bot.listenerCount("physicsTick"), 1); // only the fixture's tick counter
});
