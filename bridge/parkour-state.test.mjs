import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import vec3 from "vec3";
import { decodeAction, readTerrain, SteppedParkour, validateGrid } from "./parkour-state.mjs";

const { Vec3 } = vec3;
const grid = { radius: 1, down: 0, up: 0 };

function fixture() {
  const bot = new EventEmitter();
  let tick = 0;
  bot.on("physicsTick", () => tick++);
  bot.entity = {
    position: new Vec3(0.5, 0, 0.5),
    velocity: new Vec3(0, 0, 0),
    yaw: Math.PI / 2,
    onGround: true,
    isCollidedVertically: true,
  };
  bot.controlState = {};
  bot.health = bot.food = 20;
  bot.physicsEnabled = true;
  bot.setControlState = (name, pressed) => {
    bot.controlState[name] = pressed;
  };
  bot.clearControlStates = () => {
    for (const name of Object.keys(bot.controlState)) bot.controlState[name] = false;
  };
  bot.blockAt = () => ({ name: "air", shapes: [] });
  bot.look = async (yaw) => {
    bot.entity.yaw = yaw;
  };
  bot.advance = (count) => {
    for (let i = 0; i < count; i++) {
      if (bot.physicsEnabled) bot.emit("physicsTick");
    }
  };
  return { bot, movement: new SteppedParkour(bot, () => tick) };
}

test("grid shape, world coordinates, collision, hazard and unknown channels", () => {
  const { bot } = fixture();
  bot.entity.position = new Vec3(-0.1, 0, 0.5);
  bot.blockAt = (position) => {
    if (position.x === -2 && position.z === 0) return null;
    if (position.x === -1 && position.z === 0)
      return { name: "stone", shapes: [[0, 0, 0, 1, 1, 1]] };
    if (position.x === 0 && position.z === 0)
      return { name: "water", shapes: [[0, 0, 0, 1, 1, 1]] };
    return { name: "air", shapes: [] };
  };
  const terrain = readTerrain(bot, grid);
  assert.deepEqual(terrain.shape, [3, 1, 3, 3]);
  assert.equal(terrain.data.length, 27);
  assert.equal(terrain.data[4], 1); // stone at floor(player.x), not trunc(player.x)
  assert.equal(terrain.data[5], 0); // water is not a collision cube
  assert.equal(terrain.data[9 + 5], 1);
  assert.equal(terrain.data[18 + 3], 1);
  assert.equal(terrain.data[3], 0);
});

test("default grid matches the agreed C/Y/Z/X dimensions and bounded input", () => {
  const { bot } = fixture();
  const terrain = readTerrain(bot);
  assert.deepEqual(terrain.shape, [3, 13, 15, 15]);
  assert.equal(terrain.data.length, 8775);
  assert.throws(() => validateGrid({ radius: -1, down: 9, up: 3 }));
  assert.throws(() => validateGrid({ radius: 16, down: 32, up: 32 }));
  assert.throws(() => validateGrid({ radius: 1.5, down: 0, up: 0 }));
});

test("factored actions preserve opposite directions and left/right yaw signs", () => {
  assert.deepEqual(decodeAction([2, 2, 1, 0, 1]), {
    keys: { forward: false, back: true, left: false, right: true, jump: true, sprint: false },
    yaw_delta: -15,
  });
  assert.equal(decodeAction([0, 0, 0, 0, 2]).yaw_delta, 15);
  for (const action of [
    [3, 0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, NaN],
  ]) {
    assert.throws(() => decodeAction(action));
  }
});

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
