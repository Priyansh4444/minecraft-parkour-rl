import type { Bot } from "mineflayer";

// mineflayer adds the jump cooldown at runtime
export type ParkourBot = Bot & { jumpTicks?: number };

export type Position = [x: number, y: number, z: number];

// keep the recorded button order used by both learners and saved experiments
export type Action = [
  longitudinal: 0 | 1 | 2,
  strafe: 0 | 1 | 2,
  jump: 0 | 1,
  sprint: 0 | 1,
  turn: 0 | 1 | 2,
];

export interface PhysicsFrame {
  position: Position;
  velocity: Position;
  health: number;
  tick: number;
  start_tick: number;
  elapsed_ticks: number;
  on_ground: boolean;
  jump_ticks: number;
  correction_count: number;
}

export interface ResetRequest {
  start: Position;
  yaw: number;
}
export interface StepRequest {
  action: Action;
  ticks: number;
}
export interface ConnectRequest {
  host: string;
  port: number;
  username: string;
  version: string;
}
export type BridgeRequest = { id: number } & (
  | ({ op: "connect" } & ConnectRequest)
  | ({ op: "parkour_reset" } & ResetRequest)
  | ({ op: "parkour_step" } & StepRequest)
  | { op: "parkour_observe" }
  | { op: "close" }
  | { op: "blocks"; positions: Position[] }
  | { op: "command"; command: string }
);

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("Expected a JSON object");
  return value as Record<string, unknown>;
}
function number(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value))
    throw new Error("Expected a finite number");
  return value;
}
function text(value: unknown): string {
  if (typeof value !== "string") throw new Error("Expected text");
  return value;
}
function position(value: unknown): Position {
  if (!Array.isArray(value) || value.length !== 3) throw new Error("Expected three coordinates");
  return [number(value[0]), number(value[1]), number(value[2])];
}
function action(value: unknown): Action {
  const limits = [3, 3, 2, 2, 3];
  if (
    !Array.isArray(value) ||
    value.length !== limits.length ||
    value.some(
      (button, index) => !Number.isInteger(button) || button < 0 || button >= limits[index],
    )
  ) {
    throw new Error(
      "Action requires [longitudinal 0..2, strafe 0..2, jump 0..1, sprint 0..1, turn 0..2]",
    );
  }
  return value as Action;
}

// validate external json once; movement receives concrete types
export function parseRequest(line: string): BridgeRequest {
  const request = object(JSON.parse(line));
  const id = number(request.id);
  switch (request.op) {
    case "connect":
      return {
        id,
        op: "connect",
        host: text(request.host),
        port: number(request.port),
        username: text(request.username),
        version: text(request.version),
      };
    case "parkour_reset":
      return {
        id,
        op: "parkour_reset",
        start: position(request.start),
        yaw: number(request.yaw),
      };
    case "parkour_step": {
      const ticks = number(request.ticks);
      if (!Number.isInteger(ticks) || ticks < 1 || ticks > 20)
        throw new Error("Step ticks must be an integer 1..20");
      return {
        id,
        op: "parkour_step",
        action: action(request.action),
        ticks,
      };
    }
    case "parkour_observe":
      return { id, op: "parkour_observe" };
    case "close":
      return { id, op: request.op };
    case "blocks": {
      if (!Array.isArray(request.positions)) throw new Error("Expected block positions");
      return { id, op: "blocks", positions: request.positions.map(position) };
    }
    case "command":
      return { id, op: "command", command: text(request.command) };
    default:
      throw new Error(`Unknown operation: ${String(request.op)}`);
  }
}
