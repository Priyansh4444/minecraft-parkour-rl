import type { Bot, ControlState } from "mineflayer";
import type { Vec3 } from "vec3";

export interface Grid {
  radius: number;
  down: number;
  up: number;
}
export interface BlockSample {
  name: string;
  shapes: readonly (readonly number[])[];
}
export interface MovementEntity {
  position: Vec3;
  velocity: Vec3;
  yaw: number;
  onGround: boolean;
  isCollidedHorizontally?: boolean;
  isCollidedVertically?: boolean;
  isInWater?: boolean;
  isInLava?: boolean;
  isInWeb?: boolean;
}
export interface ParkourBot
  extends Pick<
    Bot,
    | "health"
    | "food"
    | "physicsEnabled"
    | "setControlState"
    | "clearControlStates"
    | "look"
    | "chat"
  > {
  on(event: "forcedMove" | "physicsTick", listener: () => void): unknown;
  once(event: "end", listener: (reason: string) => void): unknown;
  once(event: "error", listener: (error: Error) => void): unknown;
  removeListener(event: "physicsTick", listener: () => void): unknown;
  removeListener(event: "end", listener: (reason: string) => void): unknown;
  removeListener(event: "error", listener: (error: Error) => void): unknown;
  entity: MovementEntity;
  controlState: Partial<Record<ControlState, boolean>>;
  jumpTicks?: number;
  jumpQueued?: boolean;
  blockAt(position: Vec3, extraInfo?: boolean): BlockSample | null;
}
export interface ResetRequest {
  start: [number, number, number];
  yaw: number;
  grid?: Grid;
}
export interface StepRequest {
  action: unknown;
  ticks: number;
  grid?: Grid;
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
  | { op: "parkour_observe"; grid?: Grid }
  | { op: "close" | "snapshot" }
  | { op: "blocks"; positions: [number, number, number][] }
  | { op: "controls"; keys: Record<string, boolean> }
  | { op: "look"; yaw: number; pitch: number }
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
function position(value: unknown): [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3) throw new Error("Expected three coordinates");
  return [number(value[0]), number(value[1]), number(value[2])];
}
function grid(value: unknown): Grid | undefined {
  if (value === undefined) return undefined;
  const fields = object(value);
  return { radius: number(fields.radius), down: number(fields.down), up: number(fields.up) };
}

/** Validate JSON at the process boundary; movement checks its own action limits. */
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
        grid: grid(request.grid),
      };
    case "parkour_step":
      return {
        id,
        op: "parkour_step",
        action: request.action,
        ticks: number(request.ticks),
        grid: grid(request.grid),
      };
    case "parkour_observe":
      return { id, op: "parkour_observe", grid: grid(request.grid) };
    case "close":
    case "snapshot":
      return { id, op: request.op };
    case "blocks": {
      if (!Array.isArray(request.positions)) throw new Error("Expected block positions");
      return { id, op: "blocks", positions: request.positions.map(position) };
    }
    case "controls": {
      const keys: Record<string, boolean> = {};
      for (const [name, pressed] of Object.entries(object(request.keys))) {
        if (typeof pressed !== "boolean") throw new Error("Controls must be boolean");
        keys[name] = pressed;
      }
      return { id, op: "controls", keys };
    }
    case "look":
      return { id, op: "look", yaw: number(request.yaw), pitch: number(request.pitch) };
    case "command":
      return { id, op: "command", command: text(request.command) };
    default:
      throw new Error(`Unknown operation: ${String(request.op)}`);
  }
}
