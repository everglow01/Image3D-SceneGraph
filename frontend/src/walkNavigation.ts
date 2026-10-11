import { applyMat3, type Mat3 } from "./gaussianViewerMetadata.ts";

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

type NumericRange = {
  default: number;
  minimum: number;
  maximum: number;
};

export type NavigationContract = {
  up: Vec3;
  floor: {
    origin: Vec3;
    basis_u: Vec3;
    basis_v: Vec3;
  };
  boundary: {
    outer: Vec2[];
    holes: Vec2[][];
  };
  spawn: {
    floor_position: Vec3;
    eye_position: Vec3;
    look_direction: Vec3;
    floor_uv: Vec2;
  };
  estimated_eye_height: number;
  player: {
    capsule_total_height: number;
    capsule_radius: number;
    max_step: number;
    max_slope_degrees: number;
  };
  controls: {
    speed: NumericRange;
    vertical_fov_degrees: NumericRange;
    mouse_sensitivity_radians_per_pixel: NumericRange;
    pitch_degrees: { minimum: number; maximum: number };
  };
  collision: {
    sha256: string;
    bytes: number;
    triangles: number;
  };
};

export type WalkSettings = {
  speedHeightRatio: number;
  fovDegrees: number;
  sensitivity: number;
};

const MAX_COLLISION_BYTES = 10 * 1024 * 1024;
const MAX_COLLISION_TRIANGLES = 50_000;
const EDGE_EPSILON = 1e-9;

export async function collisionSha256(buffer: ArrayBuffer): Promise<string> {
  if (buffer.byteLength > MAX_COLLISION_BYTES) throw new Error("Collision mesh exceeds the browser budget");
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
  }
  // Web Crypto is unavailable on plain HTTP intranets; retain SHA-256 verification there.
  const constants = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
  ]);
  const state = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const padded = new Uint8Array(Math.ceil((buffer.byteLength + 9) / 64) * 64);
  padded.set(new Uint8Array(buffer)); padded[buffer.byteLength] = 0x80;
  const data = new DataView(padded.buffer);
  data.setUint32(padded.length - 4, buffer.byteLength * 8);
  const words = new Uint32Array(64);
  const rotate = (value: number, bits: number) => (value >>> bits) | (value << (32 - bits));
  for (let offset = 0; offset < padded.length; offset += 64) {
    for (let i = 0; i < 16; i++) words[i] = data.getUint32(offset + i * 4);
    for (let i = 16; i < 64; i++) {
      const x = words[i - 15], y = words[i - 2];
      words[i] = words[i - 16] + (rotate(x, 7) ^ rotate(x, 18) ^ (x >>> 3)) + words[i - 7] + (rotate(y, 17) ^ rotate(y, 19) ^ (y >>> 10));
    }
    let [a, b, c, d, e, f, g, h] = state;
    for (let i = 0; i < 64; i++) {
      const first = (h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + constants[i] + words[i]) >>> 0;
      const second = ((rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h = g; g = f; f = e; e = (d + first) >>> 0; d = c; c = b; b = a; a = (first + second) >>> 0;
    }
    [a, b, c, d, e, f, g, h].forEach((value, i) => { state[i] += value; });
  }
  return Array.from(state, word => word.toString(16).padStart(8, "0")).join("");
}

export function parseNavigationContract(value: unknown): NavigationContract {
  const record = objectRecord(value, "Navigation contract");
  if (
    record.schema_version !== 1 ||
    record.status !== "available" ||
    record.coordinate_frame !== "normalized" ||
    record.world_units !== "arbitrary"
  ) {
    throw new Error("Navigation contract schema is invalid");
  }

  const floor = objectRecord(record.floor, "Navigation floor");
  const boundary = objectRecord(record.boundary, "Navigation boundary");
  const spawn = objectRecord(record.spawn, "Navigation spawn");
  const player = objectRecord(record.player, "Navigation player");
  const controls = objectRecord(record.controls, "Navigation controls");
  const collision = objectRecord(record.collision, "Navigation collision");
  const provenance = objectRecord(record.provenance, "Navigation provenance");
  if (
    boundary.coordinate_frame !== "floor_uv" ||
    !Array.isArray(provenance.validation_image_ids_used) ||
    provenance.validation_image_ids_used.length !== 0 ||
    !Array.isArray(provenance.test_image_ids_used) ||
    provenance.test_image_ids_used.length !== 0
  ) {
    throw new Error("Navigation contract is not Train-only normalized data");
  }

  const up = vec3(record.up, "Navigation up");
  const origin = vec3(floor.origin, "Navigation floor origin");
  const basisU = vec3(floor.basis_u, "Navigation floor basis U");
  const basisV = vec3(floor.basis_v, "Navigation floor basis V");
  requireUnitVector(up, "Navigation up");
  requireUnitVector(basisU, "Navigation floor basis U");
  requireUnitVector(basisV, "Navigation floor basis V");
  if (Math.abs(dot3(up, basisU)) > 1e-3 || Math.abs(dot3(up, basisV)) > 1e-3 || Math.abs(dot3(basisU, basisV)) > 1e-3) {
    throw new Error("Navigation floor basis is not orthogonal");
  }

  const outer = polygon(boundary.outer, "Navigation outer boundary");
  if (!Array.isArray(boundary.holes)) {
    throw new Error("Navigation boundary holes must be an array");
  }
  const holes = boundary.holes.map((hole, index) => polygon(hole, `Navigation boundary hole ${index}`));
  const eyeHeight = positiveNumber(record.estimated_eye_height, "Navigation eye height");
  const radius = positiveNumber(player.capsule_radius, "Navigation capsule radius");
  const totalHeight = positiveNumber(player.capsule_total_height, "Navigation capsule height");
  const maxStep = positiveNumber(player.max_step, "Navigation max step");
  const maxSlope = finiteNumber(player.max_slope_degrees, "Navigation max slope");
  if (totalHeight <= radius * 2 || maxStep >= eyeHeight || maxSlope <= 0 || maxSlope > 35) {
    throw new Error("Navigation player dimensions are invalid");
  }

  const speed = numericRange(controls.speed, "Navigation speed", true);
  const fov = numericRange(controls.vertical_fov_degrees, "Navigation FOV", true);
  const sensitivity = numericRange(
    controls.mouse_sensitivity_radians_per_pixel,
    "Navigation mouse sensitivity",
    true
  );
  const pitch = objectRecord(controls.pitch_degrees, "Navigation pitch");
  const pitchMinimum = finiteNumber(pitch.minimum, "Navigation minimum pitch");
  const pitchMaximum = finiteNumber(pitch.maximum, "Navigation maximum pitch");
  if (
    speed.minimum < 0.4 * eyeHeight - EDGE_EPSILON ||
    speed.maximum > 1.2 * eyeHeight + EDGE_EPSILON ||
    fov.minimum < 50 ||
    fov.maximum > 90 ||
    sensitivity.minimum < 0.0005 ||
    sensitivity.maximum > 0.005 ||
    pitchMinimum < -85 ||
    pitchMaximum > 85 ||
    pitchMinimum >= pitchMaximum
  ) {
    throw new Error("Navigation control ranges are invalid");
  }

  const collisionBytes = positiveInteger(collision.bytes, "Navigation collision bytes");
  const collisionTriangles = positiveInteger(collision.triangles, "Navigation collision triangles");
  if (collisionBytes > MAX_COLLISION_BYTES || collisionTriangles > MAX_COLLISION_TRIANGLES) {
    throw new Error("Navigation collision exceeds the browser budget");
  }
  if (typeof collision.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(collision.sha256)) {
    throw new Error("Navigation collision hash is invalid");
  }

  const result: NavigationContract = {
    up,
    floor: { origin, basis_u: basisU, basis_v: basisV },
    boundary: { outer, holes },
    spawn: {
      floor_position: vec3(spawn.floor_position, "Navigation spawn floor position"),
      eye_position: vec3(spawn.eye_position, "Navigation spawn eye position"),
      look_direction: vec3(spawn.look_direction, "Navigation spawn look direction"),
      floor_uv: vec2(spawn.floor_uv, "Navigation spawn floor UV")
    },
    estimated_eye_height: eyeHeight,
    player: {
      capsule_total_height: totalHeight,
      capsule_radius: radius,
      max_step: maxStep,
      max_slope_degrees: maxSlope
    },
    controls: {
      speed,
      vertical_fov_degrees: fov,
      mouse_sensitivity_radians_per_pixel: sensitivity,
      pitch_degrees: { minimum: pitchMinimum, maximum: pitchMaximum }
    },
    collision: {
      sha256: collision.sha256,
      bytes: collisionBytes,
      triangles: collisionTriangles
    }
  };

  if (!pointInNavigationBoundary(result.spawn.floor_uv, result.boundary)) {
    throw new Error("Navigation spawn lies outside the boundary");
  }
  return result;
}

export function defaultWalkSettings(contract: NavigationContract): WalkSettings {
  return {
    speedHeightRatio: contract.controls.speed.default / contract.estimated_eye_height,
    fovDegrees: contract.controls.vertical_fov_degrees.default,
    sensitivity: contract.controls.mouse_sensitivity_radians_per_pixel.default
  };
}

export function parseWalkSettings(value: string | null, contract: NavigationContract): WalkSettings {
  const defaults = defaultWalkSettings(contract);
  if (!value) {
    return defaults;
  }
  try {
    const record = objectRecord(JSON.parse(value), "Walk settings");
    const speedHeightRatio = finiteNumber(record.speedHeightRatio, "Walk speed");
    const fovDegrees = finiteNumber(record.fovDegrees, "Walk FOV");
    const sensitivity = finiteNumber(record.sensitivity, "Walk sensitivity");
    const minimumSpeed = contract.controls.speed.minimum / contract.estimated_eye_height;
    const maximumSpeed = contract.controls.speed.maximum / contract.estimated_eye_height;
    if (
      speedHeightRatio < minimumSpeed ||
      speedHeightRatio > maximumSpeed ||
      fovDegrees < contract.controls.vertical_fov_degrees.minimum ||
      fovDegrees > contract.controls.vertical_fov_degrees.maximum ||
      sensitivity < contract.controls.mouse_sensitivity_radians_per_pixel.minimum ||
      sensitivity > contract.controls.mouse_sensitivity_radians_per_pixel.maximum
    ) {
      return defaults;
    }
    return { speedHeightRatio, fovDegrees, sensitivity };
  } catch {
    return defaults;
  }
}

export function pointInNavigationBoundary(
  point: Vec2,
  boundary: Pick<NavigationContract["boundary"], "outer" | "holes">
): boolean {
  return pointInPolygon(point, boundary.outer) && !boundary.holes.some((hole) => pointInPolygonStrict(point, hole));
}

export function nearestPointOnPolygon(point: Vec2, polygon: Vec2[]): Vec2 {
  let best: Vec2 = polygon[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < polygon.length; index += 1) {
    const start = polygon[index];
    const end = polygon[(index + 1) % polygon.length];
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const denominator = dx * dx + dy * dy;
    const amount = denominator === 0 ? 0 : Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator));
    const candidate: Vec2 = [start[0] + dx * amount, start[1] + dy * amount];
    const distance = (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2;
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return best;
}

export function planarWalkVelocity(
  forward: Vec3,
  up: Vec3,
  forwardInput: number,
  strafeInput: number,
  speed: number
): Vec3 {
  const forwardDotUp = dot3(forward, up);
  const planarForward: Vec3 = [
    forward[0] - up[0] * forwardDotUp,
    forward[1] - up[1] * forwardDotUp,
    forward[2] - up[2] * forwardDotUp
  ];
  const length = Math.hypot(...planarForward);
  if (length < EDGE_EPSILON || speed <= 0 || !Number.isFinite(speed)) {
    return [0, 0, 0];
  }
  const normalized: Vec3 = planarForward.map((value) => value / length) as Vec3;
  const right: Vec3 = [
    normalized[1] * up[2] - normalized[2] * up[1],
    normalized[2] * up[0] - normalized[0] * up[2],
    normalized[0] * up[1] - normalized[1] * up[0]
  ];
  const inputLength = Math.hypot(forwardInput, strafeInput);
  if (inputLength < EDGE_EPSILON) {
    return [0, 0, 0];
  }
  const scale = speed / Math.max(1, inputLength);
  return [
    (normalized[0] * forwardInput + right[0] * strafeInput) * scale,
    (normalized[1] * forwardInput + right[1] * strafeInput) * scale,
    (normalized[2] * forwardInput + right[2] * strafeInput) * scale
  ];
}

export function clampPitchRadians(value: number, minimumDegrees: number, maximumDegrees: number): number {
  const minimum = (minimumDegrees * Math.PI) / 180;
  const maximum = (maximumDegrees * Math.PI) / 180;
  return Math.max(minimum, Math.min(maximum, value));
}

export function removeVelocityIntoNormal(velocity: Vec3, normal: Vec3): Vec3 {
  const inward = dot3(velocity, normal);
  return inward < 0
    ? [velocity[0] - normal[0] * inward, velocity[1] - normal[1] * inward, velocity[2] - normal[2] * inward]
    : [...velocity];
}

function pointInPolygon(point: Vec2, polygon: Vec2[]): boolean {
  if (pointOnPolygon(point, polygon)) {
    return true;
  }
  return pointInPolygonStrict(point, polygon);
}

function pointInPolygonStrict(point: Vec2, polygon: Vec2[]): boolean {
  let inside = false;
  let previous = polygon[polygon.length - 1];
  for (const current of polygon) {
    if ((current[1] > point[1]) !== (previous[1] > point[1])) {
      const crossing = ((previous[0] - current[0]) * (point[1] - current[1])) / (previous[1] - current[1]) + current[0];
      if (point[0] < crossing) {
        inside = !inside;
      }
    }
    previous = current;
  }
  return inside;
}

function pointOnPolygon(point: Vec2, polygon: Vec2[]): boolean {
  return polygon.some((start, index) => {
    const end = polygon[(index + 1) % polygon.length];
    const cross = (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (end[0] - start[0]);
    if (Math.abs(cross) > EDGE_EPSILON) {
      return false;
    }
    return (
      point[0] >= Math.min(start[0], end[0]) - EDGE_EPSILON &&
      point[0] <= Math.max(start[0], end[0]) + EDGE_EPSILON &&
      point[1] >= Math.min(start[1], end[1]) - EDGE_EPSILON &&
      point[1] <= Math.max(start[1], end[1]) + EDGE_EPSILON
    );
  });
}

function polygon(value: unknown, label: string): Vec2[] {
  if (!Array.isArray(value) || value.length < 3) {
    throw new Error(`${label} requires at least three points`);
  }
  const result = value.map((point, index) => vec2(point, `${label} point ${index}`));
  let area = 0;
  for (let index = 0; index < result.length; index += 1) {
    const current = result[index];
    const next = result[(index + 1) % result.length];
    area += current[0] * next[1] - current[1] * next[0];
  }
  if (Math.abs(area) < EDGE_EPSILON) {
    throw new Error(`${label} has no area`);
  }
  return result;
}

function numericRange(value: unknown, label: string, positive: boolean): NumericRange {
  const range = objectRecord(value, label);
  const minimum = finiteNumber(range.minimum, `${label} minimum`);
  const maximum = finiteNumber(range.maximum, `${label} maximum`);
  const defaultValue = finiteNumber(range.default, `${label} default`);
  if ((positive && minimum <= 0) || minimum > defaultValue || defaultValue > maximum) {
    throw new Error(`${label} range is invalid`);
  }
  return { minimum, maximum, default: defaultValue };
}

function vec2(value: unknown, label: string): Vec2 {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error(`${label} must be a finite 2D vector`);
  }
  return [finiteNumber(value[0], label), finiteNumber(value[1], label)];
}

function vec3(value: unknown, label: string): Vec3 {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${label} must be a finite 3D vector`);
  }
  return [finiteNumber(value[0], label), finiteNumber(value[1], label), finiteNumber(value[2], label)];
}

function requireUnitVector(value: Vec3, label: string) {
  const length = Math.hypot(...value);
  if (Math.abs(length - 1) > 1e-3) {
    throw new Error(`${label} must have unit length`);
  }
}

function dot3(left: Vec3, right: Vec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
}

function positiveNumber(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (result <= 0) {
    throw new Error(`${label} must be positive`);
  }
  return result;
}

function positiveInteger(value: unknown, label: string): number {
  const result = positiveNumber(value, label);
  if (!Number.isInteger(result)) {
    throw new Error(`${label} must be an integer`);
  }
  return result;
}

function objectRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

// Rotate every world-space vector of the contract by the viewer's display-only
// upright rotation so walk physics stays registered with the rotated splats.
// Boundary coordinates live in the floor uv plane and are untouched.
export function transformNavigationContract(
  contract: NavigationContract,
  rotation: Mat3
): NavigationContract {
  return {
    ...contract,
    up: applyMat3(rotation, contract.up),
    floor: {
      origin: applyMat3(rotation, contract.floor.origin),
      basis_u: applyMat3(rotation, contract.floor.basis_u),
      basis_v: applyMat3(rotation, contract.floor.basis_v)
    },
    spawn: {
      ...contract.spawn,
      floor_position: applyMat3(rotation, contract.spawn.floor_position),
      eye_position: applyMat3(rotation, contract.spawn.eye_position),
      look_direction: applyMat3(rotation, contract.spawn.look_direction)
    }
  };
}
