import assert from "node:assert/strict";
import test from "node:test";

import {
  clampPitchRadians,
  defaultWalkSettings,
  nearestPointOnPolygon,
  parseNavigationContract,
  parseWalkSettings,
  planarWalkVelocity,
  pointInNavigationBoundary,
  removeVelocityIntoNormal,
  transformNavigationContract
} from "../src/walkNavigation.ts";
import type { Mat3 } from "../src/gaussianViewerMetadata.ts";

function contractValue() {
  return {
    schema_version: 1,
    status: "available",
    coordinate_frame: "normalized",
    world_units: "arbitrary",
    up: [0, 0, 1],
    floor: {
      origin: [0, 0, 0],
      basis_u: [1, 0, 0],
      basis_v: [0, 1, 0]
    },
    boundary: {
      coordinate_frame: "floor_uv",
      outer: [[0, 0], [4, 0], [4, 4], [0, 4]],
      holes: [[[1, 1], [2, 1], [2, 2], [1, 2]]]
    },
    spawn: {
      floor_position: [0.5, 0.5, 0],
      eye_position: [0.5, 0.5, 1],
      look_direction: [1, 0, 0],
      floor_uv: [0.5, 0.5]
    },
    estimated_eye_height: 1,
    player: {
      capsule_total_height: 1.12,
      capsule_radius: 0.18,
      max_step: 0.12,
      max_slope_degrees: 35
    },
    controls: {
      speed: { default: 0.8, minimum: 0.4, maximum: 1.2 },
      vertical_fov_degrees: { default: 70, minimum: 50, maximum: 90 },
      mouse_sensitivity_radians_per_pixel: { default: 0.002, minimum: 0.0005, maximum: 0.005 },
      pitch_degrees: { minimum: -85, maximum: 85 }
    },
    provenance: {
      validation_image_ids_used: [],
      test_image_ids_used: []
    },
    collision: {
      sha256: "a".repeat(64),
      bytes: 2048,
      triangles: 100
    }
  };
}

test("strict navigation parsing accepts Train-only arbitrary-unit contract", () => {
  const contract = parseNavigationContract(contractValue());
  assert.deepEqual(contract.spawn.floor_uv, [0.5, 0.5]);
  assert.deepEqual(defaultWalkSettings(contract), {
    speedHeightRatio: 0.8,
    fovDegrees: 70,
    sensitivity: 0.002
  });
});

test("navigation parser rejects held-out provenance and invalid spawn", () => {
  const heldOut = contractValue();
  heldOut.provenance.test_image_ids_used = ["test-1"];
  assert.throws(() => parseNavigationContract(heldOut), /Train-only/);

  const outside = contractValue();
  outside.spawn.floor_uv = [9, 9];
  assert.throws(() => parseNavigationContract(outside), /outside/);
});

test("concave boundary includes edges and excludes holes", () => {
  const boundary = parseNavigationContract(contractValue()).boundary;
  assert.equal(pointInNavigationBoundary([0, 2], boundary), true);
  assert.equal(pointInNavigationBoundary([3, 3], boundary), true);
  assert.equal(pointInNavigationBoundary([1.5, 1.5], boundary), false);
  assert.equal(pointInNavigationBoundary([5, 3], boundary), false);
  assert.deepEqual(nearestPointOnPolygon([5, 3], boundary.outer), [4, 3]);
});

test("walk settings reject corrupted storage and preserve valid values", () => {
  const contract = parseNavigationContract(contractValue());
  assert.deepEqual(parseWalkSettings("not-json", contract), defaultWalkSettings(contract));
  assert.deepEqual(
    parseWalkSettings(JSON.stringify({ speedHeightRatio: 1, fovDegrees: 80, sensitivity: 0.003 }), contract),
    { speedHeightRatio: 1, fovDegrees: 80, sensitivity: 0.003 }
  );
  assert.deepEqual(
    parseWalkSettings(JSON.stringify({ speedHeightRatio: 99, fovDegrees: 80, sensitivity: 0.003 }), contract),
    defaultWalkSettings(contract)
  );
});

test("fixed-step velocity is normalized, planar, and slides along walls", () => {
  const velocity = planarWalkVelocity([1, 0, 0], [0, 0, 1], 1, 1, 2);
  assert.ok(Math.abs(Math.hypot(...velocity) - 2) < 1e-12);
  assert.ok(Math.abs(velocity[0] - Math.SQRT2) < 1e-12);
  assert.ok(Math.abs(velocity[1] + Math.SQRT2) < 1e-12);
  assert.equal(velocity[2], 0);
  assert.deepEqual(removeVelocityIntoNormal([-2, 1, 0], [1, 0, 0]), [0, 1, 0]);
  assert.equal(clampPitchRadians(Math.PI, -85, 85), (85 * Math.PI) / 180);
});

test("transformNavigationContract rotates world vectors and preserves the boundary", () => {
  const contract = parseNavigationContract(contractValue());
  const rotation: Mat3 = [
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
  ];
  const transformed = transformNavigationContract(contract, rotation);
  assert.deepEqual(transformed.up, [1, 0, 0]);
  assert.deepEqual(transformed.floor.basis_u, [0, 0, -1]);
  assert.deepEqual(transformed.spawn.floor_position, [0, 0.5, -0.5]);
  assert.deepEqual(transformed.spawn.look_direction, [0, 0, -1]);
  assert.deepEqual(transformed.boundary, contract.boundary);
  assert.deepEqual(transformed.player, contract.player);
  assert.deepEqual(transformed.collision, contract.collision);
});
