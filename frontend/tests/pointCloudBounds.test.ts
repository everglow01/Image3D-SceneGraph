import assert from "node:assert/strict";
import test from "node:test";

import { robustCloudBounds } from "../src/pointCloudBounds.ts";

test("robustCloudBounds ignores stray outliers", () => {
  const positions: number[] = [];
  for (let index = 0; index < 99; index += 1) {
    positions.push((index % 10) / 10, Math.floor(index / 10) / 10, 0.5);
  }
  positions.push(1000, -1000, 500);
  const bounds = robustCloudBounds(positions, positions.length / 3);
  assert.ok(Math.abs(bounds.center[0] - 0.45) < 0.11);
  assert.ok(Math.abs(bounds.center[1] - 0.45) < 0.11);
  assert.ok(bounds.radius < 1, `outlier must not inflate radius, got ${bounds.radius}`);
});

test("robustCloudBounds falls back for empty input", () => {
  assert.deepEqual(robustCloudBounds([], 0), { center: [0, 0, 0], radius: 1 });
});
