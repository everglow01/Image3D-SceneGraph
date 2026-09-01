import assert from "node:assert/strict";
import test from "node:test";

import { parseCameraFrames } from "../src/cameraOverlay.ts";

const camera = {
  camera_id: 1,
  model: "PINHOLE",
  width: 100,
  height: 100,
  params: [100, 100, 50, 50]
};

function frames(qvec: number[], tvec: number[]) {
  return parseCameraFrames(
    {
      coordinate_system: "colmap_world",
      cameras: [camera],
      images: [{ camera_id: 1, qvec, tvec }]
    },
    1
  );
}

function forward(frame: ReturnType<typeof frames>[number]): [number, number, number] {
  const mean = frame.corners.reduce(
    (acc, corner) => [acc[0] + corner[0] / 4, acc[1] + corner[1] / 4, acc[2] + corner[2] / 4],
    [-frame.center[0], -frame.center[1], -frame.center[2]]
  );
  return mean as [number, number, number];
}

test("identity COLMAP camera looks down +z with unit frustum depth", () => {
  const [frame] = frames([1, 0, 0, 0], [0, 0, 0]);
  assert.ok(frame.corners.every((corner) => Math.abs(corner[2] - 1) < 1e-9));
  assert.deepEqual(forward(frame), [0, 0, 1]);
});

test("COLMAP frustum corners use the world-from-camera rotation, not its transpose", () => {
  const s = Math.SQRT1_2;
  const [frame] = frames([s, 0, s, 0], [0, 0, 0]);
  // 90 degrees about y maps the camera forward (0,0,1) to world +x.
  assert.ok(Math.abs(forward(frame)[0] - 1) < 1e-9);
  assert.ok(Math.abs(forward(frame)[1]) < 1e-9);
  assert.ok(Math.abs(forward(frame)[2]) < 1e-9);
});

test("COLMAP camera center is -R^T t", () => {
  const s = Math.SQRT1_2;
  const [frame] = frames([s, 0, s, 0], [0, 0, 1]);
  assert.ok(Math.abs(frame.center[0] - 1) < 1e-9);
  assert.ok(Math.abs(frame.center[1]) < 1e-9);
  assert.ok(Math.abs(frame.center[2]) < 1e-9);
});

test("unsupported camera coordinate system is rejected", () => {
  assert.throws(() => parseCameraFrames({ coordinate_system: "nope" }, 1), /Unsupported/);
});
