import assert from "node:assert/strict";
import test from "node:test";

import {
  applyAxisSigns,
  cameraLinePositions,
  parseAlignmentTransform,
  parseCameraFrames,
  transformCameraFrames
} from "../src/cameraOverlay.ts";

const COLMAP_CAMERA = {
  coordinate_system: "colmap_world",
  cameras: [
    {
      camera_id: 1,
      model: "PINHOLE",
      width: 2,
      height: 2,
      params: [2, 2, 1, 1]
    }
  ],
  images: [
    {
      image_id: 1,
      camera_id: 1,
      name: "a.jpg",
      qvec: [1, 0, 0, 0],
      tvec: [-1, -2, -3]
    }
  ]
};

test("COLMAP poses produce world centers and frusta", () => {
  const [frame] = parseCameraFrames(COLMAP_CAMERA, 2);

  assert.deepEqual(frame.center, [1, 2, 3]);
  assert.deepEqual(frame.corners[0], [0, 1, 5]);
  assert.deepEqual(frame.corners[2], [2, 3, 5]);
});

test("COLMAP rotation maps camera forward into world", () => {
  const payload = structuredClone(COLMAP_CAMERA);
  payload.images[0].qvec = [Math.SQRT1_2, 0, Math.SQRT1_2, 0];
  payload.images[0].tvec = [0, 0, 0];

  const [frame] = parseCameraFrames(payload, 2);

  assert.ok(Math.abs(frame.corners[0][0] + 2) < 1e-12);
});

test("VGGT camera_to_world schema is supported", () => {
  const [frame] = parseCameraFrames(
    {
      coordinate_system: "opencv_camera_from_world",
      image_shape: { width: 2, height: 2 },
      cameras: [
        {
          intrinsic: [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
          camera_to_world: [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6]]
        }
      ]
    },
    2
  );

  assert.deepEqual(frame.center, [4, 5, 6]);
  assert.deepEqual(frame.corners[0], [3, 4, 8]);
});

test("alignment transforms every overlay vertex before centering", () => {
  const frames = parseCameraFrames(COLMAP_CAMERA, 2);
  const transform = parseAlignmentTransform({
    transform: [[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
  });

  const [aligned] = transformCameraFrames(frames, transform, [8, 19, 30]);
  const [raw] = transformCameraFrames(frames, null, [0, 0, 0]);

  assert.deepEqual(raw.center, [1, 2, 3]);
  assert.deepEqual(aligned.center, [0, 2, 3]);
  assert.deepEqual(aligned.corners[0], [1, 1, 5]);
  assert.deepEqual(applyAxisSigns(aligned.center, { x: 1, y: -1, z: -1 }), [0, -2, -3]);
});

test("line geometry includes every frustum edge and trajectory link", () => {
  const frame = parseCameraFrames(COLMAP_CAMERA, 2)[0];
  assert.equal(cameraLinePositions([frame, frame]).length, 102);
});
