import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveGaussianViewerFrame,
  deriveUprightRotation,
  parseContentLength,
  parseGaussianCameraPath,
  parseGaussianExportMetadata,
  signedUprightAxis,
  viewerAlphaThreshold,
  type Mat4
} from "../src/gaussianViewerMetadata.ts";

test("missing Content-Length remains unknown", () => {
  assert.equal(parseContentLength(null), null);
  assert.equal(parseContentLength(""), null);
  assert.equal(parseContentLength("invalid"), null);
  assert.equal(parseContentLength("247352"), 247352);
});

test("export metadata controls SH and alpha threshold", () => {
  const metadata = parseGaussianExportMetadata({
    sh_degree: 3,
    viewer_minimum_opacity: 0.005,
    scene_center: [0.1, -0.2, 0.3],
    scene_radius_p95: 1.25,
    world_from_normalized: [
      [2, 0, 0, 1],
      [0, 2, 0, 2],
      [0, 0, 2, 3],
      [0, 0, 0, 1]
    ]
  });

  assert.deepEqual(metadata, {
    sh_degree: 3,
    viewer_minimum_opacity: 0.005,
    scene_center: [0.1, -0.2, 0.3],
    scene_radius_p95: 1.25,
    world_from_normalized: [
      [2, 0, 0, 1],
      [0, 2, 0, 2],
      [0, 0, 2, 3],
      [0, 0, 0, 1]
    ]
  });
  assert.equal(viewerAlphaThreshold(metadata.viewer_minimum_opacity), 1);
});

test("legacy export metadata falls back to bounding-box framing", () => {
  assert.deepEqual(
    parseGaussianExportMetadata({ sh_degree: 2, viewer_minimum_opacity: 0.01 }),
    {
      sh_degree: 2,
      viewer_minimum_opacity: 0.01,
      scene_center: null,
      scene_radius_p95: null,
      world_from_normalized: null
    }
  );
});

const identity: Mat4 = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
];

function metadata(worldFromNormalized: Mat4 = identity) {
  return parseGaussianExportMetadata({
    sh_degree: 3,
    viewer_minimum_opacity: 0.005,
    world_from_normalized: worldFromNormalized
  });
}

function keyframe(center: [number, number, number], imageUp: [number, number, number]) {
  const worldFromCamera: Mat4 = identity.map((row) => [...row]) as Mat4;
  worldFromCamera[0][1] = -imageUp[0];
  worldFromCamera[1][1] = -imageUp[1];
  worldFromCamera[2][1] = -imageUp[2];
  return { center_normalized: center, world_from_camera: worldFromCamera };
}

test("camera path centers orbit on poses and aligns tilted room up", () => {
  const path = parseGaussianCameraPath({
    schema_version: 1,
    coordinate_frame: "normalized",
    keyframes: [
      keyframe([1, 2, 3], [0, Math.SQRT1_2, Math.SQRT1_2]),
      keyframe([3, 4, 5], [0, Math.SQRT1_2, Math.SQRT1_2])
    ]
  });

  const frame = deriveGaussianViewerFrame(metadata(), path);

  assert.deepEqual(frame?.center, [2, 3, 4]);
  assert.ok(frame);
  assert.ok(Math.abs(frame.up[0]) < 1e-12);
  assert.ok(Math.abs(frame.up[1] - Math.SQRT1_2) < 1e-12);
  assert.ok(Math.abs(frame.up[2] - Math.SQRT1_2) < 1e-12);
});

test("camera up honors rotated normalization and corrects inversion", () => {
  const worldFromNormalized: Mat4 = [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
  ];
  const path = parseGaussianCameraPath({
    schema_version: 1,
    coordinate_frame: "normalized",
    keyframes: [keyframe([0, 0, 0], [0, 1, 0]), keyframe([0, 0, 1], [0, 1, 0])]
  });

  assert.deepEqual(deriveGaussianViewerFrame(metadata(worldFromNormalized), path), {
    center: [0, 0, 0.5],
    up: [0, 0, 1]
  });
});

test("unusable camera orientation falls back without failing the asset", () => {
  const path = parseGaussianCameraPath({
    schema_version: 1,
    coordinate_frame: "normalized",
    keyframes: [keyframe([0, 0, 0], [0, 1, 0]), keyframe([1, 0, 0], [0, -1, 0])]
  });

  assert.equal(deriveGaussianViewerFrame(metadata(), path), null);
  assert.throws(
    () => parseGaussianCameraPath({ schema_version: 1, coordinate_frame: "normalized", keyframes: [] }),
    /at least two/
  );
});

test("deriveUprightRotation composes alignment over the export transform", () => {
  const identity: Mat4 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
  const meta = (world: Mat4 | null) => ({
    sh_degree: 0,
    viewer_minimum_opacity: 0.005,
    scene_center: null,
    scene_radius_p95: null,
    world_from_normalized: world
  });
  const alignment = {
    transform: [
      [0, 0, 1, 0],
      [0, 1, 0, 0],
      [-1, 0, 0, 0],
      [0, 0, 0, 1]
    ]
  };
  const expected = [
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
  ];
  assert.deepEqual(deriveUprightRotation(meta(identity), alignment), expected);
  const scaled = identity.map((row, rowIndex) =>
    row.map((value, columnIndex) => (rowIndex < 3 && columnIndex < 3 ? value * 7 : value))
  ) as Mat4;
  assert.deepEqual(deriveUprightRotation(meta(scaled), alignment), expected);
  assert.equal(deriveUprightRotation(meta(identity), {}), null);
  assert.equal(deriveUprightRotation(meta(null), alignment), null);
  assert.deepEqual(signedUprightAxis([0.19, 0.03, -0.98]), [0, 0, -1]);
  assert.deepEqual(signedUprightAxis([0.01, -0.02, 0.99]), [0, 0, 1]);
});
