import assert from "node:assert/strict";
import test from "node:test";
import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { acceptsFrame, captureView, imagePoint, nativeCamera, rectangle, referenceView, renderSize, restoreView } from "../src/cloudGaussianEditor.ts";

function project(point: Vector3, pose: number[][], k: number[][]): number[] {
  const p = [point.x, point.y, point.z, 1];
  const xyz = pose.slice(0, 3).map(row => row.reduce((sum, v, i) => sum + v * p[i], 0));
  return [k[0][0] * xyz[0] / xyz[2] + k[0][2], k[1][1] * xyz[1] / xyz[2] + k[1][2], xyz[2]];
}

test("native +Z/y-down agrees with Three projection including upright rotation", () => {
  const camera = new PerspectiveCamera(55, 16 / 9, .01, 1000);
  camera.up.set(0, 0, 1); camera.position.set(4, 3, 2); camera.lookAt(0, 0, 0);
  const upright = new Matrix4().makeRotationZ(0.7);
  const value = nativeCamera(camera, upright, 1920, 1080);
  for (const point of [new Vector3(), new Vector3(.5, .2, .7), new Vector3(-1, -.3, .4)]) {
    const browser = point.clone().applyMatrix4(upright).project(camera);
    const native = project(point, value.camera_from_normalized, value.intrinsic);
    assert.ok(Math.abs(native[0] - (browser.x + 1) * 960) < 1e-7);
    assert.ok(Math.abs(native[1] - (1 - browser.y) * 540) < 1e-7);
    assert.ok(native[2] > 0);
  }
});

test("letterbox mapping uses CSS pixels independently of DPR", () => {
  const box = { left: 10, top: 20, width: 1000, height: 1000 };
  assert.equal(imagePoint(510, 21, box, 1920, 1080), null);
  assert.deepEqual(imagePoint(510, 520, box, 1920, 1080), [960, 540]);
  const resized = { left: 10, top: 20, width: 500, height: 500 };
  assert.deepEqual(imagePoint(260, 270, resized, 1920, 1080), [960, 540]);
  assert.deepEqual(rectangle([3, 5], [1, 2]), [[3, 5], [1, 5], [1, 2], [3, 2]]);
});

test("Z-up orbit horizontal drag preserves height when controls are created after axis setup", () => {
  const camera = new PerspectiveCamera(); camera.up.set(0, 0, 1); camera.position.set(3, 4, 2);
  const controls = new OrbitControls(camera, null);
  controls.target.set(0, 0, 0); controls.update();
  const height = camera.position.z;
  controls.rotateLeft(.15); controls.update();
  assert.ok(Math.abs(camera.position.z - height) < 1e-12);
});

test("reference camera uses published pose and saved view restores only its own coordinates", () => {
  const camera = new PerspectiveCamera(55); camera.up.set(0, 0, 1);
  const path = { keyframes: [{ center_normalized: [1, 2, 3] as [number, number, number], world_from_camera: [
    [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]
  ] as [[number, number, number, number], [number, number, number, number], [number, number, number, number], [number, number, number, number]] }] };
  const target = referenceView(camera, path, [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
  ], new Matrix4(), new Vector3(), 1);
  assert.deepEqual(camera.position.toArray(), [1, 2, 3]);
  assert.ok(target.z > 3);
  const saved = captureView("model-1", camera, target);
  camera.position.set(0, 0, 0);
  assert.deepEqual(restoreView(camera, saved).toArray(), target.toArray());
  assert.deepEqual(camera.position.toArray(), [1, 2, 3]);
  const rotated = referenceView(camera, path, [
    [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]
  ], new Matrix4(), new Vector3(), 1);
  assert.ok(rotated.x < camera.position.x, "world +Z points along normalized -X after rotation");
});

test("zoomed and panned fixed frame maps back to original pixels after CSS resize", () => {
  const box = { left: 10, top: 20, width: 400, height: 300 };
  assert.deepEqual(imagePoint(230, 190, box, 800, 600, 2, [20, 20]), [400, 300]);
  const resized = { left: 20, top: 40, width: 800, height: 600 };
  assert.deepEqual(imagePoint(440, 360, resized, 800, 600, 2, [20, 20]), [400, 300]);
});


test("render sizes respect edge and total pixel limits; stale generations cannot display", () => {
  for (const [w, h] of [[1920, 1080], [400, 800], [500, 500], [4096, 2160]]) {
    const [width, height] = renderSize(w, h);
    assert.ok(width <= 1920 && height <= 1920 && width * height <= 1920 * 1080);
    assert.ok(Math.abs(width / height - w / h) < .002);
  }
  assert.equal(acceptsFrame(3, 2, 1, 1), false);
  assert.equal(acceptsFrame(3, 3, 2, 1), false);
  assert.equal(acceptsFrame(3, 3, 2, 2), true);
});
