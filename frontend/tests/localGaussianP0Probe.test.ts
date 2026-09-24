import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import { P0_PROBE_YAWS } from "../src/localGaussianP0Probe.ts";
import { p0IdentityFixture } from "../src/localGaussianP0Source.ts";

test("P0 synthetic camera poses keep the target ROI inside the actual viewport", () => {
  const fixture = p0IdentityFixture();
  for (const [view, angle] of P0_PROBE_YAWS.entries()) {
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    camera.rotation.y = angle; camera.updateMatrixWorld(true);
    const target = new THREE.Vector3().fromArray(fixture.expected, 12 * 11)
      .applyMatrix4(new THREE.Matrix4().makeRotationZ(view * 0.27)).project(camera);
    const x = 128 * (1 + target.x), y = 128 * (1 - target.y);
    assert.ok(x >= 2 && x <= 254 && y >= 2 && y <= 254, `view ${view}: (${x}, ${y})`);
    assert.ok(target.z > -1 && target.z < 1);
  }
});
