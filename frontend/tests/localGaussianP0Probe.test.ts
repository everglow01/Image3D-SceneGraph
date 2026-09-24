import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import { P0_PROBE_YAWS } from "../src/localGaussianP0Probe.ts";
import { p0IdentityFixture, p0FullMask } from "../src/localGaussianP0Source.ts";
import { P0_OCCLUSION_CASES, p0OcclusionPly } from "../src/localGaussianP0OcclusionProbe.ts";
import { selectP0Surface } from "../src/localGaussianP0Selection.ts";

test("P0 GPU occlusion fixtures have fixed source-ID expectations and SH3 rows", async () => {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  for (const fixture of P0_OCCLUSION_CASES) {
    const source = { sha256: "a".repeat(64), count: fixture.rows.length, geometry: new Float32Array(fixture.rows.flat()) };
    const [x, y] = fixture.pixel ?? [32, 32];
    const result = await selectP0Surface(source, { sourceSha256: source.sha256, modelGeneration: 1, cameraGeneration: 1, sequence: 1,
      width: 65, height: 65, modelToView: new THREE.Matrix4().makeRotationZ(fixture.rotation ?? 0).elements,
      projection: camera.projectionMatrix.elements, polygon: [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]],
      visible: p0FullMask(source.count), layerTolerance: 0.02 });
    assert.deepEqual(fixture.rows.map((_, id) => id).filter(id => result.selected[id >> 3] & (1 << (id & 7))), fixture.selected, fixture.name);
    const bytes = p0OcclusionPly(fixture.rows), header = new TextDecoder().decode(bytes.subarray(0, bytes.length - fixture.rows.length * 248));
    assert.match(header, /property float f_rest_44/);
    assert.match(header, /end_header\n$/);
  }
});

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
