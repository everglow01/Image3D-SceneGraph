import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { P0FrontLayer, p0PolygonContains, selectP0Surface, type P0SelectionRequest } from "../src/localGaussianP0Selection.ts";
import { p0FullMask, type P0Source } from "../src/localGaussianP0Source.ts";
import { createP0WorkerHandler, type P0WorkerReply } from "../src/localGaussianP0Worker.ts";

const sha256 = "b".repeat(64);
function source(rows: number[][]): P0Source { return { sha256, count: rows.length, geometry: new Float32Array(rows.flat()) }; }
function row(z: number, alpha: number, x = 0, y = 0, scale = 0.2) { return [x, y, z, scale, scale, scale, 0, 0, 0, 1, alpha]; }
function request(s: P0Source): P0SelectionRequest {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  return { sourceSha256: s.sha256, modelGeneration: 1, cameraGeneration: 1, sequence: 1,
    width: 65, height: 65, modelToView: new THREE.Matrix4().elements, projection: camera.projectionMatrix.elements,
    polygon: [[32, 32], [33, 32], [33, 33], [32, 33]], visible: p0FullMask(s.count), layerTolerance: 0.02 };
}

test("P0 front layers share the Python oracle fixtures including uncertain foreground", () => {
  const cases = JSON.parse(readFileSync(new URL("../../tests/fixtures/gaussian_front_layers.json", import.meta.url), "utf8"));
  for (const fixture of cases) {
    const layer = new P0FrontLayer(fixture.tolerance);
    for (const [id, depth, alpha] of fixture.contributions) layer.add(id, depth, alpha);
    assert.deepEqual(layer.result(), fixture.selected, fixture.name);
  }
});

test("P0 source-order IDs survive depth sorting and visibility changes", async () => {
  const s = source([row(-4, 0.95), row(-2, 0.95)]), r = request(s);
  const selected = await selectP0Surface(s, r);
  assert.deepEqual([...selected.selected], [2]);
  assert.equal(selected.selectedCount, 1);
  r.visible[0] = 1;
  assert.deepEqual([...(await selectP0Surface(s, r)).selected], [1]);
  r.visible[0] = 3;
  assert.deepEqual([...(await selectP0Surface(s, r)).selected], [2]);
});

test("P0 weak fog skips to a dominant layer but substantive transparency abstains", async () => {
  const weak = source([row(-1, 0.02), row(-3, 0.9)]);
  assert.deepEqual([...(await selectP0Surface(weak, request(weak))).selected], [2]);
  const uncertain = source([row(-1, 0.2), row(-3, 0.9)]);
  assert.equal((await selectP0Surface(uncertain, request(uncertain))).selectedCount, 0);
});

test("P0 clips centers like Spark and uses ellipse footprints, not center-only hits", async () => {
  const s = source([row(-0.005, 0.9), row(2, 0.9), row(-2, 0.95, 0.15, 0, 0.5)]);
  const result = await selectP0Surface(s, request(s));
  assert.deepEqual([...result.selected], [4]);
  assert.equal(result.candidates, 1);
  assert.equal(p0PolygonContains([[0, 0], [3, 0], [0, 3]], 0.5, 0.5), true);
  assert.equal(p0PolygonContains([[0, 0], [3, 0], [0, 3]], 2.5, 2.5), false);
});

test("P0 respects screen Y orientation and transformed cameras", async () => {
  const s = source([row(-2, 0.99, 0, 0.5, 0.1), row(-2, 0.99, 0, -0.5, 0.1)]), r = request(s);
  r.polygon = [[31, 17], [34, 17], [34, 20], [31, 20]];
  assert.deepEqual([...(await selectP0Surface(s, r)).selected], [1]);
  r.modelToView = new THREE.Matrix4().makeRotationZ(Math.PI).elements;
  assert.deepEqual([...(await selectP0Surface(s, r)).selected], [2]);
});

test("P0 fails closed for wrong sources, large ROI, unsupported cameras and cancellation", async () => {
  const s = source([row(-2, 0.9)]), r = request(s);
  await assert.rejects(selectP0Surface(s, { ...r, sourceSha256: "c".repeat(64) }), /身份/);
  await assert.rejects(selectP0Surface(s, { ...r, width: 256, height: 256,
    polygon: [[0, 0], [256, 0], [256, 256], [0, 256]] }), /预算/);
  await assert.rejects(selectP0Surface(s, { ...r, modelToView: new THREE.Matrix4().makeScale(2, 1, 1).elements }), /刚体/);
  await assert.rejects(selectP0Surface(s, r, () => true), /取消/);
  s.geometry[0] = NaN;
  await assert.rejects(selectP0Surface(s, r), /非有限/);
});

test("P0 uses rotated anisotropic covariance for the actual ellipse footprint", async () => {
  const s = source([[0, 0, -2, 0.6, 0.02, 0.02, 0, 0, 0, 1, 0.95]]), r = request(s);
  r.polygon = [[37, 32], [38, 32], [38, 33], [37, 33]];
  assert.equal((await selectP0Surface(s, r)).selectedCount, 1);
  s.geometry[8] = Math.SQRT1_2; s.geometry[9] = Math.SQRT1_2;
  assert.equal((await selectP0Surface(s, r)).selectedCount, 0);
});

test("P0 budget exhaustion rejects the whole request instead of returning a partial selection", async () => {
  const geometry = new Float32Array(100_001 * 11), one = row(-2, 0.95);
  for (let i = 0; i < 100_001; i++) geometry.set(one, i * 11);
  const s = { sha256, count: 100_001, geometry };
  await assert.rejects(selectP0Surface(s, request(s)), /预算超限|超时/);
});

test("P0 worker suppresses cancelled and old-source results and rejects sequence replay", async () => {
  const replies: P0WorkerReply[] = [], handle = createP0WorkerHandler(reply => replies.push(reply));
  const s = source([row(-2, 0.9)]), r = request(s);
  await handle({ type: "source", source: s, modelGeneration: 1 });
  const pending = handle({ type: "select", request: r });
  await handle({ type: "cancel" }); await pending;
  assert.equal(replies.filter(v => v.type === "result").length, 0);
  await handle({ type: "select", request: r });
  assert.equal(replies.at(-1)?.type, "error");
  const next = handle({ type: "select", request: { ...r, sequence: 2 } });
  await handle({ type: "source", source: { ...s, sha256: "d".repeat(64) }, modelGeneration: 2 });
  await next;
  assert.equal(replies.filter(v => v.type === "result").length, 0);
  await handle({ type: "select", request: { ...r, sequence: 3, modelGeneration: 2, sourceSha256: "d".repeat(64) } });
  assert.equal(replies.at(-1)?.type, "result");
});
