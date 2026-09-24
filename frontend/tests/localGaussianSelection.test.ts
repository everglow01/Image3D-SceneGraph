import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { selectLocalGaussians, type LocalSelectionRequest } from "../src/localGaussianSelection.ts";
import { LocalSelectionClient } from "../src/localGaussianSelectionClient.ts";
import { p0FullMask, p0Visible, type P0Source } from "../src/localGaussianP0Source.ts";
import { createP0WorkerHandler, type P0WorkerReply } from "../src/localGaussianP0Worker.ts";

const f = JSON.parse(readFileSync(new URL("../../tests/fixtures/gaussian_local_selection.json", import.meta.url), "utf8"));
const source: P0Source = { sha256: "b".repeat(64), count: f.points.length,
  geometry: new Float32Array(f.points.flatMap((p: number[]) => [...p, 0.2, 0.2, 0.2, 0, 0, 0, 1, 0.95])) };
function request(): LocalSelectionRequest {
  return { sourceSha256: source.sha256, modelGeneration: 1, cameraGeneration: 2, sequence: 1,
    width: f.width, height: f.height, modelToView: new THREE.Matrix4().makeScale(1, -1, -1)
      .multiply(new THREE.Matrix4().set(...f.camera_from_normalized.flat() as Parameters<THREE.Matrix4["set"]>)).elements,
    projection: new THREE.PerspectiveCamera(90, 1, 0.01, 1e6).projectionMatrix.elements,
    polygon: [[0, 0], [100, 0], [100, 100], [0, 100]], visible: p0FullMask(source.count), layerTolerance: 0.02, mode: "surface" };
}

test("深度、穿透、源轴盒与 Python 同组金样一致", async () => {
  for (const c of f.cases) {
    const result = await selectLocalGaussians(source, { ...request(), mode: c.mode, polygon: c.polygon ?? request().polygon,
      depthRange: c.depth, box: c.mode === "box" ? { min: c.min, max: c.max } : undefined });
    const ids = Array.from({ length: source.count }, (_, i) => i).filter(i => p0Visible(result.selected, i));
    assert.deepEqual(ids, c.selected, c.name);
  }
});

test("非表层选择遵守可见 mask、预算、取消及输入边界", async () => {
  const r = { ...request(), mode: "through" as const, visible: new Uint8Array([1]) };
  assert.equal((await selectLocalGaussians(source, r)).selectedCount, 1);
  await assert.rejects(selectLocalGaussians(source, r, () => true), /取消/);
  await assert.rejects(selectLocalGaussians(source, { ...r, mode: "depth", depthRange: [3, 1] }), /深度/);
  await assert.rejects(selectLocalGaussians(source, { ...r, mode: "box", box: { min: [0, 0, 0], max: [0, 1, 1] } }), /三维盒/);
  await assert.rejects(selectLocalGaussians(source, { ...r, sourceSha256: "c".repeat(64) }), /身份/);
});

test("P1 表层复用贡献语义，单像素只在可信前层返回深度", async () => {
  const r = { ...request(), width: 256, height: 256, polygon: [[127, 127], [128, 127], [128, 128], [127, 128]] as [number, number][] };
  assert.equal((await selectLocalGaussians(source, r)).surfaceDepth, 1);
  const uncertain = { ...source, geometry: source.geometry.slice() }; uncertain.geometry[5 * 11 + 10] = 0.2;
  assert.equal((await selectLocalGaussians(uncertain, r)).surfaceDepth, null);
  const behind = { sha256: source.sha256, count: 1, geometry: new Float32Array([0, 0, -2, 0.2, 0.2, 0.2, 0, 0, 0, 1, 0.95]) };
  assert.equal((await selectLocalGaussians(behind, { ...r, visible: new Uint8Array([1]),
    polygon: [[0, 0], [256, 0], [256, 256], [0, 256]] })).selectedCount, 0);
});

test("Worker 快速换工具或换源不会发送旧本地结果", async () => {
  const replies: P0WorkerReply[] = [], handle = createP0WorkerHandler(r => replies.push(r));
  await handle({ type: "source", source, modelGeneration: 1 });
  const pending = handle({ type: "select-local", request: { ...request(), mode: "through" } });
  await handle({ type: "cancel" }); await pending;
  assert.equal(replies.some(r => r.type === "result"), false);
  await handle({ type: "select-local", request: { ...request(), mode: "through", sequence: 2 } });
  assert.equal(replies.at(-1)?.type, "result");
});

test("客户端取消结算等待者并拒绝陈旧响应，释放终止 Worker", async () => {
  const messages: any[] = [];
  const worker = { onmessage: null as any, onerror: null as any, onmessageerror: null as any,
    postMessage: (m: any) => messages.push(m), terminate: () => messages.push("terminated") };
  const client = new LocalSelectionClient(worker as unknown as Worker, source.sha256, source);
  worker.onmessage({ data: { type: "ready", sourceSha256: source.sha256, modelGeneration: 1 } }); await client.ready;
  const a = client.select(request()); const first = messages.at(-1).request;
  const b = client.select(request()); const second = messages.at(-1).request;
  assert.equal(await a, null);
  worker.onmessage({ data: { type: "result", result: { ...first, selected: new Uint8Array([1]) } } });
  worker.onmessage({ data: { type: "result", result: { ...second, selected: new Uint8Array([2]) } } });
  assert.equal((await b)?.selected[0], 2);
  const c = client.select(request()); client.dispose(); assert.equal(await c, null);
  assert.equal(messages.at(-1), "terminated");
});
