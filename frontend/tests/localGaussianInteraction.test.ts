import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import { LocalGaussianInteraction } from "../src/localGaussianInteraction.ts";
import type { LocalSelectionClient } from "../src/localGaussianSelectionClient.ts";
import type { SparkLocalEdit, SparkPageViewer } from "../src/SparkPageViewer.ts";
import type { P0SelectionResult } from "../src/localGaussianP0Selection.ts";

function harness() {
  const oldRaf = globalThis.requestAnimationFrame, oldCancel = globalThis.cancelAnimationFrame;
  const frames = new Map<number, FrameRequestCallback>(); let frame = 0;
  globalThis.requestAnimationFrame = callback => { frames.set(++frame, callback); return frame; };
  globalThis.cancelAnimationFrame = id => { frames.delete(id); };
  const listeners = new Map<string, Set<EventListener>>(); const captures = new Set<number>();
  const canvas = { tabIndex: -1, style: { touchAction: "" }, focus() {}, getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 }),
    addEventListener: (type: string, fn: EventListener) => { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type)!.add(fn); },
    removeEventListener: (type: string, fn: EventListener) => { listeners.get(type)?.delete(fn); if (!listeners.get(type)?.size) listeners.delete(type); },
    setPointerCapture: (id: number) => captures.add(id), hasPointerCapture: (id: number) => captures.has(id), releasePointerCapture: (id: number) => captures.delete(id) };
  let signature = "first", ended = false, terminated = false;
  const requests: any[] = [], displays: Uint8Array[] = [];
  let reply!: (value: P0SelectionResult | null) => void, arrived!: () => void;
  const requested = new Promise<void>(resolve => { arrived = resolve; });
  const client = { cancel() {}, dispose() { terminated = true; }, select(r: any) {
    requests.push(r); arrived(); return new Promise<P0SelectionResult | null>(resolve => { reply = resolve; });
  } } as unknown as LocalSelectionClient;
  const viewer = { controls: { enabled: true }, camera: new THREE.PerspectiveCamera(), renderer: { domElement: canvas },
    localEditView: () => ({ signature, width: 400, height: 200, modelToView: new THREE.Matrix4().elements,
      projection: new THREE.PerspectiveCamera().projectionMatrix.elements }),
    updateLocalEdit: async (_handle: unknown, visible: Uint8Array) => { displays.push(visible); },
    endLocalEdit: async () => { ended = true; } } as unknown as SparkPageViewer;
  const handle: SparkLocalEdit = { source: { sha256: "a".repeat(64), count: 6, geometry: new Float32Array(66) },
    sourceRoot: new THREE.Group(), overlay: new THREE.Group(), bounds: new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)) };
  const session = new LocalGaussianInteraction(viewer, handle, client, () => {});
  return { session, viewer, handle, requests, displays, requested,
    reply: (bits: number) => reply({ selected: new Uint8Array([bits]), selectedCount: 1 } as P0SelectionResult),
    changeCamera: () => { signature = "changed"; },
    tick: () => { const [id, callback] = frames.entries().next().value!; frames.delete(id); callback(0); },
    event: (type: string, x: number, y: number) => listeners.get(type)!.values().next().value!({ button: 0, pointerId: 1, clientX: x, clientY: y,
      preventDefault() {}, stopImmediatePropagation() {} } as unknown as Event),
    async close() {
      try { await session.dispose(); assert.equal(ended, true); assert.equal(terminated, true); assert.equal(listeners.size, 0); assert.equal(frames.size, 0); }
      finally { globalThis.requestAnimationFrame = oldRaf; globalThis.cancelAnimationFrame = oldCancel; }
    } };
}
const polygon: [number, number][] = [[10, 10], [20, 10], [20, 20], [10, 20]];

test("相机变化即使尚未被动画帧观察，也拒绝旧选择响应", async () => {
  const h = harness();
  try {
    const pending = h.session.selectPolygon(polygon); await h.requested;
    h.changeCamera(); h.reply(2); await pending;
    assert.equal(h.session.state.counts.selected, 0);
  } finally { await h.close(); }
});

test("完成的选集跨视角保留；保护或换工具后旧结果不落地", async () => {
  const h = harness();
  try {
    const pending = h.session.selectPolygon(polygon); await h.requested; h.reply(2); await pending;
    h.changeCamera(); h.tick(); assert.equal(h.session.state.counts.selected, 1);
  } finally { await h.close(); }
  const h2 = harness();
  try {
    const pending = h2.session.selectPolygon(polygon); await h2.requested;
    await h2.session.changeTool("navigate"); h2.reply(4); await pending;
    assert.equal(h2.session.state.counts.selected, 0);
  } finally { await h2.close(); }
});

test("CSS坐标转换到实际绘图像素，拖选锁导航而松开释放", async () => {
  const h = harness();
  try {
    h.event("pointerdown", 20, 30); assert.equal(h.viewer.controls!.enabled, false);
    h.event("pointermove", 40, 50); h.event("pointerup", 40, 50);
    assert.equal(h.viewer.controls!.enabled, true);
    await h.requested;
    assert.deepEqual(h.requests[0].polygon, [[20, 20], [60, 20], [60, 60], [20, 60]]);
    h.reply(2);
  } finally { await h.close(); }
});

test("真实 Three 盒控制柄使用源轴局部边界，拖动时锁定导航", async () => {
  const h = harness();
  try {
    h.handle.sourceRoot.rotation.z = Math.PI / 2;
    await h.session.changeTool("box");
    h.session.setBoxMode("scale");
    const control = (h.handle.overlay.children[0] as THREE.Object3D & { controls: import("three/addons/controls/TransformControls.js").TransformControls }).controls;
    assert.equal(control.getMode(), "scale"); assert.equal(control.space, "local");
    control.dispatchEvent({ type: "dragging-changed", value: true }); assert.equal(h.viewer.controls!.enabled, false);
    const box = h.handle.sourceRoot.children[0]; box.position.set(1, 2, 3); box.scale.set(2, 4, 6);
    control.dispatchEvent({ type: "objectChange" });
    control.dispatchEvent({ type: "dragging-changed", value: false }); assert.equal(h.viewer.controls!.enabled, true);
    const pending = h.session.selectBox(); await h.requested;
    assert.deepEqual(h.requests[0].box, { min: [0, 0, 0], max: [2, 4, 6] });
    assert.equal(h.requests[0].mode, "box"); h.reply(1); await pending;
  } finally { await h.close(); }
});

test("更新中的保护状态与卸载阻断在途结果；预览不提供选择", async () => {
  const h = harness();
  try {
    h.session.state.select(new Uint8Array([1]), "replace");
    const pending = h.session.selectPolygon(polygon); await h.requested;
    await h.session.act(() => h.session.state.protectSelected(true));
    h.reply(2); await pending; assert.equal(h.session.state.masks.selected[0], 1);
    await h.session.preview("original");
    await assert.rejects(h.session.selectPolygon(polygon), /返回编辑/);
    await assert.rejects(h.session.deleteSelection(), /原始对照/);
  } finally { await h.close(); }
});
