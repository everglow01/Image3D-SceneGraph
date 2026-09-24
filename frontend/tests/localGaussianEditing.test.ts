import assert from "node:assert/strict";
import test from "node:test";
import { LOCAL_HISTORY_BYTES, LocalGaussianEditing, localEditorShortcut } from "../src/localGaussianEditing.ts";

const sha = "a".repeat(64);
const mask = (bits: number) => new Uint8Array([bits]);

test("本地选集新建增减与源 mask 隔离，拒绝空减选和 padding", () => {
  const s = new LocalGaussianEditing(sha, 6);
  assert.throws(() => s.select(mask(1), "subtract"), /空选集/);
  assert.throws(() => s.select(mask(128), "replace"), /补齐/);
  const input = mask(3); s.select(input, "replace"); input.fill(0);
  s.select(mask(4), "add"); s.select(mask(2), "subtract");
  assert.equal(s.masks.selected[0], 5);
  const output = s.masks; output.visible.fill(0); output.selected.fill(0);
  assert.equal(s.counts.visible, 6); assert.equal(s.counts.selected, 2);
});

test("保护、连续删除和撤销重做共用历史，新操作截断重做", () => {
  const s = new LocalGaussianEditing(sha, 6);
  s.select(mask(1), "replace"); s.protectSelected(true);
  s.select(mask(7), "replace"); s.deleteSelected();
  assert.equal(s.masks.visible[0], 57); assert.equal(s.masks.protected[0], 1);
  assert.equal(s.masks.selected[0], 1);
  s.undo(); assert.equal(s.masks.visible[0], 63);
  s.undo(); assert.equal(s.masks.protected[0], 0);
  s.redo(); s.redo(); assert.equal(s.masks.visible[0], 57);
  s.undo(); s.select(mask(8), "replace"); s.deleteSelected();
  assert.equal(s.canRedo, false); assert.equal(s.masks.visible[0], 55);
  s.select(mask(1), "replace"); s.protectSelected(false);
  s.undo(); assert.equal(s.masks.protected[0], 1);
});

test("隔离、删除预览和原始对照不修改可见状态或历史", () => {
  const s = new LocalGaussianEditing(sha, 6);
  s.select(mask(3), "replace"); s.protectSelected(true);
  s.select(mask(7), "replace");
  const bytes = s.historyBytes;
  s.setPreview("isolate"); assert.equal(s.displayMask[0], 7);
  s.setPreview("deletion"); assert.equal(s.displayMask[0], 59);
  assert.equal(s.masks.visible[0], 63); assert.equal(s.historyBytes, bytes);
  s.deleteSelected(); assert.equal(s.previewMode, "none");
  s.setPreview("original"); assert.equal(s.displayMask[0], 63);
  assert.equal(s.masks.visible[0], 59);
});

test("大比例删除必须确认，禁止全删，保护项排除后计算比例", () => {
  const s = new LocalGaussianEditing(sha, 6);
  s.select(mask(63), "replace");
  assert.throws(() => s.deleteSelected(true), /全部/);
  s.select(mask(31), "replace");
  assert.throws(() => s.deleteSelected(), /50%/);
  assert.equal(s.counts.visible, 6);
  s.deleteSelected(true); assert.equal(s.counts.visible, 1);
  s.select(mask(31), "replace"); assert.equal(s.counts.selected, 0);
});

test("100步与统一内存预算限制保护和可见快照，不复制模型", () => {
  const s = new LocalGaussianEditing(sha, 3_000_000);
  const selected = new Uint8Array(375_000); selected[0] = 1;
  s.select(selected, "replace");
  for (let i = 0; i < 110; i++) s.protectSelected(i % 2 === 0);
  assert.equal(s.historyBytes, 101 * 2 * 375_000);
  assert.ok(s.historyBytes <= LOCAL_HISTORY_BYTES);
  let undone = 0; while (s.undo()) undone++;
  assert.equal(undone, 100);
});

test("快捷键不劫持文本输入、输入法或其他组合键", () => {
  const event = { key: "Delete", ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, target: null };
  assert.equal(localEditorShortcut(event), "delete");
  assert.equal(localEditorShortcut({ ...event, key: "z", ctrlKey: true }), "undo");
  assert.equal(localEditorShortcut({ ...event, key: "z", metaKey: true, shiftKey: true }), "redo");
  assert.equal(localEditorShortcut({ ...event, isComposing: true }), null);
  assert.equal(localEditorShortcut({ ...event, altKey: true }), null);
  assert.equal(localEditorShortcut({ ...event, target: { closest: () => ({}) } as unknown as EventTarget }), null);
  assert.equal(localEditorShortcut({ ...event, target: { isContentEditable: true } as unknown as EventTarget }), null);
});
