import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { createHookHarness, findNode } from "./hookHarness.ts";
import { LocalGaussianEditing } from "../src/localGaussianEditing.ts";

function harness() {
  const hooks = createHookHarness(), exports: Record<string, any> = {};
  const state = new LocalGaussianEditing("a".repeat(64), 8);
  const listeners = new Map<string, (event: any) => void>(), timers = new Map<number, () => void>();
  const masks: (Uint8Array | null)[] = [], downloads: any[] = [], blobs: Blob[] = [];
  let timer = 0, historical = false, resolveSave!: (value: unknown) => void;
  const sync = { editId: "c".repeat(32), dirty: true, busy: false, conflict: false, needsRetry: false, needsBaselineCheck: false,
    baseRevision: 0, lastVersion: null as any, versions: [{ version: "v00000001", revision: 1, visible_count: 7, exported: false }],
    exportState: "", exportVersion: "", saves: 0, exports: [] as string[], imported: 0,
    get hasUnsavedWork() { return this.dirty || state.counts.protected > 0; },
    save: () => { sync.saves++; sync.busy = true; return new Promise(resolve => { resolveSave = resolve; }); },
    draft: () => JSON.stringify({ schema: "test_draft", visible: [state.masks.visible[0]] }),
    restoreDraft: async () => { sync.imported++; return { camera: null, offline: true, conflict: false }; },
    readVersion: async () => new Uint8Array([254]),
    startExport: async (v: string) => { sync.exports.push(v); sync.exportVersion = v; sync.exportState = "running"; },
    pollExport: async () => { sync.exportState = "done"; sync.versions[0].exported = true; },
    downloadUrl: (v: string, name: string) => `/api/gaussian-edits/${sync.editId}/versions/${v}/assets/${name}` };
  const editor = { state, ready: true, get editable() { return !historical; }, cancel() {}, async refresh() {},
    async showHistoricalMask(mask: Uint8Array | null) { historical = mask !== null; masks.push(mask); } };
  const code = ts.transpileModule(readFileSync(new URL("../src/LocalGaussianSavePanel.tsx", import.meta.url), "utf8"),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX } }).outputText;
  runInNewContext(code, { exports, Blob, console,
    URL: { createObjectURL: (blob: Blob) => { blobs.push(blob); return "blob:draft"; }, revokeObjectURL() {} },
    document: { body: { appendChild() {} }, createElement: () => {
      const link = { href: "", download: "", click() { downloads.push({ href: this.href, download: this.download }); }, remove() {} }; return link;
    } },
    window: { addEventListener: (name: string, fn: (event: any) => void) => listeners.set(name, fn), removeEventListener: (name: string) => listeners.delete(name),
      setInterval: (fn: () => void) => { timers.set(++timer, fn); return timer; }, clearInterval: (id: number) => timers.delete(id),
      setTimeout: (fn: () => void) => { fn(); }, confirm: () => true },
    require: (name: string) => {
      if (name === "react") return hooks.react;
      if (name === "react/jsx-runtime") return { jsx: (type: unknown, props: unknown) => ({ type, props }), jsxs: (type: unknown, props: unknown) => ({ type, props }) };
      if (name === "./cloudGaussianEditor.ts") return { captureView: () => null, restoreView: () => null };
      if (name === "./localGaussianDraft.ts") return { LOCAL_DRAFT_LIMIT: 2 * 1024 * 1024 };
      throw new Error(`未预期模块：${name}`);
    }
  });
  let tree: any;
  const render = () => { tree = exports.LocalGaussianSavePanel({ editor, persistence: sync, viewer: { controls: null } }); };
  return { hooks, sync, state, masks, listeners, timers, downloads, blobs,
    render, node: (label: string) => findNode(tree, n => n.type === "button" && n.props.children === label),
    find: (predicate: (node: any) => boolean) => findNode(tree, predicate),
    async flush() { hooks.invalidate(); await hooks.flush(render); },
    finishSave() { sync.busy = false; sync.lastVersion = sync.versions[0]; resolveSave(sync.lastVersion); } };
}

test("保存期间草稿下载仍可用，较新修改不显示为已保存，离页提醒读取最新状态", async () => {
  const h = harness(); await h.flush();
  h.node("保存版本").props.onClick(); await h.flush();
  assert.equal(h.sync.saves, 1); assert.equal(h.node("保存版本").props.disabled, true);
  assert.notEqual(h.node("下载当前编辑草稿").props.disabled, true);
  h.state.select(new Uint8Array([1]), "replace"); h.state.deleteSelected();
  h.node("下载当前编辑草稿").props.onClick(); await h.flush();
  assert.equal(h.downloads[0].download, `${h.sync.editId}-local-draft.json`);
  assert.deepEqual(JSON.parse(await h.blobs[0].text()).visible, [254]);
  h.finishSave(); await h.flush();
  assert.ok(h.find(n => n.type === "p" && typeof n.props.children === "string" && n.props.children.includes("较新的本地修改仍未保存")));
  let prevented = false;
  h.listeners.get("beforeunload")!({ preventDefault() { prevented = true; } }); assert.equal(prevented, true);
  h.sync.dirty = false; prevented = false;
  h.listeners.get("beforeunload")!({ preventDefault() { prevented = true; } }); assert.equal(prevented, false);
  h.state.select(new Uint8Array([2]), "replace"); h.state.protectSelected(true);
  h.listeners.get("beforeunload")!({ preventDefault() { prevented = true; } }); assert.equal(prevented, true);
  h.hooks.unmount(); assert.equal(h.listeners.size, 0); assert.equal(h.timers.size, 0);
});

test("历史只读查看与返回不保存，导出独立且使用服务端下载链接", async () => {
  const h = harness(); await h.flush(); const original = h.state.masks.visible.slice();
  h.node("只读查看此版本").props.onClick(); await h.flush();
  assert.equal(h.masks[0]![0], 254); assert.equal(h.node("保存版本").props.disabled, true);
  h.node("返回当前编辑").props.onClick(); await h.flush();
  assert.equal(h.masks[1], null); assert.deepEqual(h.state.masks.visible, original);
  h.node("导出此版本 PLY／ZIP").props.onClick(); await h.flush();
  assert.deepEqual(h.sync.exports, ["v00000001"]); assert.equal(h.sync.saves, 0);
  for (const callback of h.timers.values()) callback(); await h.flush();
  assert.match(h.find(n => n.type === "a" && n.props.children === "下载 ZIP").props.href, /\/assets\/bundle.zip$/);
  h.hooks.unmount();
});

test("草稿文件在读取前受限，待确认保存禁用导入", async () => {
  const h = harness(); await h.flush();
  const input = () => h.find(n => n.type === "input" && n.props.type === "file");
  input().props.onChange({ target: { files: [{ size: 2 * 1024 * 1024 + 1, text() { assert.fail("超限文件不应读取"); } }], value: "file" } });
  await h.flush(); assert.equal(h.sync.imported, 0); assert.ok(h.find(n => n.props?.role === "alert"));
  h.sync.needsRetry = true; await h.flush(); assert.equal(input().props.disabled, true);
  h.hooks.unmount();
});
