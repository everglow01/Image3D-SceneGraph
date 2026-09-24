import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { createHookHarness, findNode } from "./hookHarness.ts";
import { LocalGaussianEditing } from "../src/localGaussianEditing.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(yes => { resolve = yes; });
  return { promise, resolve };
}

test("独立面板快速换源等待旧打开和清理，不让旧cleanup释放新会话", async () => {
  const hooks = createHookHarness(), exports: Record<string, any> = {};
  const opening = deferred<any>(), closing = deferred<void>();
  const opened: string[] = [], closed: string[] = [], workers: { terminated: boolean }[] = [];
  const a = "a".repeat(64), b = "b".repeat(64);
  const handle = (sha256: string) => ({ source: { sha256, count: 6, geometry: new Float32Array(66) } });
  const viewer = { renderer: { domElement: {} }, beginLocalEdit: (sha: string) => {
    opened.push(sha); return sha === a ? opening.promise : Promise.resolve(handle(sha));
  }, endLocalEdit: (h: any) => { closed.push(h.source.sha256); return h.source.sha256 === a ? closing.promise : Promise.resolve(); } };
  const code = ts.transpileModule(readFileSync(new URL("../src/LocalGaussianEditor.tsx", import.meta.url), "utf8")
    .replaceAll("import.meta.url", JSON.stringify(new URL("../src/LocalGaussianEditor.tsx", import.meta.url).href)),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX } }).outputText;
  const frames = new Map<number, () => void>(); let nextFrame = 0;
  runInNewContext(code, { exports, AbortController, URL, console, document: { body: {} },
    requestAnimationFrame: (callback: () => void) => { frames.set(++nextFrame, callback); return nextFrame; },
    cancelAnimationFrame: (id: number) => frames.delete(id),
    Worker: class { terminated = false; constructor() { workers.push(this); } terminate() { this.terminated = true; } },
    require: (name: string) => {
      if (name === "react") return { ...hooks.react, useMemo: (f: () => unknown) => f() };
      if (name === "react/jsx-runtime") return { jsx: (type: unknown, props: unknown) => ({ type, props }), jsxs: (type: unknown, props: unknown) => ({ type, props }) };
      if (name === "react-dom") return { createPortal: (node: unknown) => node };
      if (name === "./localGaussianSelectionClient.ts") return { LocalSelectionClient: class {
        ready = Promise.resolve(); worker: { terminate(): void };
        constructor(worker: { terminate(): void }) { this.worker = worker; }
        dispose() { this.worker.terminate(); }
      } };
      if (name === "./localGaussianInteraction.ts") return { LocalGaussianInteraction: class {
        state: LocalGaussianEditing; ready = true; mode = "surface"; operation = "replace"; tool = "rectangle"; polygon = []; highlight = true;
        private handle: any; private client: any;
        constructor(_viewer: unknown, h: any, client: any) { this.handle = h; this.client = client; this.state = new LocalGaussianEditing(h.source.sha256, h.source.count); }
        async refresh() {}
        async dispose() { this.ready = false; this.client.dispose(); await viewer.endLocalEdit(this.handle); }
      } };
      if (name.endsWith(".css")) return {};
      throw new Error(`未预期模块：${name}`);
    }
  });
  let sha = a, tree: any;
  const render = () => { tree = exports.LocalGaussianEditor({ viewer, sourceSha256: sha }); };
  await hooks.flush(render, () => opened.length === 1);
  sha = b; hooks.invalidate(); await hooks.flush(render);
  assert.deepEqual(opened, [a]);
  opening.resolve(handle(a)); await hooks.flush(render, () => closed.length === 1);
  assert.deepEqual(opened, [a]); assert.equal(workers.length, 0);
  closing.resolve();
  await hooks.flush(render, () => Boolean(findNode(tree, n => n.type === "button" && n.props.children === "隐藏选中项（Delete）")));
  assert.deepEqual(opened, [a, b]); assert.deepEqual(closed, [a]); assert.equal(workers.length, 1);
  hooks.unmount(); await hooks.flush(() => {}, () => closed.length === 2);
  assert.deepEqual(closed, [a, b]); assert.equal(workers[0].terminated, true); assert.equal(frames.size, 0);
});
