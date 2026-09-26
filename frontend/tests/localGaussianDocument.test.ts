import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as documents from "../src/localGaussianDocument.ts";
import type { CloudSource } from "../src/CloudGaussianViewer.tsx";
import { createHookHarness, findNode } from "./hookHarness.ts";

const source: CloudSource = { job_id: "job", asset_role: "scene_splat", label: "原模型" };
const loaded = { plySha256: "a".repeat(64), count: 8, metadataText: "{}" };
const doc: documents.LocalEditDocument = { edit_id: "c".repeat(32), revision: 0, visible_count: 8,
  source: { ...source, ply_sha256: loaded.plySha256, metadata_sha256: "b".repeat(64), gaussian_count: 8, ply_asset: "scene.ply", metadata_asset: "export.json" } };
const plyUrl = "/api/jobs/job/assets/scene.ply", metadataUrl = "/api/jobs/job/assets/export.json";

test("产品文档须匹配Job、variant/role、已加载源SHA/count和两条资产路径", () => {
  documents.validateLocalDocument(doc, source, loaded, plyUrl, metadataUrl);
  for (const changed of [{ job_id: "other" }, { variant_id: "other" }, { asset_role: "scene_splat_vggt_filtered" },
    { ply_sha256: "d".repeat(64) }, { metadata_sha256: "invalid" }, { gaussian_count: 7 }, { ply_asset: "other.ply" }, { metadata_asset: "other.json" }]) {
    assert.throws(() => documents.validateLocalDocument({ ...doc, source: { ...doc.source, ...changed } } as documents.LocalEditDocument, source, loaded, plyUrl, metadataUrl));
  }
  assert.throws(() => documents.validateLocalDocument({ ...doc, edit_id: "../wrong" }, source, loaded, plyUrl, metadataUrl));
});

test("创建结果未知不自动重复，列表核对后打开同源文档；不请求云渲染", async t => {
  const hooks = createHookHarness(), exports: Record<string, any> = {}, calls: { url: string; options: RequestInit }[] = [];
  const control = { current: null as any }, childType = () => null;
  let tree: any, loseCreate = true, metadataText = "{}";
  t.mock.method(globalThis, "fetch", async (input: string, options: RequestInit = {}) => {
    calls.push({ url: input, options });
    if (input === metadataUrl) return new Response(metadataText);
    if (options.method === "POST" && loseCreate) { loseCreate = false; throw new TypeError("响应丢失"); }
    return new Response(JSON.stringify(input.includes("?job_id=") ? { edits: [doc, { ...doc, source: { ...doc.source, job_id: "other" } }] } : doc));
  });
  const code = ts.transpileModule(readFileSync(new URL("../src/LocalGaussianDocumentPanel.tsx", import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX }
  }).outputText;
  runInNewContext(code, { exports, AbortController, AbortSignal, console, fetch: globalThis.fetch, window: { alert() {} },
    require: (name: string) => {
      if (name === "react") return hooks.react;
      if (name === "react/jsx-runtime") return { jsx: (type: unknown, props: unknown) => ({ type, props }), jsxs: (type: unknown, props: unknown) => ({ type, props }) };
      if (name === "./LocalGaussianEditor") return { LocalGaussianEditor: childType };
      if (name === "./localGaussianDocument") return documents;
      if (name.endsWith(".css")) return {};
      throw new Error(name);
    }
  });
  const flush = () => hooks.flush(() => { tree = exports.LocalGaussianDocumentPanel({ viewer: {}, source, loaded, sourceUrl: plyUrl, metadataUrl, handleRef: control }); });
  const button = (label: string) => findNode(tree, n => n.type === "button" && n.props.children === label)?.props;
  await flush(); assert.equal(calls.filter(c => c.options.method === "POST").length, 0);
  button("打开本地编辑").onClick(); assert.equal(await control.current.leave(), false); await flush();
  assert.equal(button("打开本地编辑").disabled, true);
  button("刷新文档列表").onClick(); await flush();
  assert.equal(calls.filter(c => c.options.method === "POST").length, 1);
  findNode(tree, n => n.type === "select").props.onChange({ target: { value: doc.edit_id } }); await flush();
  metadataText = "changed"; button("打开本地编辑").onClick(); await flush();
  assert.match(findNode(tree, n => n.props?.role === "alert").props.children, /元数据/);
  metadataText = "{}"; button("打开本地编辑").onClick(); await flush();
  const opened = findNode(tree, n => n.type === childType).props;
  assert.equal(opened.documentBinding.editId, doc.edit_id); assert.equal(opened.documentBinding.gaussianCount, 8);
  assert.equal(calls.some(c => /render-sessions|\/ice|\/offer/.test(c.url)), false);
  const create = calls.find(c => c.options.method === "POST")!;
  assert.equal(new Headers(create.options.headers).get("X-Image3D-Editor"), "1");
  assert.deepEqual(JSON.parse(String(create.options.body)), { job_id: "job", asset_role: "scene_splat" });
  opened.handleRef.current = { leave: async () => false, dispose: async () => {} };
  assert.equal(await control.current.leave(), false);
  opened.handleRef.current.leave = async () => true; assert.equal(await control.current.leave(), true);
  hooks.unmount(); await control.current.dispose();
});
