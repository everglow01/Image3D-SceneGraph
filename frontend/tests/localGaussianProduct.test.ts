import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { createHookHarness, findNode } from "./hookHarness.ts";
import * as leave from "../src/gaussianViewerLeave.ts";
import * as backend from "../src/backendOptions.ts";
import * as evidence from "../src/reconstructionEvidence.ts";
import * as kinds from "../src/resultKind.ts";
import * as variants from "../src/gaussianVariants.ts";
import * as sfm from "../src/sfmOptions.ts";
import * as trainers from "../src/trainerOptions.ts";

function compile(file: string) {
  return ts.transpileModule(readFileSync(new URL(`../src/${file}.tsx`, import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX }
  }).outputText;
}
const jsx = (type: unknown, props: unknown) => ({ type, props });

test("App在Job、variant、几何模式、证据视图变更前检查离开；刷新不重置正在编辑的variant", async () => {
  const hooks = createHookHarness(), exports: Record<string, any> = {};
  let tree: any, pendingRefresh: ((value: unknown) => void) | null = null, holdRefresh = false, newNavigation = false;
  const timers = new Map<number, () => void>(); let timer = 0;
  const manifest = (job_id: string) => ({ job_id, status: "done", navigation_status: "queued", stage: "done", progress: 1, mode: "multi_image", output_type: "gaussian_splat",
    geometry_backend: "project_3dgs", created_at: "2026-09-26", inputs: [], metrics: {}, assets: {
      scene_splat: "scene.ply", gaussian_export_metadata: "export.json", scene_splat_vggt_filtered: "filtered.ply",
      gaussian_vggt_filtered_export_metadata: "filtered.json", sfm_sparse_point_cloud: "points.ply",
      ...(newNavigation ? { navigation: "navigation.json", collision_mesh: "collision.glb" } : {}) } });
  runInNewContext(compile("App"), { exports, console, window: { addEventListener() {}, removeEventListener() {},
    setInterval: (fn: () => void) => { timers.set(++timer, fn); return timer; }, clearInterval: (id: number) => timers.delete(id) },
    fetch: async (url: string) => {
      if (url.endsWith("/a/manifest") && holdRefresh) { holdRefresh = false; return await new Promise(resolve => { pendingRefresh = resolve; }); }
      return { ok: true, json: async () => url === "/api/backends" ? { backends: [] } : url === "/api/jobs" ? { jobs: [manifest("a"), manifest("b")] } :
        url.endsWith("/scene") ? { objects: [] } : manifest(url.split("/")[3]) };
    },
    require: (name: string) => {
      if (name === "react") return { ...hooks.react, useMemo: (f: () => unknown) => f() };
      if (name === "react/jsx-runtime") return { jsx, jsxs: jsx };
      if (name === "lucide-react") return new Proxy({}, { get: (_target, key) => String(key) });
      if (name.endsWith(".png")) return "logo";
      const modules: Record<string, unknown> = { "./backendOptions": backend, "./reconstructionEvidence": evidence, "./resultKind": kinds,
        "./gaussianVariants": variants, "./sfmOptions": sfm, "./trainerOptions": trainers, "./gaussianViewerLeave": leave,
        "./GeometryViewer": { GeometryViewer: "GeometryViewer" }, "./ReconstructionEvidenceRail": { ReconstructionEvidenceRail: "Evidence" } };
      if (name in modules) return modules[name];
      throw new Error(name);
    }
  });
  const flush = () => hooks.flush(() => { tree = exports.App(); });
  const geometry = () => findNode(tree, n => n.type === "GeometryViewer").props;
  const jobSelector = () => findNode(tree, n => n.props?.["aria-label"] === "选择历史任务").props;
  const button = (text: string) => findNode(tree, n => n.type === "button" && [n.props.children].flat(Infinity).some((c: any) => c === text || c?.props?.children === text)).props;
  await flush(); jobSelector().onChange({ target: { value: "a" } }); await flush();
  assert.equal(geometry().jobId, "a"); assert.match(geometry().splatUrl, /filtered.ply$/);
  let allow = false, checks = 0;
  geometry().leaveRef.current = async () => { checks++; return allow; };
  button("Original").onClick(); await flush(); assert.match(geometry().splatUrl, /filtered.ply$/);
  button("SfM 稀疏点云").onClick(); await flush(); assert.ok(geometry().splatUrl);
  findNode(tree, n => n.type === "Evidence").props.onSelect("matching"); await flush(); assert.equal(geometry().inspectionRequest, null);
  jobSelector().onChange({ target: { value: "b" } }); await flush(); assert.equal(geometry().jobId, "a"); assert.equal(jobSelector().value, "a");
  assert.equal(checks, 4);
  allow = true; button("Original").onClick(); await flush(); assert.match(geometry().splatUrl, /\/scene.ply$/);
  const before = checks; button("刷新").onClick(); await flush();
  assert.equal(checks, before); assert.match(geometry().splatUrl, /\/scene.ply$/);
  geometry().onLocalModeChange(true); await flush(); newNavigation = true;
  for (const tick of timers.values()) tick(); await flush();
  assert.equal(checks, before); assert.equal(geometry().navigationUrl, null);
  geometry().onLocalModeChange(false); newNavigation = false; await flush();
  holdRefresh = true; button("刷新").onClick(); await flush();
  jobSelector().onChange({ target: { value: "b" } }); await flush(); assert.equal(geometry().jobId, "b");
  pendingRefresh!({ ok: true, json: async () => manifest("a") }); await flush(); assert.equal(geometry().jobId, "b");
  hooks.unmount();
});

test("Geometry本地与云切换复用同一离开门禁，拒绝时保持原组件", async () => {
  const hooks = createHookHarness(), exports: Record<string, any> = {};
  let tree: any, allow = false;
  const leaveRef = { current: async () => allow };
  runInNewContext(compile("GeometryViewer"), { exports, require: (name: string) => {
    if (name === "react") return hooks.react;
    if (name === "react/jsx-runtime") return { jsx, jsxs: jsx };
    if (name === "./gaussianViewerLeave") return leave;
    return { [name.slice(2)]: name.slice(2) };
  } });
  const props = { leaveRef, cloudSource: { job_id: "a", asset_role: "scene_splat", label: "原模型" },
    splatUrl: "/scene.ply", onInspectionStateChange() {} };
  const flush = () => hooks.flush(() => { tree = exports.GeometryViewer(props); });
  const button = (text: string) => findNode(tree, n => n.type === "button" && n.props.children === text).props;
  await flush(); assert.equal(findNode(tree, n => n.type === "GaussianSplatViewer").props.leaveRef, leaveRef);
  button("云端查看与修剪").onClick(); await flush(); assert.ok(findNode(tree, n => n.type === "GaussianSplatViewer"));
  allow = true; button("云端查看与修剪").onClick(); await flush();
  assert.equal(findNode(tree, n => n.type === "CloudGaussianViewer").props.leaveRef, leaveRef);
  allow = false; button("本地查看").onClick(); await flush(); assert.ok(findNode(tree, n => n.type === "CloudGaussianViewer"));
  allow = true; button("本地查看").onClick(); await flush(); assert.ok(findNode(tree, n => n.type === "GaussianSplatViewer"));
  hooks.unmount();
});
