import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as THREE from "three";
import * as metadata from "../src/gaussianViewerMetadata.ts";

const code = ts.transpileModule(readFileSync(new URL("../src/GaussianSplatViewer.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX }
}).outputText;

function pageHarness(options: { sourceUrl?: string; search?: string } = {}) {
  const hooks: any[] = [], effects: (() => void)[] = [], instances: any[] = [];
  let cursor = 0, dirty = true, tree: any;
  const mount = { clientWidth: 800, clientHeight: 400, replaceChildren() {}, appendChild() {} };
  class Renderer {
    domElement = { remove() {} };
    disposed = false;
    contextLost = false;
    setPixelRatio() {} setClearColor() {} setSize() {}
    dispose() { this.disposed = true; }
    forceContextLoss() { this.contextLost = true; }
  }
  const react = {
    useRef: (initial: unknown) => {
      const index = cursor++;
      return hooks[index] ??= { current: initial };
    },
    useState: (initial: unknown) => {
      const index = cursor++;
      if (!(index in hooks)) hooks[index] = initial;
      return [hooks[index], (next: any) => {
        const value = typeof next === "function" ? next(hooks[index]) : next;
        if (!Object.is(value, hooks[index])) { hooks[index] = value; dirty = true; }
      }];
    },
    useEffect: (callback: () => (() => void) | void, deps: unknown[]) => {
      const index = cursor++, previous = hooks[index];
      if (!previous || deps.some((d, i) => !Object.is(d, previous.deps[i]))) {
        effects.push(() => {
          previous?.cleanup?.(); hooks[index] = { deps, cleanup: callback() };
        });
      }
    }
  };
  const jsx = (type: unknown, props: any) => {
    if (props.ref) props.ref.current = mount;
    return { type, props };
  };
  class Viewer {
    kind = "legacy";
    camera = new THREE.PerspectiveCamera(50, 2, 0.01, 100);
    controls = { target: new THREE.Vector3(), update() {}, saveState() {} };
    stopped = false;
    disposed = false;
    disposal = Promise.resolve();
    disposeError: Error | null = null;
    renderer = { domElement: {} };
    constructor(options?: { renderer?: Renderer }) {
      if (options?.renderer) this.renderer = options.renderer;
      instances.push(this);
    }
    async addSplatScene(sourceUrl: string) { this.sourceUrl = sourceUrl; }
    sourceUrl = "";
    getSplatMesh() { return { minSphericalHarmonicsDegree: 2,
      material: new THREE.ShaderMaterial(), computeBoundingBox: () => this.bounds() }; }
    bounds() { return new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)); }
    start() {} stop() { this.stopped = true; } forceRenderNextFrame() {}
    async dispose() {
      await this.disposal;
      this.disposed = true;
      if (this.disposeError) throw this.disposeError;
    }
  }
  class Spark extends Viewer {
    kind = "spark";
    constructor(..._args: unknown[]) { super(); this.camera.aspect = 1.5; }
    async load() {}
    getBoundingBox() { return this.bounds(); }
  }
  const exports: Record<string, any> = {};
  runInNewContext(code, { exports, AbortController, URLSearchParams,
    window: { location: { search: options.search ?? "" } },
    fetch: async (_url: string, _options: unknown) => ({ ok: true, headers: { get: () => null },
      json: async () => ({ sh_degree: 3, viewer_minimum_opacity: 0.005, scene_radius_p95: 1 }) }),
    document: { pointerLockElement: null },
    ResizeObserver: class { observe() {} disconnect() {} },
    require: (name: string) => {
      if (name === "react") return react;
      if (name === "react/jsx-runtime") return { jsx, jsxs: jsx };
      if (name === "three") return { ...THREE, WebGLRenderer: Renderer };
      if (name === "@mkkellogg/gaussian-splats-3d") return { Viewer, RenderMode: { OnChange: 1 } };
      if (name.endsWith("package.json")) return { name: "legacy", version: "0.4.7" };
      if (name === "./gaussianViewerMetadata") return metadata;
      if (name === "./SparkPageViewer") return { SparkPageViewer: Spark };
      return {};
    }
  });
  const props = { sourceUrl: options.sourceUrl ?? "/scene.ply", metadataUrl: "/export.json", cameraPathUrl: null,
    alignmentUrl: null, jobId: "job", sfmDiagnosticsUrl: null, inspectionRequest: null,
    onInspectionStateChange() {}, collisionMeshUrl: null, navigationUrl: null,
    navigationStatus: null, navigationReason: null };
  const find = (node: any, predicate: (node: any) => boolean): any => {
    if (!node || typeof node !== "object") return null;
    if (predicate(node)) return node;
    for (const child of [node.props?.children].flat(Infinity)) {
      const result = find(child, predicate); if (result) return result;
    }
    return null;
  };
  const flush = async () => {
    for (let i = 0; i < 40; i++) {
      await Promise.resolve();
      if (dirty) {
        dirty = false; cursor = 0;
        tree = exports.GaussianSplatViewer(props);
        while (effects.length) effects.shift()!();
      }
    }
  };
  return { instances, flush,
    selector: () => find(tree, n => n.type === "select").props,
    experimentSelector: () => find(tree, n => n.props?.["aria-label"] === "浏览资产试验")?.props,
    hint: () => find(tree, n => n.props?.className === "viewer-hint")?.props.children.flat().join(""),
    changeSource: () => { props.sourceUrl = "/other.ply"; dirty = true; },
    changeNavigationStatus: (status: string) => { props.navigationStatus = status; dirty = true; },
    unmount: () => { for (const hook of hooks) hook?.cleanup?.(); } };
}

test("page defaults to legacy, waits for release, and preserves camera/target when switching", async () => {
  const h = pageHarness(); await h.flush();
  assert.equal(h.selector().value, "legacy");
  assert.equal(h.instances.length, 1);
  const old = h.instances[0];
  old.camera.position.set(4, 5, 6); old.camera.fov = 71; old.camera.zoom = 1.2;
  old.controls.target.set(0.2, 0.3, 0.4);
  let release!: () => void;
  old.disposal = new Promise<void>(resolve => { release = resolve; });
  h.selector().onChange({ target: { value: "spark" } }); await h.flush();
  assert.equal(old.stopped, true);
  assert.equal(h.instances.length, 1);
  release(); await h.flush();
  assert.equal(old.disposed, true);
  assert.equal(old.renderer.disposed, true);
  assert.equal(old.renderer.contextLost, true);
  const spark = h.instances[1];
  assert.equal(spark.kind, "spark");
  assert.deepEqual(spark.camera.position.toArray(), [4, 5, 6]);
  assert.equal(spark.camera.fov, 71);
  assert.equal(spark.camera.zoom, 1.2);
  assert.equal(spark.camera.aspect, 1.5);
  assert.deepEqual(spark.controls.target.toArray(), [0.2, 0.3, 0.4]);
  assert.match(h.hint(), /浏览器 SH3/);
  h.selector().onChange({ target: { value: "legacy" } }); await h.flush();
  assert.equal(spark.disposed, true);
  assert.equal(h.instances[2].kind, "legacy");
  assert.deepEqual(h.instances[2].camera.position.toArray(), [4, 5, 6]);
  assert.match(h.hint(), /浏览器 SH2/);
  h.unmount(); await h.flush();
  assert.equal(h.instances[2].disposed, true);
});

test("navigation status changes do not download the same splat again", async () => {
  const h = pageHarness(); await h.flush();
  assert.equal(h.instances.length, 1);
  h.changeNavigationStatus("available"); await h.flush();
  assert.equal(h.instances.length, 1);
  assert.equal(h.instances[0].disposed, false);
  h.unmount(); await h.flush();
});

test("compact browser assets are opt-in and preserve the same model camera", async () => {
  const sourceUrl = "/api/jobs/train_validation_final_fit_v1_comparison/assets/variants/mcmc-final-fit/scene.ply";
  const ordinary = pageHarness({ sourceUrl }); await ordinary.flush();
  assert.equal(ordinary.experimentSelector(), undefined);
  ordinary.unmount(); await ordinary.flush();

  const h = pageHarness({ sourceUrl, search: "?browser-ksplat-ab=1" }); await h.flush();
  assert.equal(h.instances[0].sourceUrl, sourceUrl);
  h.instances[0].camera.position.set(4, 5, 6);
  h.instances[0].controls.target.set(0.2, 0.3, 0.4);
  h.experimentSelector().onChange({ target: { value: "level1" } }); await h.flush();
  assert.match(h.instances[1].sourceUrl, /level1\.ksplat$/);
  assert.equal(h.instances[0].disposed, true);
  assert.deepEqual(h.instances[1].camera.position.toArray(), [4, 5, 6]);
  assert.deepEqual(h.instances[1].controls.target.toArray(), [0.2, 0.3, 0.4]);
  h.experimentSelector().onChange({ target: { value: "level2" } }); await h.flush();
  assert.match(h.instances[2].sourceUrl, /level2\.ksplat$/);
  h.experimentSelector().onChange({ target: { value: "ply" } }); await h.flush();
  assert.equal(h.instances[3].sourceUrl, sourceUrl);
  h.unmount(); await h.flush();

  const other = pageHarness({ sourceUrl: "/other.ply", search: "?browser-ksplat-ab=1" });
  await other.flush();
  assert.equal(other.experimentSelector(), undefined);
  other.unmount(); await other.flush();
});

test("rapid switching cannot bypass pending disposal or restore a camera into another model", async () => {
  const h = pageHarness(); await h.flush();
  const old = h.instances[0]; old.camera.position.set(40, 50, 60);
  let release!: () => void;
  old.disposal = new Promise<void>(resolve => { release = resolve; });
  h.selector().onChange({ target: { value: "spark" } }); await h.flush();
  h.selector().onChange({ target: { value: "legacy" } }); await h.flush();
  h.changeSource(); await h.flush();
  assert.equal(h.instances.length, 1);
  release(); await h.flush();
  assert.equal(h.instances.length, 2);
  assert.equal(h.instances[1].kind, "legacy");
  assert.notDeepEqual(h.instances[1].camera.position.toArray(), [40, 50, 60]);
  h.unmount(); await h.flush();
});

test("legacy abort rejection allows switching only when its cleanup actually completed", async () => {
  const h = pageHarness(); await h.flush();
  h.instances[0].disposeError = new Error("Scene disposed");
  h.selector().onChange({ target: { value: "spark" } }); await h.flush();
  assert.equal(h.instances.length, 2);
  assert.equal(h.instances[0].renderer.contextLost, true);
  h.unmount(); await h.flush();

  const failed = pageHarness(); await failed.flush();
  failed.instances[0].dispose = async () => { throw new Error("cleanup incomplete"); };
  failed.selector().onChange({ target: { value: "spark" } }); await failed.flush();
  assert.equal(failed.instances.length, 1);
  failed.unmount(); await failed.flush();
});
