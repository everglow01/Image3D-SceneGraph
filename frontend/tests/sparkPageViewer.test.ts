import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as THREE from "three";

function deferred() {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

const source = readFileSync(new URL("../src/SparkPageViewer.ts", import.meta.url), "utf8");
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;

function harness(options: { initialization?: Promise<void>; sh?: number; constructorError?: boolean } = {}) {
  const frames = new Map<number, () => void>();
  const events = new Map<string, (event: Event) => void>();
  let frameId = 0, disposed = 0, observerDisconnected = false, updateCount = 0;
  let update: () => Promise<void> = () => Promise.resolve();
  let signal: AbortSignal | undefined;
  let meshOptions: Record<string, unknown> = {}, rendererOptions: Record<string, unknown> = {};
  const errors: unknown[] = [];
  const sizes: number[][] = [];
  const mount = { clientWidth: 800, clientHeight: 400, appendChild() {} };
  const canvas = { addEventListener: (k: string, f: (e: Event) => void) => events.set(k, f),
    removeEventListener: (k: string) => events.delete(k), remove() { disposed++; } };
  class Renderer {
    domElement = canvas;
    setPixelRatio() {} setClearColor() {}
    setSize(w: number, h: number) { sizes.push([w, h]); }
    render() {}
    getContext() { return { NO_ERROR: 0, isContextLost: () => false, getError: () => 0 }; }
    dispose() { disposed++; } forceContextLoss() { disposed++; }
  }
  class Controls {
    target = new THREE.Vector3();
    update() {} dispose() { disposed++; }
  }
  class Spark extends THREE.Object3D {
    display = { numSplats: 6 };
    constructor(opts: Record<string, unknown>) {
      super(); rendererOptions = opts;
      if (options.constructorError) throw new Error("initialization failed");
    }
    update() { updateCount++; return update(); }
    dispose() { disposed++; }
  }
  class Mesh extends THREE.Object3D {
    initialized = options.initialization ?? Promise.resolve();
    numSplats = 6;
    extSplats = { getNumSh: () => options.sh ?? 3, setMaxSh() {} };
    constructor(opts: Record<string, unknown>) { super(); meshOptions = opts; }
    getBoundingBox() { return new THREE.Box3(new THREE.Vector3(1, 2, 3), new THREE.Vector3(2, 3, 4)); }
    dispose() { disposed++; }
  }
  let resized!: () => void;
  const exports: Record<string, any> = {};
  runInNewContext(code, {
    exports, AbortController,
    require: (name: string) => name === "three" ? { ...THREE, WebGLRenderer: Renderer }
      : name === "@mkkellogg/gaussian-splats-3d" ? { OrbitControls: Controls }
      : { SparkRenderer: Spark, SplatMesh: Mesh, SplatFileType: { PLY: "ply" } },
    fetch: async (_url: string, opts: { signal: AbortSignal }) => {
      signal = opts.signal; return { ok: true, body: { cancel: async () => {} } };
    },
    ResizeObserver: class {
      constructor(f: () => void) { resized = f; }
      observe() {} disconnect() { observerDisconnected = true; }
    },
    requestAnimationFrame: (f: () => void) => { frames.set(++frameId, f); return frameId; },
    cancelAnimationFrame: (id: number) => frames.delete(id)
  });
  const scene = new THREE.Scene();
  const create = () => new exports.SparkPageViewer(mount, scene, (error: unknown) => errors.push(error));
  return { create, scene, errors, mount, sizes, frames, events,
    resized: () => resized(), setUpdate: (f: () => Promise<void>) => { update = f; },
    tick: () => { const [id, f] = frames.entries().next().value!; frames.delete(id); f(); },
    state: () => ({ disposed, signal, observerDisconnected, updateCount, meshOptions, rendererOptions }) };
}

const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

test("Spark page loads abortable PLY with audited SH/encoding and transformed bounds", async () => {
  const h = harness(), viewer = h.create();
  const rotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
  await viewer.load("/scene.ply", 3, rotation);
  assert.equal(h.state().meshOptions.fileType, "ply");
  assert.equal(h.state().meshOptions.extSplats, true);
  assert.equal(h.state().rendererOptions.accumExtSplats, true);
  assert.equal(h.state().rendererOptions.autoUpdate, false);
  assert.equal(h.state().rendererOptions.enableLod, false);
  assert.equal(h.state().rendererOptions.sortRadial, false);
  assert.ok(viewer.getBoundingBox().min.distanceTo(new THREE.Vector3(-3, 1, 3)) < 1e-12);
  h.mount.clientWidth = 400; h.resized();
  assert.equal(viewer.camera.aspect, 1);
  await viewer.dispose();
  assert.equal(h.state().signal?.aborted, true);
  assert.equal(h.state().observerDisconnected, true);
  assert.equal(h.events.size, 0);
  assert.equal(h.scene.children.length, 0);
});

test("switching during decoding aborts fetch and releases only after initialization settles", async () => {
  const init = deferred(), h = harness({ initialization: init.promise }), viewer = h.create();
  const loading = viewer.load("/scene.ply", 3, null);
  await settle();
  const disposal = viewer.dispose();
  assert.equal(viewer.dispose(), disposal);
  assert.equal(h.state().signal?.aborted, true);
  const before = h.state().disposed;
  await settle();
  assert.equal(h.state().disposed, before);
  init.resolve();
  await Promise.all([loading, disposal]);
  assert.equal(h.state().updateCount, 0);
  assert.equal(h.state().disposed, 6);
});

test("moving camera never overlaps Spark updates and teardown waits for sorting", async () => {
  const h = harness(), viewer = h.create();
  await viewer.load("/scene.ply", 3, null);
  const sort = deferred(); h.setUpdate(() => sort.promise);
  viewer.start(); h.tick(); h.tick();
  assert.equal(h.state().updateCount, 2);
  const disposal = viewer.dispose();
  assert.equal(h.frames.size, 0);
  assert.equal(h.state().disposed, 1);
  sort.resolve(); await disposal;
  assert.equal(h.state().disposed, 6);
});

test("SH mismatch and rendering/context errors stay visible, never silently fall back", async () => {
  const wrong = harness({ sh: 2 }), invalid = wrong.create();
  await assert.rejects(invalid.load("/scene.ply", 3, null), /SH/);
  await invalid.dispose();
  const h = harness(), viewer = h.create();
  await viewer.load("/scene.ply", 3, null);
  h.setUpdate(() => Promise.reject(new Error("sort failed")));
  viewer.start(); h.tick(); await settle();
  assert.equal(h.frames.size, 0);
  assert.match(String(h.errors[0]), /sort failed/);
  h.events.get("webglcontextlost")!({ preventDefault() {} } as Event);
  assert.match(String(h.errors[1]), /WebGL/);
  await viewer.dispose();
});

test("constructor failure also releases the canvas and WebGL context", () => {
  const h = harness({ constructorError: true });
  assert.throws(h.create, /initialization failed/);
  assert.equal(h.state().disposed, 4);
});
