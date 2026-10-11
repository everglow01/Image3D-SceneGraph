import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as THREE from "three";
import { Capsule } from "three/examples/jsm/math/Capsule.js";
import { Octree } from "three/examples/jsm/math/Octree.js";
import * as walk from "../src/walkNavigation.ts";

const code = ts.transpileModule(readFileSync(new URL("../src/GaussianSplatViewer.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX }
}).outputText + "\nexports.probe = { loadWalkRuntime, installWalkHandlers, uninstallWalkHandlers, stepWalk };";

async function harness(wall = false, navigationDir?: string) {
  const scene = new THREE.Scene(), collision = new THREE.Group();
  collision.add(new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.MeshBasicMaterial()));
  if (wall) {
    const obstacle = new THREE.Mesh(new THREE.BoxGeometry(0.1, 5, 3), new THREE.MeshBasicMaterial());
    obstacle.position.set(1, 0, 1.5); collision.add(obstacle);
  }
  collision.updateMatrixWorld(true);
  let bytes = new Uint8Array([1, 2, 3]).buffer;
  let contract = {
    schema_version: 1, status: "available", coordinate_frame: "normalized", world_units: "arbitrary",
    up: [0, 0, 1], floor: { origin: [0, 0, 0], basis_u: [1, 0, 0], basis_v: [0, 1, 0] },
    boundary: { coordinate_frame: "floor_uv", outer: [[-4, -4], [4, -4], [4, 4], [-4, 4]], holes: [] },
    spawn: { floor_position: [0, 0, 0], eye_position: [0, 0, 1], look_direction: [1, 0, 0], floor_uv: [0, 0] },
    estimated_eye_height: 1,
    player: { capsule_total_height: 1.12, capsule_radius: 0.18, max_step: 0.12, max_slope_degrees: 35 },
    controls: { speed: { default: 0.8, minimum: 0.4, maximum: 1.2 },
      vertical_fov_degrees: { default: 70, minimum: 50, maximum: 90 },
      mouse_sensitivity_radians_per_pixel: { default: 0.002, minimum: 0.0005, maximum: 0.005 },
      pitch_degrees: { minimum: -85, maximum: 85 } },
    provenance: { validation_image_ids_used: [], test_image_ids_used: [] },
    collision: { bytes: 3, triangles: wall ? 14 : 2, sha256: await walk.collisionSha256(bytes) }
  };
  if (navigationDir) {
    contract = JSON.parse(readFileSync(join(navigationDir, "navigation.json"), "utf8"));
    bytes = Uint8Array.from(readFileSync(join(navigationDir, "collision.glb"))).buffer;
  }
  function target() {
    const listeners = new Map<string, Set<Function>>();
    return {
      addEventListener(type: string, handler: Function) { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type)!.add(handler); },
      removeEventListener(type: string, handler: Function) { listeners.get(type)?.delete(handler); },
      emit(type: string, event: any = {}) { for (const handler of listeners.get(type) ?? []) handler(event); }
    };
  }
  const frames = new Map<number, Function>(); let serial = 0;
  const window = { ...target(), requestAnimationFrame(fn: Function) { frames.set(++serial, fn); return serial; },
    cancelAnimationFrame(id: number) { frames.delete(id); }, setTimeout() { return 1; }, clearTimeout() {} };
  const document = { ...target(), pointerLockElement: null as unknown, hidden: false,
    exitPointerLock() { this.pointerLockElement = null; this.emit("pointerlockchange"); } };
  const exports: any = {};
  runInNewContext(code, { exports, window, document, console,
    fetch: async (url: string) => ({ ok: true, json: async () => contract, arrayBuffer: async () => bytes }),
    require(name: string) {
      if (name === "three") return THREE;
      if (name === "./walkNavigation") return walk;
      if (name.endsWith("/Capsule.js")) return { Capsule };
      if (name.endsWith("/Octree.js")) return { Octree };
      if (name.endsWith("/GLTFLoader.js")) return navigationDir ? { GLTFLoader } : { GLTFLoader: class { async parseAsync() { return { scene: collision }; } } };
      return {};
    }
  });
  const api = exports.probe;
  const runtime = await api.loadWalkRuntime("navigation", "collision", scene, new AbortController().signal, null);
  const viewer = { camera: new THREE.PerspectiveCamera(), renderer: { domElement: {} },
    controls: { target: new THREE.Vector3(), enabled: true, enableDamping: true, update() {}, saveState() {} }, forceRenderNextFrame() {} };
  const mode = { current: "orbit" }, settings = { current: walk.defaultWalkSettings(runtime.contract) };
  api.installWalkHandlers(viewer, runtime, mode, settings, () => {}, () => {}, () => {}, () => false);
  const advance = (seconds: number) => {
    for (let i = 0; i < seconds * 120; i++) api.stepWalk(runtime, settings.current, 1 / 120, () => {});
  };
  const key = (code: string, pressed = true) => window.emit(pressed ? "keydown" : "keyup", { code, preventDefault() {} });
  const lock = () => { document.pointerLockElement = viewer.renderer.domElement; document.emit("pointerlockchange"); };
  const close = () => { document.exitPointerLock(); api.uninstallWalkHandlers(runtime); };
  return { runtime, viewer, document, window, frames, mode, key, lock, advance, close };
}

for (const navigationDir of (process.env.IMAGE3D_NAVIGATION_SMOKE_DIRS ?? "").split(":").filter(Boolean)) {
  test(`已发布导航资产的真实 GLB/SHA/出生点及 WASD 物理验收：${navigationDir}`, async () => {
    const h = await harness(false, navigationDir); h.lock();
    const distances: Record<string, number> = {};
    const start = h.runtime.capsule.start.clone(), height = h.runtime.contract.estimated_eye_height;
    for (const key of ["KeyW", "KeyA", "KeyS", "KeyD"]) {
      h.runtime.capsule.copy(h.runtime.spawnCapsule); h.runtime.safeCapsule.copy(h.runtime.spawnCapsule);
      h.key(key); h.advance(2); h.key(key, false);
      const delta = h.runtime.capsule.start.clone().sub(start);
      assert.ok(Math.abs(delta.dot(h.runtime.up)) < height * 1e-5, "玩家不得脱离地板平面");
      const contact = h.runtime.octree.capsuleIntersect(h.runtime.capsule);
      assert.ok(!contact || contact.depth <= height * 1e-6, "玩家不得卡在碰撞体内");
      distances[key] = delta.length() / height;
    }
    assert.ok(Math.max(...Object.values(distances)) > 0.2, "出生点四向全部受阻，不能认为漫游可用");
    console.log(JSON.stringify({ navigationDir, two_second_displacement_in_eye_heights: distances,
      collision_triangles: h.runtime.contract.collision.triangles,
      scope: "实际碰撞资产与产品控制器的CPU物理验收，非WebGL画面或人工浏览器验收" }));
    h.close();
  });
}

test("真实 Octree 地板出生点可加载，WASD 四向移动、抬头不飞行且斜走不加速", async () => {
  const h = await harness(); h.lock();
  const start = h.runtime.capsule.start.clone();
  h.runtime.pitch = 1.4;
  for (const [key, expected] of [["KeyW", [0.8, 0, 0]], ["KeyS", [-0.8, 0, 0]], ["KeyA", [0, 0.8, 0]], ["KeyD", [0, -0.8, 0]]] as const) {
    h.runtime.capsule.copy(h.runtime.spawnCapsule); h.runtime.safeCapsule.copy(h.runtime.spawnCapsule);
    h.key(key); h.advance(1); h.key(key, false);
    assert.ok(h.runtime.capsule.start.clone().sub(start).distanceTo(new THREE.Vector3(...expected)) < 1e-6);
  }
  h.runtime.capsule.copy(h.runtime.spawnCapsule); h.runtime.safeCapsule.copy(h.runtime.spawnCapsule);
  h.key("KeyW"); h.key("KeyD"); h.advance(1);
  assert.ok(Math.abs(h.runtime.capsule.start.distanceTo(start) - 0.8) < 1e-6);
  h.close();
});

test("WASD 碰墙停止并沿墙滑动，不能越界或因碰撞持续升高", async () => {
  const h = await harness(true); h.lock(); h.key("KeyW"); h.advance(3);
  assert.ok(h.runtime.capsule.start.x <= 0.95 - 0.18 + 1e-6);
  const y = h.runtime.capsule.start.y;
  h.key("KeyD"); h.advance(1);
  assert.ok(h.runtime.capsule.start.y < y - 0.4);
  assert.ok(Math.abs(h.runtime.capsule.start.z - 0.182) < 1e-6);
  h.key("KeyW", false); h.advance(12);
  assert.ok(h.runtime.capsule.start.y >= -4);
  assert.ok(!h.runtime.octree.capsuleIntersect(h.runtime.capsule));
  h.close();
});

test("仅锁定当前画布后接收移动，失焦、隐藏及 Esc 清键，重复锁定不叠加动画", async () => {
  const h = await harness();
  h.key("KeyW"); assert.equal(h.runtime.keys.size, 0);
  h.lock(); h.lock(); assert.equal(h.frames.size, 1);
  h.key("KeyW"); assert.equal(h.runtime.keys.size, 1);
  h.window.emit("blur"); assert.equal(h.runtime.keys.size, 0);
  h.key("KeyD"); h.document.hidden = true; h.document.emit("visibilitychange"); assert.equal(h.runtime.keys.size, 0);
  h.document.hidden = false; h.key("KeyA"); h.document.exitPointerLock();
  assert.equal(h.runtime.keys.size, 0); assert.equal(h.mode.current, "orbit"); assert.equal(h.frames.size, 0);
  assert.equal(h.viewer.controls?.enabled, true);
  h.close();
});
