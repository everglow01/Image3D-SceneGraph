import * as THREE from "three";
import { SparkRenderer, SplatMesh, SplatFileType } from "@sparkjsdev/spark";
import { P0AlphaMask } from "./localGaussianP0Mask.ts";
import { captureP0Source, p0FullMask, P0_PLY_FIELDS, verifyP0EncodedRow } from "./localGaussianP0Source.ts";
import type { P0SelectionRequest } from "./localGaussianP0Selection.ts";
import type { P0WorkerReply } from "./localGaussianP0Worker.ts";

const row = (z: number, alpha: number, x = 0, y = 0, scale = 0.2) => [x, y, z, scale, scale, scale, 0, 0, 0, 1, alpha];
export const P0_OCCLUSION_CASES = [
  { name: "opaque-front", rows: [row(-4, 0.95), row(-2, 0.95)], selected: [1] },
  { name: "weak-fog", rows: [row(-3, 0.9), row(-1, 0.02)], selected: [0] },
  { name: "uncertain-front", rows: [row(-3, 0.9), row(-1, 0.2)], selected: [] },
  { name: "substantive-front", rows: [row(-3, 0.9), row(-1, 0.06)], selected: [] },
  { name: "large-edge", rows: [row(-4, 0.95), row(-2, 0.95, 0.15, 0, 0.5)], selected: [1] },
  { name: "near-clip", rows: [row(-0.005, 0.95), row(2, 0.95), row(-2, 0.95)], selected: [2] },
  { name: "rotated-axis", rows: [row(-2, 0.99, 0, 0.5, 0.1), row(-2, 0.99, 0, -0.5, 0.1)], selected: [1], rotation: Math.PI, pixel: [32, 18] },
  { name: "anisotropic-edge", rows: [[0, 0, -2, 0.6, 0.02, 0.02, 0, 0, 0, 1, 0.95]], selected: [0], pixel: [37, 32] }
];

export function p0OcclusionPly(rows: number[][]) {
  const header = new TextEncoder().encode(["ply", "format binary_little_endian 1.0", `element vertex ${rows.length}`,
    ...P0_PLY_FIELDS.map(k => `property float ${k}`), "end_header", ""].join("\n"));
  const bytes = new Uint8Array(header.length + rows.length * 248); bytes.set(header);
  const view = new DataView(bytes.buffer, header.length);
  rows.forEach((r, id) => {
    const values: Record<string, number> = { x: r[0], y: r[1], z: r[2],
      scale_0: Math.log(r[3]), scale_1: Math.log(r[4]), scale_2: Math.log(r[5]),
      rot_0: r[9], rot_1: r[6], rot_2: r[7], rot_3: r[8], opacity: Math.log(r[10] / (1 - r[10])),
      f_dc_0: 0.4, f_dc_1: -0.2, f_dc_2: 0.1, f_rest_40: 0.2 };
    P0_PLY_FIELDS.forEach((key, k) => view.setFloat32(id * 248 + k * 4, values[key] ?? 0, true));
  });
  return bytes;
}

// Explicit GPU fixture audit, not a product editor or semantic background guarantee.
export async function runLocalGaussianP0OcclusionProbe(mount: HTMLElement) {
  const check = (value: unknown, message: string) => { if (!value) throw new Error(message); };
  check(mount instanceof HTMLElement && mount.childNodes.length === 0, "P0 需要独立空容器");
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  let spark: SparkRenderer | undefined, mesh: SplatMesh | undefined, mask: P0AlphaMask | undefined, worker: Worker | undefined;
  const references: SplatMesh[] = [], reports = [];
  const difference = (a: Uint8Array, b: Uint8Array) => a.reduce((max, v, i) => i % 4 === 3 ? max : Math.max(max, Math.abs(v - b[i])), 0);
  try {
    renderer.setPixelRatio(1); renderer.setSize(65, 65); renderer.setClearColor(0, 1);
    renderer.toneMapping = THREE.NoToneMapping; mount.appendChild(renderer.domElement);
    const gl = renderer.getContext(), debug = gl.getExtension("WEBGL_debug_renderer_info");
    const gpu = debug ? String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)) : "";
    check(/NVIDIA.*4060/.test(gpu) && !/SwiftShader|llvmpipe/i.test(gpu), "P0 必须使用真实RTX4060");
    let shaderError = false; renderer.debug.onShaderError = () => { shaderError = true; };
    spark = new SparkRenderer({ renderer, autoUpdate: false, enableLod: false, enableDriveLod: false, enableLodFetching: false,
      accumExtSplats: true, sortRadial: false, minSortIntervalMs: 0, preBlurAmount: 0.3, blurAmount: 0,
      focalAdjustment: 1, minAlpha: 1 / 255, maxStdDev: 3, maxPixelRadius: 1e10 });
    scene.add(spark);
    const load = async (bytes: Uint8Array) => {
      const result = new SplatMesh({ fileBytes: bytes, fileType: SplatFileType.PLY, extSplats: true, lod: false, enableLod: false });
      references.push(result); await result.initialized;
      result.maxSh = 3; result.extSplats!.setMaxSh(3); result.updateGenerator(); return result;
    };
    const draw = async () => {
      scene.updateMatrixWorld(true); camera.updateMatrixWorld(true); renderer.getDrawingBufferSize(spark!.renderSize);
      await spark!.update({ scene, camera }); check(!spark!.sorting && !spark!.sortDirty, "P0 排序未完成");
      renderer.render(scene, camera);
      const pixels = new Uint8Array(65 * 65 * 4); gl.readPixels(0, 0, 65, 65, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      check(!shaderError && !gl.isContextLost() && gl.getError() === gl.NO_ERROR, "P0 GPU检查失败"); return pixels;
    };
    worker = new Worker(new URL("./gaussianSelection.worker.ts", import.meta.url), { type: "module" });
    const call = (message: unknown, transfer: Transferable[] = []) => new Promise<P0WorkerReply>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("P0 Worker超时")), 5000);
      worker!.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
      worker!.onmessage = event => { clearTimeout(timer); resolve(event.data); }; worker!.postMessage(message, transfer);
    });
    for (const [generation, fixture] of P0_OCCLUSION_CASES.entries()) {
      const bytes = p0OcclusionPly(fixture.rows);
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), v => v.toString(16).padStart(2, "0")).join("");
      mesh = await load(bytes); mesh.rotation.z = fixture.rotation ?? 0; scene.add(mesh); camera.rotation.y = 0;
      const baseline = await draw(), noise = difference(baseline, await draw()); check(noise <= 1, "P0 基线不稳定");
      const source = await captureP0Source(mesh, hash);
      check((await call({ type: "source", source, modelGeneration: generation }, [source.geometry.buffer])).type === "ready", "P0 Worker未就绪");
      const [x, y] = fixture.pixel ?? [32, 32];
      const request: P0SelectionRequest = { sourceSha256: hash, modelGeneration: generation, cameraGeneration: 0, sequence: 1,
        width: 65, height: 65, modelToView: new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld).elements,
        projection: camera.projectionMatrix.elements.slice(), polygon: [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]], visible: p0FullMask(source.count), layerTolerance: 0.02 };
      const response = await call({ type: "select", request });
      if (response.type !== "result") throw new Error(`P0 选择失败：${JSON.stringify(response)}`);
      const ids = fixture.rows.map((_, id) => id).filter(id => response.result.selected[id >> 3] & (1 << (id & 7)));
      check(JSON.stringify(ids) === JSON.stringify(fixture.selected), `P0 ${fixture.name} 误选：${ids}`);
      for (let id = 0; id < source.count; id++) {
        const ref = await load(p0OcclusionPly([fixture.rows[id]])); verifyP0EncodedRow(mesh.extSplats!, id, ref.extSplats!);
      }
      const kept = fixture.rows.filter((_, id) => !ids.includes(id));
      const reference = kept.length ? await load(p0OcclusionPly(kept)) : undefined;
      if (reference) reference.quaternion.copy(mesh.quaternion);
      mask = new P0AlphaMask(mesh); check(difference(baseline, await draw()) <= 1, "P0 全保留不一致");
      const visible = p0FullMask(source.count); ids.forEach(id => { visible[id >> 3] &= ~(1 << (id & 7)); });
      const views = [];
      for (const yaw of [0, 0.18]) {
        camera.rotation.y = yaw; mask.update(p0FullMask(source.count)); const before = await draw();
        mask.update(visible); const hidden = await draw();
        scene.remove(mesh); if (reference) scene.add(reference); const expected = await draw();
        if (reference) scene.remove(reference); scene.add(mesh);
        const referenceDifference = difference(hidden, expected); check(referenceDifference <= 1, `P0 ${fixture.name} 跨视角删错源行`);
        mask.update(p0FullMask(source.count)); const restoreDifference = difference(before, await draw());
        check(restoreDifference <= 1, "P0 撤销不一致"); views.push({ yaw, referenceDifference, restoreDifference });
      }
      reports.push({ name: fixture.name, selected: ids, noise, views });
      mask.dispose(); mask = undefined; scene.remove(mesh); mesh = undefined;
      references.splice(0).forEach(ref => ref.dispose()); await draw();
    }
    return { status: "passed", gpu, cases: reports, productionEditor: "not_enabled" };
  } finally {
    worker?.terminate(); mask?.dispose(); references.forEach(ref => ref.dispose()); spark?.dispose(); scene.clear();
    renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
  }
}
