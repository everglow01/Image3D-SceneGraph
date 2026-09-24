import * as THREE from "three";
import { SparkRenderer, SplatMesh, SplatFileType } from "@sparkjsdev/spark";
import { P0AlphaMask } from "./localGaussianP0Mask.ts";
import { captureP0Source, p0FullMask, p0IdentityFixture, P0_PLY_FIELDS, verifyP0FixtureRows } from "./localGaussianP0Source.ts";
import type { P0SelectionRequest } from "./localGaussianP0Selection.ts";
import type { P0WorkerMessage, P0WorkerReply } from "./localGaussianP0Worker.ts";

function requireCheck(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
function maxDifference(a: Uint8Array, b: Uint8Array) {
  requireCheck(a.length === b.length, "P0 像素尺寸不一致");
  let maximum = 0;
  for (let i = 0; i < a.length; i++) if (i % 4 !== 3) maximum = Math.max(maximum, Math.abs(a[i] - b[i]));
  return maximum;
}
async function sha256(bytes: Uint8Array) {
  requireCheck(globalThis.crypto?.subtle, "P0 探针需要现有 localhost 或 HTTPS 页面的 Web Crypto；不要关闭浏览器安全限制");
  const digest = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2, "0")).join("");
}
function workerCall(worker: Worker, message: P0WorkerMessage, signal?: AbortSignal, transfer: Transferable[] = []) {
  return new Promise<P0WorkerReply>((resolve, reject) => {
    const cleanup = () => { clearTimeout(timer); worker.removeEventListener("message", received); worker.removeEventListener("error", failed); signal?.removeEventListener("abort", aborted); };
    const received = (event: MessageEvent<P0WorkerReply>) => {
      cleanup();
      if (event.data.type === "error") reject(new Error(event.data.message)); else resolve(event.data);
    };
    const failed = (event: ErrorEvent) => { cleanup(); reject(new Error(event.message)); };
    const aborted = () => { cleanup(); worker.postMessage({ type: "cancel" }); reject(new Error("P0 浏览探针已取消")); };
    const timer = setTimeout(() => { cleanup(); worker.postMessage({ type: "cancel" }); reject(new Error("P0 Worker 响应超时")); }, 5000);
    worker.addEventListener("message", received); worker.addEventListener("error", failed);
    signal?.addEventListener("abort", aborted, { once: true });
    if (signal?.aborted) { aborted(); return; }
    try { worker.postMessage(message, transfer); } catch (error) { cleanup(); reject(error); }
  });
}

// Explicit synthetic probe only. Importing this module neither mounts a viewer nor starts a worker.
export async function runLocalGaussianP0Probe(mount: HTMLElement, signal?: AbortSignal) {
  requireCheck(mount instanceof HTMLElement && mount.childNodes.length === 0, "P0 需要独立的空容器，不能占用产品画布");
  signal?.throwIfAborted();
  const fixture = p0IdentityFixture(), sourceSha256 = await sha256(fixture.bytes);
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  let spark: SparkRenderer | undefined, mesh: SplatMesh | undefined, mask: P0AlphaMask | undefined, worker: Worker | undefined;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  const shaderErrors: string[] = [];
  const viewReports: Record<string, unknown>[] = [];
  const count = fixture.count;
  try {
    renderer.setPixelRatio(1); renderer.setSize(256, 256); renderer.setClearColor(0, 1);
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.debug.onShaderError = () => { shaderErrors.push("P0 shader 编译或链接失败"); };
    mount.appendChild(renderer.domElement);
    const gl = renderer.getContext(), debug = gl.getExtension("WEBGL_debug_renderer_info");
    const gpu = debug ? String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)) : "";
    requireCheck(/NVIDIA/i.test(gpu) && /4060/.test(gpu) && !/SwiftShader|llvmpipe|software/i.test(gpu), `P0 必须实测 RTX 4060 硬件渲染，当前：${gpu || "不可识别"}`);
    spark = new SparkRenderer({ renderer, autoUpdate: false, enableLod: false, enableDriveLod: false,
      enableLodFetching: false, accumExtSplats: true, sortRadial: false, minSortIntervalMs: 0,
      preBlurAmount: 0.3, blurAmount: 0, focalAdjustment: 1, minAlpha: 1 / 255, maxStdDev: 3, maxPixelRadius: 1e10 });
    mesh = new SplatMesh({ fileBytes: fixture.bytes, fileType: SplatFileType.PLY, extSplats: true, lod: false, enableLod: false });
    await mesh.initialized; signal?.throwIfAborted();
    mesh.maxSh = 3; mesh.extSplats?.setMaxSh(3); mesh.updateGenerator();
    scene.add(spark, mesh);
    const render = async () => {
      signal?.throwIfAborted();
      scene.updateMatrixWorld(true); camera.updateMatrixWorld(true);
      renderer.getDrawingBufferSize(spark!.renderSize);
      await spark!.update({ scene, camera });
      signal?.throwIfAborted();
      requireCheck(!spark!.sorting && !spark!.sortDirty && mesh!.numSplats === count, "P0 排序未完成或源数量变化");
      renderer.render(scene, camera);
      requireCheck(!shaderErrors.length && !gl.isContextLost() && gl.getError() === gl.NO_ERROR, "P0 着色器或 WebGL 检查失败");
      const pixels = new Uint8Array(256 * 256 * 4);
      gl.readPixels(0, 0, 256, 256, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      requireCheck(gl.getError() === gl.NO_ERROR, "P0 像素读取失败");
      return pixels;
    };
    const source = await captureP0Source(mesh, sourceSha256, signal);
    verifyP0FixtureRows(source, fixture.expected);
    worker = new Worker(new URL("./gaussianSelection.worker.ts", import.meta.url), { type: "module" });
    const ready = await workerCall(worker, { type: "source", source, modelGeneration: 1 }, signal, [source.geometry.buffer]);
    requireCheck(ready.type === "ready" && ready.sourceSha256 === sourceSha256, "P0 Worker 源身份不一致");
    for (const [view, angle] of [0, 0.18].entries()) {
      camera.rotation.y = angle;
      mesh.rotation.z = view * 0.27;
      const baseline = await render(), repeat = await render();
      mesh.maxSh = 0; mesh.extSplats!.setMaxSh(0); mesh.updateGenerator();
      const withoutSh = await render();
      requireCheck(maxDifference(baseline, withoutSh) > 1, "P0 非零 SH3 没有影响画面");
      mesh.maxSh = 3; mesh.extSplats!.setMaxSh(3); mesh.updateGenerator();
      await render();
      requireCheck(baseline.some((v, i) => i % 4 !== 3 && v > 20), "P0 合成基线为空白");
      const noise = maxDifference(baseline, repeat);
      requireCheck(noise <= 1, "P0 基线重复渲染尚未稳定");
      const ext = mesh.extSplats!;
      const attributes = [...ext.extArrays, ...["sh1", "sh2", "sh3a", "sh3b"].map(k => ext.extra[k])];
      requireCheck(attributes.every(a => a instanceof Uint32Array), "P0 缺少完整 SH3 属性数组");
      const before = await Promise.all((attributes as Uint32Array[]).map(a => sha256(new Uint8Array(a.buffer, a.byteOffset, a.byteLength))));
      const target = new THREE.Vector3().fromArray(fixture.expected, 12 * 11).applyMatrix4(mesh.matrixWorld).project(camera);
      const x = 128 * (1 + target.x), y = 128 * (1 - target.y);
      const request: P0SelectionRequest = { sourceSha256, modelGeneration: 1, cameraGeneration: view + 1, sequence: view + 1,
        width: 256, height: 256, modelToView: new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld).elements,
        projection: camera.projectionMatrix.elements.slice(), polygon: [[x - 2, y - 2], [x + 2, y - 2], [x + 2, y + 2], [x - 2, y + 2]],
        visible: p0FullMask(count), layerTolerance: 0.02 };
      const selected = await workerCall(worker, { type: "select", request }, signal);
      requireCheck(selected.type === "result" && selected.result.cameraGeneration === view + 1 &&
        selected.result.sequence === view + 1 && selected.result.sourceSha256 === sourceSha256, "P0 选择响应已过期");
      requireCheck(selected.result.selectedCount > 0, "P0 合成目标未形成可信选集");
      mask = new P0AlphaMask(mesh);
      const retained = await render();
      requireCheck(maxDifference(baseline, retained) <= 1, "P0 全保留掩码改变了原画面");
      const onlyTarget = new Uint8Array(Math.ceil(count / 8)); onlyTarget[12 >> 3] |= 1 << (12 & 7);
      mask.update(onlyTarget);
      const isolatedTarget = await render();
      const rowBytes = P0_PLY_FIELDS.length * 4, oldHeaderLength = fixture.bytes.length - count * rowBytes;
      const header = new TextEncoder().encode(new TextDecoder().decode(fixture.bytes.subarray(0, oldHeaderLength)).replace(`element vertex ${count}`, "element vertex 1"));
      const referenceBytes = new Uint8Array(header.length + rowBytes);
      referenceBytes.set(header); referenceBytes.set(fixture.bytes.subarray(oldHeaderLength + 12 * rowBytes, oldHeaderLength + 13 * rowBytes), header.length);
      const reference = new SplatMesh({ fileBytes: referenceBytes, fileType: SplatFileType.PLY, extSplats: true, lod: false, enableLod: false });
      try {
        await reference.initialized;
        reference.maxSh = 3; reference.extSplats!.setMaxSh(3); reference.quaternion.copy(mesh.quaternion); reference.updateGenerator();
        scene.remove(mesh); scene.add(reference);
        const expectedTarget = await render();
        requireCheck(maxDifference(isolatedTarget, expectedTarget) <= 1, "P0 GPU掩码索引不对应原始第12行");
      } finally { scene.remove(reference); reference.dispose(); scene.add(mesh); }
      const visible = p0FullMask(count);
      selected.result.selected.forEach((v, i) => { visible[i] &= ~v; });
      mask.update(visible);
      const deleted = await render();
      requireCheck(maxDifference(retained, deleted) > 5, "P0 隐藏没有改变目标画面");
      mask.update(new Uint8Array(visible.length));
      const empty = await render();
      requireCheck(empty.every((v, i) => i % 4 === 3 || v === 0), "P0 全隐藏后仍有高斯贡献");
      mask.update(p0FullMask(count));
      const restored = await render();
      const restoreDifference = maxDifference(baseline, restored);
      requireCheck(restoreDifference <= 1, "P0 撤销未恢复原画面");
      mask.dispose(); mask = undefined;
      await render();
      verifyP0FixtureRows(await captureP0Source(mesh, sourceSha256, signal), fixture.expected);
      const after = await Promise.all((attributes as Uint32Array[]).map(a => sha256(new Uint8Array(a.buffer, a.byteOffset, a.byteLength))));
      requireCheck(before.every((hash, i) => hash === after[i]), "P0 修改了原始几何／SH数组");
      viewReports.push({ view, sourceRowsChecked: count, selectedCount: selected.result.selectedCount,
        selectedIds: Array.from({ length: count }, (_, id) => id).filter(id => selected.result.selected[id >> 3] & (1 << (id & 7))),
        selectMs: selected.result.elapsedMs, baselineNoise: noise, restoreDifference, attributesUnchanged: true });
    }
    requireCheck(sourceSha256 === await sha256(fixture.bytes), "P0 源PLY被改写");
    return { status: "passed", profile: "local_gaussian_p0_synthetic_v1", gpu, sourceSha256, count,
      sh: 3, lod: false, views: viewReports, realModelAcceptance: "not_run", productionEditor: "not_enabled" };
  } finally {
    worker?.terminate(); mask?.dispose(); mesh?.dispose(); spark?.dispose(); scene.clear();
    renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
  }
}
