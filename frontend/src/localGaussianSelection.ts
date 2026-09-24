import { p0PolygonContains, selectP0Surface, validateP0Selection, type P0SelectionRequest, type P0SelectionResult } from "./localGaussianP0Selection.ts";
import { P0_STRIDE, p0Visible, type P0Source } from "./localGaussianP0Source.ts";

export type LocalSelectionMode = "surface" | "depth" | "through" | "box";
export type LocalSelectionRequest = P0SelectionRequest & {
  mode: LocalSelectionMode;
  depthRange?: [number, number];
  box?: { min: [number, number, number]; max: [number, number, number] };
};
export const LOCAL_SURFACE_PIXELS = 1024 * 1024;

export async function selectLocalGaussians(source: P0Source, r: LocalSelectionRequest,
  cancelled: () => boolean = () => false): Promise<P0SelectionResult> {
  if (r.mode === "surface") return selectP0Surface(source, r, cancelled, LOCAL_SURFACE_PIXELS);
  if (!["depth", "through", "box"].includes(r.mode)) throw new Error("未知本地选择范围");
  validateP0Selection(source, r, 4096 * 4096);
  if (r.mode === "depth" && (!Array.isArray(r.depthRange) || r.depthRange.length !== 2 ||
      !r.depthRange.every(Number.isFinite) || r.depthRange[0] < 0.01 || r.depthRange[0] >= r.depthRange[1] || r.depthRange[1] > 1e6)) {
    throw new Error("深度范围须满足 0.01 ≤ 近端 < 远端 ≤ 1e6");
  }
  if (r.mode === "box" && (!r.box || ![r.box.min, r.box.max].every(v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)) ||
      r.box.min.some((v, i) => v >= r.box!.max[i]))) throw new Error("三维盒须为源坐标中有序的有限边界");
  const start = performance.now(), channel = new MessageChannel();
  const check = () => {
    if (cancelled()) throw new Error("选择已取消");
    if (performance.now() - start > 2000) throw new Error("选择超过2秒预算，未应用任何结果");
  };
  const yieldTask = () => new Promise<void>((resolve, reject) => {
    channel.port1.onmessage = () => { try { check(); resolve(); } catch (e) { reject(e); } };
    channel.port2.postMessage(0);
  });
  try {
    const selected = new Uint8Array(Math.ceil(source.count / 8));
    const m = r.modelToView, p = r.projection, g = source.geometry;
    const [near, far] = r.mode === "depth" ? r.depthRange! : [0.01, 1e6];
    let selectedCount = 0, visits = 0;
    for (let id = 0; id < source.count; id++) {
      if (id % 2048 === 0) await yieldTask();
      if (!p0Visible(r.visible, id)) continue;
      const b = id * P0_STRIDE, x = g[b], y = g[b + 1], z = g[b + 2];
      if (![x, y, z].every(Number.isFinite)) throw new Error("源中心包含非有限值");
      let inside = false;
      if (r.mode === "box") {
        inside = [x, y, z].every((v, k) => v >= r.box!.min[k] && v <= r.box!.max[k]);
      } else {
        const vx = m[0] * x + m[4] * y + m[8] * z + m[12];
        const vy = m[1] * x + m[5] * y + m[9] * z + m[13];
        const vz = m[2] * x + m[6] * y + m[10] * z + m[14], depth = -vz;
        if (depth < near || depth > far) continue;
        visits += r.polygon.length;
        if (visits > 32_000_000) throw new Error("多边形访问预算超限，未应用任何结果");
        inside = p0PolygonContains(r.polygon, r.width * (0.5 + (p[0] * vx / depth - p[8]) / 2),
          r.height * (0.5 - (p[5] * vy / depth - p[9]) / 2));
      }
      if (inside) { selected[id >> 3] |= 1 << (id & 7); selectedCount++; }
    }
    check();
    return { sourceSha256: source.sha256, modelGeneration: r.modelGeneration, cameraGeneration: r.cameraGeneration,
      sequence: r.sequence, selected, selectedCount, confidentPixels: 0, candidates: selectedCount, contributions: 0,
      elapsedMs: performance.now() - start };
  } finally { channel.port1.close(); channel.port2.close(); }
}
