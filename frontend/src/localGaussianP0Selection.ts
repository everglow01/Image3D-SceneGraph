import * as THREE from "three";
import { P0_STRIDE, p0Visible, validateP0Mask, validateP0Source, type P0Source } from "./localGaussianP0Source.ts";

export type P0SelectionRequest = {
  sourceSha256: string;
  modelGeneration: number;
  cameraGeneration: number;
  sequence: number;
  width: number;
  height: number;
  modelToView: number[];
  projection: number[];
  polygon: [number, number][];
  visible: Uint8Array;
  layerTolerance: number;
};
export type P0SelectionResult = {
  sourceSha256: string;
  modelGeneration: number;
  cameraGeneration: number;
  sequence: number;
  selected: Uint8Array;
  selectedCount: number;
  confidentPixels: number;
  candidates: number;
  contributions: number;
  elapsedMs: number;
};

export const P0_SELECTION_LIMITS = Object.freeze({ maxRoiPixels: 128 * 128, maxCandidates: 100_000,
  maxVisits: 8_000_000, maxMs: 2000, tileSize: 16 });
const MIN_ALPHA = 1 / 255;

export function p0PolygonContains(polygon: [number, number][], x: number, y: number) {
  let inside = false;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i], b = polygon[(i + 1) % polygon.length];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    if (Math.abs((x - a[0]) * dy - (y - a[1]) * dx) < 1e-7 &&
        x >= Math.min(a[0], b[0]) && x <= Math.max(a[0], b[0]) &&
        y >= Math.min(a[1], b[1]) && y <= Math.max(a[1], b[1])) return true;
    if ((a[1] > y) !== (b[1] > y) && x < a[0] + (y - a[1]) * dx / dy) inside = !inside;
  }
  return inside;
}

export function validateP0Selection(source: P0Source, r: P0SelectionRequest) {
  validateP0Source(source);
  validateP0Mask(r.visible, source.count);
  if (r.sourceSha256 !== source.sha256 ||
      ![r.modelGeneration, r.cameraGeneration, r.sequence].every(v => Number.isSafeInteger(v) && v >= 0) ||
      ![r.width, r.height].every(v => Number.isSafeInteger(v) && v >= 1 && v <= 4096) ||
      !Number.isFinite(r.layerTolerance) || r.layerTolerance < 0 || r.layerTolerance > 0.2 ||
      ![r.modelToView, r.projection].every(m => Array.isArray(m) && m.length === 16 && m.every(Number.isFinite))) {
    throw new Error("P0 选择身份、相机或参数不合法");
  }
  const m = new THREE.Matrix4().fromArray(r.modelToView), e = m.elements;
  if (Math.abs(m.determinant() - 1) > 1e-5 || e[3] !== 0 || e[7] !== 0 || e[11] !== 0 || e[15] !== 1) {
    throw new Error("P0 仅支持刚体模型相机变换");
  }
  for (let i = 0; i < 3; i++) for (let j = i; j < 3; j++) {
    const dot = e[i * 4] * e[j * 4] + e[i * 4 + 1] * e[j * 4 + 1] + e[i * 4 + 2] * e[j * 4 + 2];
    if (Math.abs(dot - (i === j ? 1 : 0)) > 1e-5) throw new Error("P0 不支持缩放或剪切相机");
  }
  const p = r.projection;
  if (p[0] <= 0 || p[5] <= 0 || p[10] >= -1 || p[14] >= 0 || p[11] !== -1 || p[15] !== 0 ||
      [1, 2, 3, 4, 6, 7, 12, 13].some(i => p[i] !== 0)) throw new Error("P0 仅支持有限无斜切透视投影");
  if (!Array.isArray(r.polygon) || r.polygon.length < 3 || r.polygon.length > 128 ||
      r.polygon.some(p => !Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite) ||
        p[0] < 0 || p[1] < 0 || p[0] > r.width || p[1] > r.height)) throw new Error("P0 多边形不合法");
  let area = 0;
  r.polygon.forEach((a, i) => { const b = r.polygon[(i + 1) % r.polygon.length]; area += a[0] * b[1] - b[0] * a[1]; });
  if (Math.abs(area) < 2) throw new Error("P0 选区面积不足一个像素");
  const left = Math.floor(Math.min(...r.polygon.map(p => p[0]))), top = Math.floor(Math.min(...r.polygon.map(p => p[1])));
  const right = Math.ceil(Math.max(...r.polygon.map(p => p[0]))), bottom = Math.ceil(Math.max(...r.polygon.map(p => p[1])));
  if ((right - left) * (bottom - top) > P0_SELECTION_LIMITS.maxRoiPixels) throw new Error("P0 选区超过 128×128 像素面积预算，请缩小选区");
  return { left, top, right, bottom };
}

// Same layer state as visible_selection.FrontLayer, fed in front-to-back order per pixel.
export class P0FrontLayer {
  transmittance = 1;
  depth = Infinity;
  coverage = 0;
  done = false;
  ids: number[] = [];
  readonly tolerance: number;
  constructor(tolerance: number) { this.tolerance = tolerance; }
  add(id: number, depth: number, alpha: number) {
    if (this.done) return;
    alpha = Math.min(alpha, 0.999);
    const weight = this.transmittance * alpha;
    this.transmittance *= 1 - alpha;
    if (weight < MIN_ALPHA) { if (this.transmittance <= 1e-4) this.done = true; return; }
    if (depth > this.depth + Math.max(0.001, this.depth * this.tolerance)) {
      if (this.coverage >= 0.05) { this.done = true; return; }
      this.ids = []; this.coverage = 0; this.depth = depth;
    }
    if (!Number.isFinite(this.depth)) this.depth = depth;
    this.coverage += weight;
    this.ids.push(id);
    if (this.coverage >= 0.5 || this.transmittance <= 1e-4) this.done = true;
  }
  result() { return this.coverage >= 0.5 ? this.ids : []; }
}

export async function selectP0Surface(source: P0Source, r: P0SelectionRequest,
  cancelled: () => boolean = () => false): Promise<P0SelectionResult> {
  const roi = validateP0Selection(source, r), start = performance.now();
  const limits = P0_SELECTION_LIMITS;
  const check = () => {
    if (cancelled()) throw new Error("P0 选择已取消");
    if (performance.now() - start > limits.maxMs) throw new Error("P0 选择超时，未应用任何结果");
  };
  const yieldTask = async () => { await new Promise<void>(resolve => setTimeout(resolve, 0)); check(); };
  // x, y, depth, inverse covariance xx/xy/yy, radius x/y, opacity; no SH or RGB copy.
  const projected = new Float64Array(Math.min(source.count, limits.maxCandidates) * 9);
  const sourceIds = new Uint32Array(Math.min(source.count, limits.maxCandidates));
  let count = 0;
  const m = new THREE.Matrix4().fromArray(r.modelToView), p = r.projection;
  const position = new THREE.Vector3(), scale = new THREE.Vector3(), q = new THREE.Quaternion();
  const transform = new THREE.Matrix4();
  const g = source.geometry, fx = r.width * p[0] / 2, fy = r.height * p[5] / 2;
  for (let id = 0; id < source.count; id++) {
    if (id % 2048 === 0) await yieldTask();
    if (!p0Visible(r.visible, id)) continue;
    const b = id * P0_STRIDE;
    for (let k = 0; k < P0_STRIDE; k++) if (!Number.isFinite(g[b + k])) throw new Error("P0 几何包含非有限值");
    position.set(g[b], g[b + 1], g[b + 2]); scale.set(g[b + 3], g[b + 4], g[b + 5]);
    q.set(g[b + 6], g[b + 7], g[b + 8], g[b + 9]);
    const alpha = g[b + 10];
    if (scale.x <= 0 || scale.y <= 0 || scale.z <= 0 || alpha < 0 || alpha > 1 || Math.abs(q.lengthSq() - 1) > 0.01) {
      throw new Error("P0 不支持退化高斯");
    }
    if (alpha < MIN_ALPHA) continue;
    transform.compose(position, q, scale).premultiply(m);
    const t = transform.elements, x = t[12], y = t[13], z = t[14], w = -z;
    const clipX = p[0] * x + p[8] * z, clipY = p[5] * y + p[9] * z, clipZ = p[10] * z + p[14];
    if (z >= 0 || Math.abs(clipZ) >= w || Math.abs(clipX) > 1.4 * w || Math.abs(clipY) > 1.4 * w) continue;
    let a = 0.3, d = 0.3, cross = 0;
    for (let k = 0; k < 3; k++) {
      const dx = fx / z * t[k * 4] - fx * x / (z * z) * t[k * 4 + 2];
      const dy = fy / z * t[k * 4 + 1] - fy * y / (z * z) * t[k * 4 + 2];
      a += dx * dx; d += dy * dy; cross -= dx * dy;
    }
    const det = a * d - cross * cross, avg = (a + d) / 2;
    const delta = Math.sqrt(Math.max(0, avg * avg - det)), e1 = avg + delta, e2 = avg - delta;
    if (![a, d, cross, det, e1, e2].every(Number.isFinite) || e2 <= 0) throw new Error("P0 投影协方差退化");
    let ex = a >= d ? 1 : 0, ey = a >= d ? 0 : 1;
    if (Math.abs(cross) > 0.001) { const length = Math.hypot(cross, e1 - a); ex = cross / length; ey = (e1 - a) / length; }
    const v1 = Math.min(1e10, 3 * Math.sqrt(e1)) ** 2 / 9;
    const v2 = Math.min(1e10, 3 * Math.sqrt(e2)) ** 2 / 9;
    a = ex * ex * v1 + ey * ey * v2; d = ey * ey * v1 + ex * ex * v2;
    cross = ex * ey * (v1 - v2);
    const determinant = a * d - cross * cross;
    if (!Number.isFinite(determinant) || determinant <= 0) throw new Error("P0 截断后投影协方差退化");
    const cx = r.width * (0.5 + clipX / w / 2), cy = r.height * (0.5 - clipY / w / 2);
    const rx = 3 * Math.sqrt(a), ry = 3 * Math.sqrt(d);
    if (cx + rx < roi.left || cx - rx > roi.right || cy + ry < roi.top || cy - ry > roi.bottom) continue;
    if (count >= limits.maxCandidates) throw new Error("P0 候选预算超限，未应用任何结果");
    projected.set([cx, cy, w, d / determinant, -cross / determinant, a / determinant, rx, ry, alpha], count * 9);
    sourceIds[count++] = id;
  }
  const order = Uint32Array.from({ length: count }, (_, i) => i);
  order.sort((a, b) => projected[a * 9 + 2] - projected[b * 9 + 2] || sourceIds[a] - sourceIds[b]);
  await yieldTask();
  const selected = new Uint8Array(Math.ceil(source.count / 8));
  let visits = 0, contributions = 0, confidentPixels = 0;
  for (let ty = roi.top; ty < roi.bottom; ty += limits.tileSize) {
    for (let tx = roi.left; tx < roi.right; tx += limits.tileSize) {
      await yieldTask();
      const tileIds: number[] = [];
      for (let i = 0; i < count; i++) {
        if (++visits > limits.maxVisits) throw new Error("P0 贡献预算超限，未应用任何结果");
        if (i % 4096 === 0) await yieldTask();
        const k = order[i] * 9;
        if (projected[k] + projected[k + 6] >= tx && projected[k] - projected[k + 6] < tx + limits.tileSize &&
            projected[k + 1] + projected[k + 7] >= ty && projected[k + 1] - projected[k + 7] < ty + limits.tileSize) tileIds.push(order[i]);
      }
      for (let y = ty; y < Math.min(roi.bottom, ty + limits.tileSize); y++) {
        for (let x = tx; x < Math.min(roi.right, tx + limits.tileSize); x++) {
          if (!p0PolygonContains(r.polygon, x + 0.5, y + 0.5)) continue;
          const layer = new P0FrontLayer(r.layerTolerance);
          for (const candidate of tileIds) {
            if (++visits > limits.maxVisits) throw new Error("P0 贡献预算超限，未应用任何结果");
            if (visits % 4096 === 0) await yieldTask();
            const k = candidate * 9, dx = x + 0.5 - projected[k], dy = y + 0.5 - projected[k + 1];
            const z2 = projected[k + 3] * dx * dx + 2 * projected[k + 4] * dx * dy + projected[k + 5] * dy * dy;
            if (z2 < 0 || z2 > 9) continue;
            const alpha = projected[k + 8] * Math.exp(-0.5 * z2);
            if (alpha < MIN_ALPHA) continue;
            contributions++;
            layer.add(sourceIds[candidate], projected[k + 2], alpha);
            if (layer.done) break;
          }
          const ids = layer.result();
          if (ids.length) confidentPixels++;
          for (const id of ids) selected[id >> 3] |= 1 << (id & 7);
        }
      }
    }
  }
  check();
  let selectedCount = 0;
  for (let i = 0; i < selected.length; i++) { let v = selected[i]; while (v) { selectedCount++; v &= v - 1; } }
  check();
  return { sourceSha256: source.sha256, modelGeneration: r.modelGeneration, cameraGeneration: r.cameraGeneration,
    sequence: r.sequence, selected, selectedCount, confidentPixels, candidates: count, contributions, elapsedMs: performance.now() - start };
}
