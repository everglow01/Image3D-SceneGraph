import type { ExtSplats, SplatMesh } from "@sparkjsdev/spark";

export const P0_MAX_SPLATS = 3_000_000;
export const P0_STRIDE = 11;
export const P0_PLY_FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
  ...Array.from({ length: 45 }, (_, i) => `f_rest_${i}`), "opacity",
  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];

export type P0Source = {
  sha256: string;
  count: number;
  // Source-row order, not display/sort order: xyz, scale xyz, quaternion xyzw, opacity.
  geometry: Float32Array;
};

export function validateP0Source(source: P0Source) {
  if (!/^[a-f0-9]{64}$/.test(source.sha256) || !Number.isSafeInteger(source.count) ||
      source.count < 1 || source.count > P0_MAX_SPLATS ||
      !(source.geometry instanceof Float32Array) || source.geometry.length !== source.count * P0_STRIDE) {
    throw new Error("P0 源身份、数量或几何数组不合法");
  }
}

export function validateP0Mask(mask: Uint8Array, count: number) {
  if (!(mask instanceof Uint8Array) || mask.length !== Math.ceil(count / 8) ||
      (count % 8 !== 0 && (mask[mask.length - 1] >> (count % 8)) !== 0)) {
    throw new Error("P0 掩码长度或补齐位不合法");
  }
}

export function p0Visible(mask: Uint8Array, id: number) {
  return (mask[id >> 3] & (1 << (id & 7))) !== 0;
}

export function p0FullMask(count: number) {
  if (!Number.isSafeInteger(count) || count < 1 || count > P0_MAX_SPLATS) throw new Error("P0 高斯数量不合法");
  const mask = new Uint8Array(Math.ceil(count / 8)).fill(255);
  if (count % 8) mask[mask.length - 1] = (1 << (count % 8)) - 1;
  return mask;
}

export function requireP0Mesh(mesh: SplatMesh) {
  const ext = mesh.extSplats;
  if (!mesh.isInitialized || !ext || mesh.numSplats !== ext.numSplats ||
      ext.getNumSh() !== 3 || mesh.maxSh !== 3 || ext.maxSh !== 3 ||
      mesh.packedSplats || mesh.paged || mesh.covSplats || mesh.enableLod ||
      (mesh.splats && mesh.splats !== ext) || mesh.opacity !== 1 || mesh.onFrame ||
      ext.lod || ext.lodSplats || ext.extra.lodTree || mesh.edits?.length ||
      mesh.worldModifiers?.length || mesh.objectModifiers?.length || mesh.splatRgba || mesh.skinning) {
    throw new Error("P0 仅支持未经修改的非 LOD SH3 ExtSplats");
  }
  return ext;
}

// The reference must be decoded independently from exactly one original PLY row.
export function verifyP0EncodedRow(actual: ExtSplats, sourceId: number, reference: ExtSplats) {
  if (!Number.isSafeInteger(sourceId) || sourceId < 0 || sourceId >= actual.numSplats || reference.numSplats !== 1 ||
      actual.extArrays.length !== 2 || reference.extArrays.length !== 2) {
    throw new Error("P0 原行编码参照不合法");
  }
  const arrays = (ext: ExtSplats) => [...ext.extArrays, ...["sh1", "sh2", "sh3a", "sh3b"].map(k => ext.extra[k])];
  const expected = arrays(reference);
  arrays(actual).forEach((array, index) => {
    const ref = expected[index];
    if (!(array instanceof Uint32Array) || !(ref instanceof Uint32Array) ||
        array.length < (sourceId + 1) * 4 || ref.length < 4) throw new Error("P0 缺少完整原行几何／SH编码");
    for (let word = 0; word < 4; word++) {
      if (array[sourceId * 4 + word] !== ref[word]) throw new Error(`P0 源行 ${sourceId} 编码与独立原行不符`);
    }
  });
}

export async function captureP0Source(mesh: SplatMesh, sha256: string, signal?: AbortSignal): Promise<P0Source> {
  const ext = requireP0Mesh(mesh);
  const count = mesh.numSplats;
  if (!/^[a-f0-9]{64}$/.test(sha256) || !Number.isSafeInteger(count) || count < 1 || count > P0_MAX_SPLATS) {
    throw new Error("P0 源身份或数量不合法");
  }
  const geometry = new Float32Array(count * P0_STRIDE);
  for (let id = 0; id < count; id++) {
    if (id % 4096 === 0) {
      await new Promise<void>(resolve => setTimeout(resolve, 0));
      signal?.throwIfAborted();
      if (mesh.extSplats !== ext || mesh.numSplats !== count) throw new Error("P0 提取期间模型已变化");
    }
    const { center, scales, quaternion, opacity } = ext.getSplat(id);
    const row = [center.x, center.y, center.z, scales.x, scales.y, scales.z,
      quaternion.x, quaternion.y, quaternion.z, quaternion.w, opacity];
    if (!row.every(Number.isFinite) || scales.x <= 0 || scales.y <= 0 || scales.z <= 0 ||
        opacity < 0 || opacity > 1 || Math.abs(quaternion.lengthSq() - 1) > 0.01) {
      throw new Error(`P0 不支持源行 ${id} 的退化或非有限高斯`);
    }
    geometry.set(row, id * P0_STRIDE);
  }
  signal?.throwIfAborted();
  return { sha256, count, geometry };
}

// A small, order-sensitive decoder probe; never substitutes for a real-source identity audit.
export function p0IdentityFixture() {
  const count = 17;
  const header = new TextEncoder().encode(["ply", "format binary_little_endian 1.0", `element vertex ${count}`,
    ...P0_PLY_FIELDS.map(field => `property float ${field}`), "end_header", ""].join("\n"));
  const bytes = new Uint8Array(header.length + count * P0_PLY_FIELDS.length * 4);
  bytes.set(header);
  const rows = new DataView(bytes.buffer, header.length);
  const expected = new Float32Array(count * P0_STRIDE);
  for (let id = 0; id < count; id++) {
    const center = [((id * 7) % count - 8) * 0.08, (id % 3 - 1) * 0.12, -2 - (id % 5) * 0.1];
    const scale = 0.01 + id * 0.001;
    const angle = id * 0.13;
    const opacity = id === 0 ? 0.0001 : 0.2 + id * 0.04;
    const values: Record<string, number> = {
      x: center[0], y: center[1], z: center[2], opacity: Math.log(opacity / (1 - opacity)),
      scale_0: Math.log(scale), scale_1: Math.log(scale * 1.3), scale_2: Math.log(scale * 0.7),
      rot_0: Math.cos(angle / 2), rot_3: Math.sin(angle / 2),
      f_dc_0: id * 0.01, f_dc_1: -id * 0.01, f_dc_2: 0.1, f_rest_40: 0.2 + id * 0.01
    };
    P0_PLY_FIELDS.forEach((field, column) => rows.setFloat32((id * P0_PLY_FIELDS.length + column) * 4, values[field] ?? 0, true));
    expected.set([...center, scale, scale * 1.3, scale * 0.7, 0, 0, Math.sin(angle / 2), Math.cos(angle / 2), opacity], id * P0_STRIDE);
  }
  return { bytes, count, expected };
}

export function verifyP0FixtureRows(source: P0Source, expected: Float32Array) {
  validateP0Source(source);
  if (source.geometry.length !== expected.length) throw new Error("P0 解码改变了源行数量");
  for (let id = 0; id < source.count; id++) {
    const base = id * P0_STRIDE;
    for (let field = 0; field < P0_STRIDE; field++) {
      // ExtSplats quantizes scale/rotation/opacity; positions retain float32 precision.
      const tolerance = field < 3 ? 1e-6 : field < 6 ? 0.0001 : 0.002;
      if (!Number.isFinite(expected[base + field]) || !Number.isFinite(source.geometry[base + field]) ||
          Math.abs(source.geometry[base + field] - expected[base + field]) > tolerance) {
        throw new Error(`P0 源行 ${id} 字段 ${field} 与解码顺序不符`);
      }
    }
  }
}
