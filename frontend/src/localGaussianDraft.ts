import type { CameraView } from "./cloudGaussianEditor.ts";
import { maskCount } from "./localGaussianEditing.ts";
import { validateP0Mask } from "./localGaussianP0Source.ts";

export type LocalSourceIdentity = { ply_sha256: string; metadata_sha256: string; gaussian_count: number };
export type LocalDraft = {
  schema: "image3d_local_draft_v1"; edit_id: string; source: LocalSourceIdentity;
  base_revision: number | null; visible: string; protected: string; camera: CameraView | null;
};
export const LOCAL_DRAFT_LIMIT = 2 * 1024 * 1024;

export function validateLocalIdentity(source: LocalSourceIdentity) {
  if (!source || Object.keys(source).sort().join() !== "gaussian_count,metadata_sha256,ply_sha256" ||
      !/^[a-f0-9]{64}$/.test(source.ply_sha256) || !/^[a-f0-9]{64}$/.test(source.metadata_sha256) ||
      !Number.isSafeInteger(source.gaussian_count) || source.gaussian_count < 1 || source.gaussian_count > 3_000_000) {
    throw new Error("本地文档源身份不合法");
  }
}

export function sameLocalIdentity(a: LocalSourceIdentity, b: LocalSourceIdentity) {
  return a.ply_sha256 === b.ply_sha256 && a.metadata_sha256 === b.metadata_sha256 && a.gaussian_count === b.gaussian_count;
}

function encodeMask(mask: Uint8Array) {
  let binary = "";
  for (let i = 0; i < mask.length; i += 8192) binary += String.fromCharCode(...mask.subarray(i, i + 8192));
  return btoa(binary);
}

function decodeMask(value: unknown, count: number) {
  const bytes = Math.ceil(count / 8);
  if (typeof value !== "string" || value.length !== 4 * Math.ceil(bytes / 3) || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
    throw new Error("草稿 mask 编码或长度不合法");
  }
  const mask = Uint8Array.from(atob(value), c => c.charCodeAt(0));
  validateP0Mask(mask, count);
  if (encodeMask(mask) !== value) throw new Error("草稿 mask 不是规范 base64");
  return mask;
}

function validateCamera(camera: CameraView | null, sha: string) {
  if (camera === null) return;
  if (!camera || Object.keys(camera).sort().join() !== "fov,key,position,target,up" || camera.key !== sha ||
      ![camera.position, camera.target, camera.up].every(v => Array.isArray(v) && v.length === 3 && v.every(x => typeof x === "number" && Number.isFinite(x) && Math.abs(x) <= 1e9)) ||
      !Number.isFinite(camera.fov) || camera.fov < 1 || camera.fov > 179) throw new Error("草稿相机不合法");
  const distance = Math.hypot(...camera.position.map((v, i) => v - camera.target[i]));
  const up = Math.hypot(...camera.up);
  const direction = camera.position.map((v, i) => v - camera.target[i]);
  const cross = Math.hypot(direction[1] * camera.up[2] - direction[2] * camera.up[1],
    direction[2] * camera.up[0] - direction[0] * camera.up[2], direction[0] * camera.up[1] - direction[1] * camera.up[0]);
  if (distance < 1e-9 || up < 1e-9 || cross < 1e-9 * distance * up) throw new Error("草稿相机方向退化");
}

export function encodeLocalDraft(editId: string, source: LocalSourceIdentity, baseRevision: number | null,
  visible: Uint8Array, protectedMask: Uint8Array, camera: CameraView | null) {
  const text = JSON.stringify({ schema: "image3d_local_draft_v1", edit_id: editId, source, base_revision: baseRevision,
    visible: encodeMask(visible), protected: encodeMask(protectedMask), camera });
  decodeLocalDraft(text, editId, source);
  return text;
}

export function decodeLocalDraft(text: string, editId: string, source: LocalSourceIdentity) {
  if (text.length > LOCAL_DRAFT_LIMIT) throw new Error("草稿超过 2 MiB 上限");
  validateLocalIdentity(source);
  const value: LocalDraft = JSON.parse(text);
  if (!value || Object.keys(value).sort().join() !== "base_revision,camera,edit_id,protected,schema,source,visible" ||
      value.schema !== "image3d_local_draft_v1" || !/^[a-f0-9]{32}$/.test(value.edit_id) || value.edit_id !== editId ||
      (value.base_revision !== null && (!Number.isSafeInteger(value.base_revision) || value.base_revision < 0))) {
    throw new Error("草稿结构或文档身份不匹配");
  }
  validateLocalIdentity(value.source);
  if (!sameLocalIdentity(value.source, source)) throw new Error("草稿与当前 PLY／metadata／高斯数量不匹配");
  validateCamera(value.camera, source.ply_sha256);
  const visible = decodeMask(value.visible, source.gaussian_count), protectedMask = decodeMask(value.protected, source.gaussian_count);
  if (!maskCount(visible)) throw new Error("草稿不能隐藏全部高斯");
  return { baseRevision: value.base_revision, visible, protected: protectedMask, camera: value.camera };
}
