import type { CloudSource } from "./CloudGaussianViewer";
import { validateLocalIdentity, type LocalSourceIdentity } from "./localGaussianDraft.ts";

export type LocalEditDocument = {
  edit_id: string; revision: number; visible_count: number;
  source: LocalSourceIdentity & CloudSource & { ply_asset: string; metadata_asset: string };
};
export type LoadedLocalSource = { plySha256: string; count: number; metadataText: string };

export async function localDocumentRequest<T>(url: string, signal: AbortSignal, source?: CloudSource): Promise<T> {
  const response = await fetch(url, { signal: AbortSignal.any([signal, AbortSignal.timeout(60_000)]), cache: "no-store", ...(source ? {
    method: "POST", headers: { "Content-Type": "application/json", "X-Image3D-Editor": "1" },
    body: JSON.stringify({ job_id: source.job_id, variant_id: source.variant_id, asset_role: source.asset_role })
  } : {}) });
  if (!response.ok) {
    const value = await response.json().catch(() => ({}));
    throw new Error(typeof value.detail === "string" ? value.detail : `编辑文档请求失败（${response.status}）`);
  }
  return response.json() as Promise<T>;
}

export function matchesLocalDocument(doc: LocalEditDocument, source: CloudSource) {
  return doc?.source?.job_id === source.job_id && (doc.source.variant_id ?? undefined) === source.variant_id && doc.source.asset_role === source.asset_role;
}

export function validateLocalDocument(doc: LocalEditDocument, source: CloudSource, loaded: LoadedLocalSource, plyUrl: string, metadataUrl: string) {
  if (!doc || !/^[a-f0-9]{32}$/.test(doc.edit_id) || !matchesLocalDocument(doc, source)) throw new Error("编辑文档来源不符");
  const identity = { ply_sha256: doc.source.ply_sha256, metadata_sha256: doc.source.metadata_sha256, gaussian_count: doc.source.gaussian_count };
  validateLocalIdentity(identity);
  if (identity.ply_sha256 !== loaded.plySha256 || identity.gaussian_count !== loaded.count ||
    plyUrl !== `/api/jobs/${source.job_id}/assets/${doc.source.ply_asset}` ||
    metadataUrl !== `/api/jobs/${source.job_id}/assets/${doc.source.metadata_asset}`) throw new Error("文档与当前已加载模型不一致，请重新加载模型");
}
