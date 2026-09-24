import { operationId, type CameraView } from "./cloudGaussianEditor.ts";
import { LocalGaussianEditing, maskCount } from "./localGaussianEditing.ts";
import { decodeLocalDraft, encodeLocalDraft, sameLocalIdentity, validateLocalIdentity, type LocalSourceIdentity } from "./localGaussianDraft.ts";
import { validateP0Mask } from "./localGaussianP0Source.ts";

export type LocalVersion = { version: string; revision: number; visible_count: number; exported?: boolean };
type Snapshot = { revision: number; visible: Uint8Array };
type PendingSave = { id: string; expectedRevision: number; visible: Uint8Array; generation: number; confirmLarge: boolean; ackRevision: number | null };
const sameMask = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i]);

export class LocalGaussianPersistence {
  readonly state: LocalGaussianEditing;
  readonly editId: string;
  readonly identity: LocalSourceIdentity;
  private token: string | null = null;
  private expires = 0;
  private base: Snapshot | null = null;
  private draftBaseRevision: number | null = null;
  private savedGeneration = -1;
  private pending: PendingSave | null = null;
  private working: Promise<unknown> | null = null;
  private closed = false;
  private readonly notify: () => void;
  conflict = false;
  versions: LocalVersion[] = [];
  lastVersion: LocalVersion | null = null;
  exportState = "";
  exportVersion = "";

  constructor(state: LocalGaussianEditing, editId: string, metadataSha256: string, notify: () => void) {
    this.state = state; this.editId = editId; this.notify = notify;
    this.identity = { ply_sha256: state.sourceSha256, metadata_sha256: metadataSha256, gaussian_count: state.count };
    validateLocalIdentity(this.identity);
    if (!/^[a-f0-9]{32}$/.test(editId)) throw new Error("编辑文档 ID 不合法");
  }
  get baseRevision() { return this.base?.revision ?? this.draftBaseRevision; }
  get needsBaselineCheck() { return !this.base && this.draftBaseRevision !== null; }
  get busy() { return this.working !== null; }
  get needsRetry() { return this.pending !== null; }
  get dirty() { return this.savedGeneration !== this.state.visibilityRevision || this.pending !== null; }
  get hasUnsavedWork() { return this.dirty || maskCount(this.state.masks.protected) > 0; }
  private get path() { return `/api/gaussian-edits/${this.editId}`; }

  private async request(suffix: string, method = "GET", body?: unknown, binary = false, authenticated = true) {
    if (this.closed) throw new Error("本地保存控制器已关闭");
    const response = await fetch(this.path + suffix, { method, cache: "no-store", signal: AbortSignal.timeout(60_000),
      headers: { "X-Image3D-Editor": "1", ...(body === undefined ? {} : { "Content-Type": binary ? "application/octet-stream" : "application/json" }),
        ...(authenticated && this.token ? { "X-Editor-Token": this.token } : {}) },
      ...(body === undefined ? {} : { body: binary ? new Blob([body as Uint8Array<ArrayBuffer>]) : JSON.stringify(body) }) });
    if (this.closed && suffix !== "/local-authorization") throw new Error("本地保存控制器已关闭");
    if (!response.ok) {
      if (response.status === 403 && authenticated) this.token = null;
      const value = await response.json().catch(() => ({}));
      throw new Error(typeof value.detail === "string" ? value.detail : `保存请求失败（${response.status}）`);
    }
    return response;
  }

  private async authorize() {
    if (this.token && Date.now() < this.expires) return false;
    this.token = null;
    const response = await this.request("/local-authorization", "POST", this.identity, false, false);
    const value = await response.json();
    // A late authorization must still be released by dispose, even if opening was abandoned.
    if (typeof value.token !== "string" || !/^[A-Za-z0-9_-]{20,128}$/.test(value.token)) throw new Error("服务器授权响应不合法");
    this.token = value.token;
    if (this.closed) throw new Error("本地保存控制器已关闭");
    validateLocalIdentity(value.identity);
    if (value.edit_id !== this.editId || !sameLocalIdentity(value.identity, this.identity) ||
        !Number.isFinite(value.expires_in_seconds) || value.expires_in_seconds < 0 || value.expires_in_seconds > 600) throw new Error("服务器授权源不匹配");
    this.expires = Date.now() + value.expires_in_seconds * 1000;
    return true;
  }

  private async readSnapshot(version?: string): Promise<Snapshot> {
    const response = await this.request(`/local-mask${version ? `?version=${encodeURIComponent(version)}` : ""}`);
    const source = { ply_sha256: response.headers.get("X-Source-Sha256") ?? "",
      metadata_sha256: response.headers.get("X-Metadata-Sha256") ?? "", gaussian_count: Number(response.headers.get("X-Gaussian-Count")) };
    validateLocalIdentity(source);
    const rawRevision = response.headers.get("X-Edit-Revision"), rawCount = response.headers.get("X-Visible-Count");
    const revision = Number(rawRevision), count = Number(rawCount);
    if (!sameLocalIdentity(source, this.identity) || rawRevision === null || rawCount === null ||
        !Number.isSafeInteger(revision) || revision < 0 || !Number.isSafeInteger(count)) throw new Error("服务器 mask 身份或版本不合法");
    const visible = new Uint8Array(await response.arrayBuffer());
    validateP0Mask(visible, this.state.count);
    if (!maskCount(visible) || maskCount(visible) !== count) throw new Error("服务器 mask 可见数不一致");
    return { revision, visible };
  }

  private async exclusive<T>(action: () => Promise<T>): Promise<T> {
    if (this.closed || this.working) throw new Error("上一项文档操作尚未完成，或编辑已关闭");
    const task = Promise.resolve().then(action); this.working = task; this.notify();
    try { return await task; }
    finally { this.working = null; if (!this.closed) this.notify(); }
  }
  private stopConflict(): never {
    this.conflict = true;
    throw new Error("服务器文档版本已变化，已停止同步并保留本地修改；请下载草稿。不会自动合并或覆盖。");
  }

  async open() {
    if (this.base || this.state.contentRevision !== 0) throw new Error("重新打开会丢失本地历史；请先下载草稿并退出编辑");
    const generation = this.state.contentRevision;
    return this.exclusive(async () => {
      await this.authorize();
      const snapshot = await this.readSnapshot();
      if (this.state.contentRevision !== generation) this.stopConflict();
      this.state.loadSnapshot(snapshot.visible);
      this.base = snapshot; this.savedGeneration = this.state.visibilityRevision;
      this.conflict = false;
      await this.readVersions();
    });
  }

  private async reconcile() {
    await this.authorize();
    const remote = await this.readSnapshot();
    if (!this.base) this.stopConflict();
    if (remote.revision === this.base.revision && sameMask(remote.visible, this.base.visible)) return;
    const p = this.pending;
    if (p && remote.revision === (p.ackRevision ?? p.expectedRevision + 1) && sameMask(remote.visible, p.visible)) return;
    this.stopConflict();
  }

  async save(confirmLarge: () => boolean) {
    const clickedVisible = this.state.masks.visible, clickedGeneration = this.state.visibilityRevision;
    return this.exclusive(async () => {
      if (this.conflict) this.stopConflict();
      // Retrying must replay the original snapshot, not whatever is currently being edited.
      if (!this.pending) {
        if (!this.base) throw new Error("尚未读回服务器基线，请先下载草稿再重新打开文档");
        const visible = clickedVisible, base = this.base;
        const large = maskCount(base.visible.map((v, i) => v & ~visible[i])) > maskCount(base.visible) / 2;
        if (large && !confirmLarge()) return;
        this.pending = { id: operationId(), expectedRevision: base.revision, visible,
          generation: clickedGeneration, confirmLarge: large, ackRevision: sameMask(visible, base.visible) ? base.revision : null };
      }
      const p = this.pending;
      await this.reconcile();
      if (p.ackRevision === null) {
        const query = new URLSearchParams({ ...this.identity, gaussian_count: String(this.state.count),
          expected_revision: String(p.expectedRevision), operation_id: p.id, confirm_large: String(p.confirmLarge) });
        const ack = await (await this.request(`/local-mask?${query}`, "PUT", p.visible, true)).json();
        if (ack.edit_id !== this.editId || ack.revision !== p.expectedRevision + 1 || ack.visible_count !== maskCount(p.visible)) throw new Error("保存 ACK 与提交快照不一致；请重试原请求");
        p.ackRevision = ack.revision;
      }
      const version = await (await this.request("/local-versions", "POST", { expected_revision: p.ackRevision })).json();
      this.validateVersion(version);
      if (version.revision !== p.ackRevision || version.visible_count !== maskCount(p.visible)) throw new Error("版本响应与保存快照不一致");
      const remote = await this.readSnapshot();
      if (remote.revision !== p.ackRevision || !sameMask(remote.visible, p.visible)) this.stopConflict();
      this.base = remote; this.savedGeneration = p.generation; this.lastVersion = version;
      this.pending = null;
      this.versions = [...this.versions.filter(v => v.version !== version.version), version];
    });
  }

  private validateVersion(value: LocalVersion) {
    if (!value || !/^v[0-9]{8}$/.test(value.version) || !Number.isSafeInteger(value.revision) ||
        Number(value.version.slice(1)) !== value.revision || !Number.isSafeInteger(value.visible_count) ||
        value.visible_count < 1 || value.visible_count > this.state.count) throw new Error("服务器版本响应不合法");
  }
  private async readVersions() {
    const document = await (await this.request("", "GET", undefined, false, false)).json();
    if (document.edit_id !== this.editId || !sameLocalIdentity(document.source, this.identity) || !Array.isArray(document.versions)) throw new Error("服务器文档源不一致");
    for (const version of document.versions) this.validateVersion(version);
    this.versions = document.versions;
  }

  draft(camera: CameraView | null) {
    const masks = this.state.masks;
    return encodeLocalDraft(this.editId, this.identity, this.baseRevision, masks.visible, masks.protected, camera);
  }
  async restoreDraft(text: string) {
    const draft = decodeLocalDraft(text, this.editId, this.identity), generation = this.state.contentRevision;
    return this.exclusive(async () => {
      if (this.pending) throw new Error("保存结果尚未确认，请先重试原请求；不能替换待确认快照");
      // A valid offline draft remains editable even when the server cannot be reached.
      let remote: Snapshot | null = null, failure: unknown;
      try { await this.authorize(); remote = await this.readSnapshot(); } catch (error) { failure = error; }
      if (this.closed || generation !== this.state.contentRevision) throw new Error("导入期间本地状态已变化，未覆盖修改");
      this.state.loadSnapshot(draft.visible, draft.protected);
      this.savedGeneration = -1;
      this.draftBaseRevision = draft.baseRevision;
      this.conflict = draft.baseRevision === null || Boolean(remote && remote.revision !== draft.baseRevision);
      this.base = remote && !this.conflict ? remote : null;
      return { camera: draft.camera, offline: Boolean(failure), conflict: this.conflict };
    });
  }

  async verifyDraftBase() {
    return this.exclusive(async () => {
      if (this.conflict || this.draftBaseRevision === null) this.stopConflict();
      await this.authorize(); const remote = await this.readSnapshot();
      if (remote.revision !== this.draftBaseRevision) this.stopConflict();
      this.base = remote; this.conflict = false;
    });
  }

  async readVersion(version: string) {
    return this.exclusive(async () => {
      const expected = this.versions.find(v => v.version === version);
      if (!expected) throw new Error("请选择已保存的不可变版本");
      await this.authorize();
      const snapshot = await this.readSnapshot(version);
      if (snapshot.revision !== expected.revision || maskCount(snapshot.visible) !== expected.visible_count) throw new Error("历史版本 mask 不匹配");
      return snapshot.visible;
    });
  }

  async startExport(version: string) {
    return this.exclusive(async () => {
      if (!this.versions.some(v => v.version === version)) throw new Error("请选择已保存的不可变版本");
      await this.authorize();
      const value = await (await this.request(`/local-exports/${version}`, "POST")).json();
      if (value.status !== "running" && value.status !== "done") throw new Error("导出启动响应不合法");
      this.exportVersion = version; this.exportState = value.status;
    });
  }
  async pollExport() {
    return this.exclusive(async () => {
      if (!this.exportVersion) return;
      const value = await (await this.request(`/versions/${this.exportVersion}/export`, "GET", undefined, false, false)).json();
      if (!["running", "done", "error", "not_exported"].includes(value.status)) throw new Error("导出状态响应不合法");
      this.exportState = value.status;
      if (value.status === "done") this.versions = this.versions.map(v => v.version === this.exportVersion ? { ...v, exported: true } : v);
      if (value.status === "error") throw new Error(value.error || "导出失败；旧模型与版本未覆盖");
    });
  }
  downloadUrl(version: string, name: "scene.ply" | "bundle.zip") {
    if (!this.versions.some(v => v.version === version && v.exported)) throw new Error("该版本尚未完成导出");
    return `${this.path}/versions/${version}/assets/${name}`;
  }

  async dispose() {
    this.closed = true;
    await this.working?.catch(() => {});
    if (this.token) {
      const token = this.token; this.token = null;
      const response = await fetch(`${this.path}/local-authorization`, { method: "DELETE", cache: "no-store", signal: AbortSignal.timeout(60_000),
        headers: { "X-Image3D-Editor": "1", "X-Editor-Token": token } });
      if (!response.ok && response.status !== 403) throw new Error(`本地授权释放失败（${response.status}），请等待授权到期`);
    }
  }
}
