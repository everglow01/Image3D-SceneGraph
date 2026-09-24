import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { LocalGaussianEditing, maskCount } from "../src/localGaussianEditing.ts";
import { LocalGaussianPersistence } from "../src/localGaussianPersistence.ts";
import { decodeLocalDraft, encodeLocalDraft } from "../src/localGaussianDraft.ts";

const sha = "a".repeat(64), metadata = "b".repeat(64), editId = "c".repeat(32);
const identity = { ply_sha256: sha, metadata_sha256: metadata, gaussian_count: 8 };
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(yes => { resolve = yes; });
  return { promise, resolve };
}
function setup(t: TestContext) {
  const state = new LocalGaussianEditing(sha, 8), sync = new LocalGaussianPersistence(state, editId, metadata, () => {});
  const server = { revision: 0, visible: new Uint8Array([255]), offline: false, loseAck: false, loseVersion: false, expired: false,
    corrupt: false, exportStatus: "running", versions: [] as any[], calls: [] as { url: URL; method: string; body: any; headers: Headers }[],
    wait: null as ReturnType<typeof deferred> | null, entered: deferred(), authWait: null as ReturnType<typeof deferred> | null };
  const requests = new Map<string, any>();
  t.mock.method(globalThis, "fetch", async (input: string, options: RequestInit = {}) => {
    const url = new URL(input, "http://localhost"), method = options.method ?? "GET", headers = new Headers(options.headers);
    const body = options.body instanceof Blob ? new Uint8Array(await options.body.arrayBuffer()) : options.body ? JSON.parse(String(options.body)) : null;
    server.calls.push({ url, method, body, headers });
    if (server.offline) throw new TypeError("断网");
    if (url.pathname.endsWith("local-authorization")) {
      if (method === "DELETE") return json({ state: "closed" });
      await server.authWait?.promise;
      server.expired = false;
      return json({ edit_id: editId, token: "test_token_".repeat(4), identity, expires_in_seconds: 600 });
    }
    if (server.expired && url.pathname.includes("/local-")) return json({ detail: "授权过期" }, 403);
    if (url.pathname.endsWith("local-mask")) {
      if (method === "GET") {
        const version = server.versions.find(v => v.version === url.searchParams.get("version"));
        const visible = version?.mask ?? server.visible, revision = version?.revision ?? server.revision;
        return new Response(visible, { headers: { "X-Edit-Revision": String(revision), "X-Visible-Count": String(maskCount(visible)),
          "X-Source-Sha256": server.corrupt ? "d".repeat(64) : sha, "X-Metadata-Sha256": metadata, "X-Gaussian-Count": "8" } });
      }
      server.entered.resolve(); await server.wait?.promise;
      const id = url.searchParams.get("operation_id")!;
      let ack = requests.get(id);
      if (!ack) {
        if (Number(url.searchParams.get("expected_revision")) !== server.revision) return json({ detail: "CAS冲突" }, 409);
        server.visible = body; server.revision++;
        ack = { edit_id: editId, revision: server.revision, visible_count: maskCount(body) }; requests.set(id, ack);
      }
      if (server.loseAck) { server.loseAck = false; throw new TypeError("ACK丢失"); }
      return json(ack);
    }
    if (url.pathname.endsWith("local-versions")) {
      if (body.expected_revision !== server.revision) return json({ detail: "CAS冲突" }, 409);
      const value = { version: `v${String(server.revision).padStart(8, "0")}`, revision: server.revision, visible_count: maskCount(server.visible), mask: server.visible.slice() };
      if (!server.versions.some(v => v.version === value.version)) server.versions.push(value);
      if (server.loseVersion) { server.loseVersion = false; throw new TypeError("版本ACK丢失"); }
      return json(value);
    }
    if (url.pathname.includes("local-exports")) return json({ status: "running" });
    if (url.pathname.endsWith("/export")) return json({ status: server.exportStatus });
    return json({ edit_id: editId, source: identity, versions: server.versions });
  });
  return { state, sync, server };
}
function remove(state: LocalGaussianEditing, byte: number) { state.select(new Uint8Array([byte]), "replace"); state.deleteSelected(); }

test("保存只确认点击快照，期间修改仍脏；保留undo并将已保存内容撤销为新版本", async t => {
  const { state, sync, server } = setup(t); await sync.open();
  assert.equal(sync.dirty, false);
  state.select(new Uint8Array([1]), "replace"); assert.equal(sync.dirty, false);
  state.deleteSelected(); server.wait = deferred();
  const save = sync.save(() => true); await server.entered.promise;
  await assert.rejects(sync.save(() => true), /尚未完成/);
  remove(state, 2); server.wait.resolve(); await save;
  assert.deepEqual(server.visible, new Uint8Array([254])); assert.equal(sync.dirty, true); assert.equal(state.canUndo, true);
  await sync.save(() => true); assert.equal(server.revision, 2); assert.equal(sync.dirty, false);
  state.undo(); state.undo(); await sync.save(() => true);
  assert.equal(server.revision, 3); assert.equal(server.visible[0], 255);
  assert.equal(server.versions[0].mask[0], 254); assert.equal(state.canRedo, true);
  await sync.dispose();
});

test("仅保护或重复保存不产生快照revision，已有导出下载身份保留", async t => {
  const { state, sync, server } = setup(t); await sync.open();
  state.select(new Uint8Array([1]), "replace"); state.protectSelected(true);
  assert.equal(sync.dirty, false); assert.equal(sync.hasUnsavedWork, true);
  await sync.save(() => true); assert.equal(server.revision, 0); assert.equal(state.canUndo, true);
  await sync.startExport("v00000000"); server.exportStatus = "done"; await sync.pollExport();
  await sync.save(() => true);
  assert.equal(server.calls.filter(c => c.method === "PUT").length, 0);
  assert.match(sync.downloadUrl("v00000000", "scene.ply"), /scene.ply$/);
  assert.equal(state.masks.protected[0], 1); await sync.dispose();
});

test("ACK丢失后原ID原内容重放，过期重新授权也不把新修改夹入重试", async t => {
  const { state, sync, server } = setup(t); await sync.open(); remove(state, 1);
  server.loseAck = true; await assert.rejects(sync.save(() => true), /ACK丢失/);
  assert.equal(server.revision, 1); assert.equal(sync.needsRetry, true);
  remove(state, 2); server.expired = true;
  await assert.rejects(sync.save(() => true), /过期/);
  await sync.save(() => true);
  const puts = server.calls.filter(c => c.method === "PUT");
  assert.equal(puts.length, 2); assert.equal(puts[0].url.search, puts[1].url.search); assert.deepEqual(puts[0].body, puts[1].body);
  assert.equal(server.revision, 1); assert.equal(sync.dirty, true); assert.equal(sync.needsRetry, false);
  assert.equal(server.calls.filter(c => c.method === "POST" && c.url.pathname.endsWith("local-authorization")).length, 2);
  assert.equal(puts[0].headers.get("Content-Type"), "application/octet-stream");
  assert.ok(!puts[0].url.search.includes("token")); await sync.dispose();
});

test("版本响应丢失只重试版本，刷新读回权威mask但不恢复临时历史", async t => {
  const { state, sync, server } = setup(t); await sync.open(); remove(state, 1);
  server.loseVersion = true; await assert.rejects(sync.save(() => true), /版本ACK丢失/);
  await sync.save(() => true); assert.equal(server.calls.filter(c => c.method === "PUT").length, 1); assert.equal(server.versions.length, 1);
  await sync.dispose();
  const reopened = new LocalGaussianEditing(sha, 8), next = new LocalGaussianPersistence(reopened, editId, metadata, () => {});
  await next.open(); assert.equal(reopened.masks.visible[0], 254); assert.equal(reopened.canUndo, false); assert.equal(next.dirty, false); await next.dispose();
});

test("CAS冲突停止同步且保留本地草稿，不自动并集或覆盖", async t => {
  const { state, sync, server } = setup(t); await sync.open(); remove(state, 1);
  server.revision = 4; server.visible = new Uint8Array([253]);
  await assert.rejects(sync.save(() => true), /停止同步/);
  assert.equal(sync.conflict, true); assert.equal(state.masks.visible[0], 254); assert.equal(server.calls.filter(c => c.method === "PUT").length, 0);
  const draft = JSON.parse(sync.draft(null)); assert.equal(draft.base_revision, 0); assert.ok(!JSON.stringify(draft).includes("token"));
  await assert.rejects(sync.save(() => true), /停止同步/); await sync.dispose();
});

test("网络故障和服务器损坏响应不丢修改；过半保存需独立确认", async t => {
  const { state, sync, server } = setup(t); await sync.open();
  remove(state, 15); remove(state, 48);
  await sync.save(() => false); assert.equal(server.revision, 0); assert.equal(sync.needsRetry, false);
  server.offline = true; await assert.rejects(sync.save(() => true), /断网/);
  state.undo(); assert.equal(state.masks.visible[0], 240);
  server.offline = false; server.corrupt = true; await assert.rejects(sync.save(() => true), /身份/);
  server.corrupt = false; await sync.save(() => true);
  assert.equal(server.visible[0], 192); assert.equal(sync.dirty, true);
  await sync.dispose();
});

test("草稿严格拒绝错源、未知字段、padding、全删、非法相机和超限文件", () => {
  const camera = { key: sha, position: [0, 0, 2], target: [0, 0, 0], up: [0, 1, 0], fov: 50 };
  const text = encodeLocalDraft(editId, identity, 2, new Uint8Array([254]), new Uint8Array([2]), camera);
  const draft = decodeLocalDraft(text, editId, identity); assert.equal(draft.visible[0], 254); assert.equal(draft.protected[0], 2);
  const change = (fields: object) => JSON.stringify({ ...JSON.parse(text), ...fields });
  for (const invalid of [change({ token: "forbidden" }), change({ base_revision: -1 }), change({ edit_id: "d".repeat(32) }),
    change({ source: { ...identity, metadata_sha256: "d".repeat(64) } }), change({ visible: "AAA=" }), change({ visible: "AA==" }),
    change({ camera: { ...camera, up: [0, 0, 0] } }), change({ camera: { ...camera, position: [0, 0, 0] } }), " ".repeat(2 * 1024 * 1024 + 1)]) {
    assert.throws(() => decodeLocalDraft(invalid, editId, identity));
  }
  assert.throws(() => encodeLocalDraft(editId, { ...identity, gaussian_count: 7 }, 0, new Uint8Array([255]), new Uint8Array([0]), null));
});

test("断网草稿先恢复本地，重连核对原revision；远端变化不能自动覆盖", async t => {
  const { state, sync, server } = setup(t); await sync.open();
  const text = encodeLocalDraft(editId, identity, 0, new Uint8Array([254]), new Uint8Array([2]), null);
  server.offline = true;
  const restored = await sync.restoreDraft(text); assert.equal(restored.offline, true); assert.equal(sync.needsBaselineCheck, true);
  assert.equal(state.masks.visible[0], 254); assert.equal(state.masks.protected[0], 2); assert.equal(JSON.parse(sync.draft(null)).base_revision, 0);
  server.offline = false; await sync.verifyDraftBase(); await sync.save(() => true); assert.equal(server.visible[0], 254);
  const stale = await sync.restoreDraft(text); assert.equal(stale.conflict, true); await assert.rejects(sync.save(() => true), /停止同步/);
  await sync.dispose();
});

test("导入等待网络时新本地修改不被覆盖", async t => {
  const { state, sync, server } = setup(t);
  server.authWait = deferred(); const text = encodeLocalDraft(editId, identity, 0, new Uint8Array([253]), new Uint8Array([0]), null);
  const restore = sync.restoreDraft(text); remove(state, 1); server.authWait.resolve();
  await assert.rejects(restore, /本地状态已变化/); assert.equal(state.masks.visible[0], 254);
  await sync.dispose();
});

test("保存待确认时不能用草稿替换原请求", async t => {
  const { state, sync, server } = setup(t); await sync.open(); remove(state, 1);
  server.loseAck = true; await assert.rejects(sync.save(() => true), /ACK丢失/);
  const text = encodeLocalDraft(editId, identity, 0, new Uint8Array([253]), new Uint8Array([0]), null);
  await assert.rejects(sync.restoreDraft(text), /尚未确认/);
  assert.equal(state.masks.visible[0], 254); assert.equal(sync.needsRetry, true);
  await sync.save(() => true); assert.equal(server.revision, 1); await sync.dispose();
});

test("历史版本只读mask和导出不提交当前本地修改，下载使用普通链接", async t => {
  const { state, sync, server } = setup(t); await sync.open(); remove(state, 1); await sync.save(() => true);
  remove(state, 2); const before = state.masks.visible.slice(), history = state.historyBytes;
  assert.equal((await sync.readVersion("v00000001"))[0], 254);
  await sync.startExport("v00000001"); await sync.pollExport(); assert.equal(sync.exportState, "running");
  server.exportStatus = "done"; await sync.pollExport();
  assert.match(sync.downloadUrl("v00000001", "bundle.zip"), /\/versions\/v00000001\/assets\/bundle.zip$/);
  assert.deepEqual(state.masks.visible, before); assert.equal(state.historyBytes, history); assert.equal(server.revision, 1);
  assert.equal(server.calls.some(c => /render-sessions|ice|offer/.test(c.url.pathname)), false);
  await sync.dispose();
});

test("关闭期间迟到的授权释放且不发起保存或覆盖本地状态", async t => {
  const { sync, server } = setup(t); server.authWait = deferred();
  const opening = sync.open(); await Promise.resolve();
  const closing = sync.dispose(); server.authWait.resolve();
  await assert.rejects(opening, /已关闭/); await closing;
  assert.equal(server.calls.filter(c => c.method === "DELETE").length, 1);
  assert.equal(server.calls.some(c => c.method === "PUT"), false);
});
