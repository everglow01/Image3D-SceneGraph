import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as THREE from "three";
import * as metadata from "../src/gaussianViewerMetadata.ts";
import * as editor from "../src/cloudGaussianEditor.ts";

const code = ts.transpileModule(readFileSync(new URL("../src/CloudGaussianViewer.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX }
}).outputText;

function harness(available = true) {
  const hooks: any[] = [], effects: (() => void)[] = [], requests: any[] = [], peers: any[] = [];
  const intervals = new Map<number, () => void>();
  let cursor = 0, dirty = true, tree: any, timer = 0, revision = 0, imageNode: any, frameCallback: (() => void) | undefined;
  const stage = { getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 500 }) };
  const video = { srcObject: null, play: async () => {}, requestVideoFrameCallback: (fn: () => void) => { frameCallback = fn; } };
  const source = { job_id: "source", asset_role: "scene_splat", label: "Original" };
  const document = () => ({ edit_id: "a".repeat(32), source, revision, visible_count: 8 - revision, can_undo: revision > 0, can_redo: false, versions: [] });
  const react = {
    useRef: (initial: unknown) => { const index = cursor++; return hooks[index] ??= { current: initial }; },
    useState: (initial: unknown) => {
      const index = cursor++;
      if (!(index in hooks)) hooks[index] = initial;
      return [hooks[index], (next: any) => {
        const value = typeof next === "function" ? next(hooks[index]) : next;
        if (!Object.is(value, hooks[index])) { hooks[index] = value; dirty = true; }
      }];
    },
    useEffect: (callback: () => (() => void) | void, deps: unknown[]) => {
      const index = cursor++, previous = hooks[index];
      if (!previous || deps.some((d, i) => !Object.is(d, previous.deps[i]))) effects.push(() => {
        previous?.cleanup?.(); hooks[index] = { deps, cleanup: callback() };
      });
    }
  };
  const jsx = (type: unknown, props: any, key: any) => {
    const node = { type, props, key };
    if (props.ref) {
      if (type === "video") props.ref.current = video;
      else if (type === "img") { imageNode = node; props.ref.current = node; }
      else props.ref.current = stage;
    }
    return node;
  };
  class Controls { target = new THREE.Vector3(); enabled = true; update() {} dispose() {} }
  class Peer {
    connectionState = "connected"; iceGatheringState = "complete";
    localDescription = { sdp: "offer" }; ontrack: any; channel: any;
    constructor() { peers.push(this); }
    createDataChannel() { return this.channel = { readyState: "open", bufferedAmount: 0, send() {}, onclose() {} }; }
    addTransceiver() {} async createOffer() { return {}; } async setLocalDescription() {}
    async setRemoteDescription() { this.ontrack({ track: {} }); }
    close() { this.connectionState = "closed"; this.channel?.onclose(); }
  }
  const exports: Record<string, any> = {};
  runInNewContext(code, { exports, crypto: globalThis.crypto, RTCPeerConnection: Peer, MediaStream: class {},
    window: { document: { hidden: false }, setInterval: (fn: () => void) => { intervals.set(++timer, fn); return timer; },
      clearInterval: (id: number) => intervals.delete(id), setTimeout, clearTimeout },
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (url: string, options: any) => {
      const body = options.body ? JSON.parse(options.body) : null;
      requests.push({ url, ...options, body });
      let value: any;
      if (url.endsWith("capabilities")) value = { cloud_available: available, reason: available ? null : "未部署" };
      else if (url === "/metadata.json") value = { sh_degree: 3, scene_radius_p95: 1 };
      else if (url.includes("gaussian-edits?")) value = { edits: [] };
      else if (url === "/api/gaussian-edits" || url.startsWith("/api/gaussian-edits/")) value = document();
      else if (url === "/api/gaussian-render-sessions") value = { session_id: "session", token: "secret", state: "loading" };
      else if (url.endsWith("/ice")) value = { iceTransportPolicy: "relay", iceServers: [] };
      else if (url.endsWith("/offer")) value = { type: "answer", sdp: "answer" };
      else if (url.endsWith("/freeze-frame")) value = { ticket: `ticket-${revision}`, revision, camera_seq: body.sequence, width: 1920, height: 1080, image: "data:image/png;base64,c2FtZQ==" };
      else if (url.endsWith("/selection")) value = { selection_token: "selected", selected_count: 1, visible_count: 8, revision };
      else if (url.endsWith("/preview")) value = { revision, image: "data:image/png;base64,cHJldmlldw==" };
      else if (url.endsWith("/operations")) { revision++; value = document(); }
      else value = { state: "viewing", revision, visible_count: 8 - revision };
      return { ok: true, json: async () => value };
    },
    require: (name: string) => {
      if (name === "react") return react;
      if (name === "react/jsx-runtime") return { jsx, jsxs: jsx };
      if (name === "three") return THREE;
      if (name.includes("OrbitControls")) return { OrbitControls: Controls };
      if (name === "./gaussianViewerMetadata") return metadata;
      if (name === "./cloudGaussianEditor") return editor;
      if (name.endsWith(".css")) return {};
      throw new Error(`Unexpected dependency: ${name}`);
    }
  });
  function find(node: any, predicate: (n: any) => boolean): any {
    if (!node || typeof node !== "object") return null;
    if (predicate(node)) return node;
    for (const child of [node.props?.children].flat(Infinity)) { const result = find(child, predicate); if (result) return result; }
    return null;
  }
  const flush = async () => {
    for (let i = 0; i < 100; i++) {
      await Promise.resolve();
      if (dirty) {
        dirty = false; cursor = 0;
        tree = exports.CloudGaussianViewer({ source, metadataUrl: "/metadata.json", alignmentUrl: null, cameraPathUrl: null });
        while (effects.length) effects.shift()!();
      }
    }
  };
  return { requests, peers, intervals, flush, video,
    button: (label: string) => find(tree, n => n.type === "button" && n.props.children === label)?.props,
    tool: () => find(tree, n => n.type === "select" && n.props.value === "rectangle").props,
    loadImage: () => imageNode.props.onLoad({ currentTarget: imageNode }),
    imageKey: () => imageNode?.key,
    displayFrame: () => frameCallback?.(),
    image: () => find(tree, n => n.type === "img"),
    unmount: () => { for (const hook of hooks) hook?.cleanup?.(); }
  };
}

test("cloud page never fetches PLY and waits for the decoded new generation after edits", async () => {
  const h = harness(); await h.flush();
  h.button("连接云端").onClick(); await h.flush(); h.displayFrame(); await h.flush();
  h.button("固定高清画面，开始修剪").onClick(); await h.flush();
  assert.equal(h.button("计算选择").disabled, true);
  h.loadImage(); await h.flush(); h.tool().onChange({ target: { value: "box" } }); await h.flush();
  h.button("计算选择").onClick(); await h.flush();
  assert.equal(h.button("确认删除选中项").disabled, true);
  h.button("隔离选中").onClick(); await h.flush(); h.loadImage(); await h.flush();
  assert.equal(h.button("确认删除选中项").disabled, false);
  h.button("确认删除选中项").onClick(); await h.flush();
  const key = h.imageKey(); h.loadImage(); await h.flush();
  h.button("重新固定高清画面").onClick(); await h.flush();
  assert.notEqual(h.imageKey(), key, "identical PNG still gets a new image-load fence");
  h.loadImage(); await h.flush();
  h.button("恢复交互观看").onClick(); await h.flush();
  assert.ok(h.image(), "keep revision-correct PNG until new peer presents a frame");
  h.displayFrame(); await h.flush(); assert.equal(h.image(), null);
  assert.ok(h.requests.every(r => !r.url.endsWith(".ply")));
  assert.ok(h.requests.filter(r => r.url.includes("/gaussian-render-sessions/")).every(r => r.headers["X-Editor-Token"] === "secret"));
  h.unmount(); await Promise.resolve();
  assert.equal(h.intervals.size, 0);
  assert.ok(h.peers.every(p => p.connectionState === "closed"));
  assert.ok(h.requests.some(r => r.method === "DELETE"));
});

test("missing cloud capability stays unavailable without local fallback", async () => {
  const h = harness(false); await h.flush();
  assert.equal(h.button("连接云端").disabled, true);
  assert.equal(h.peers.length, 0);
  assert.ok(h.requests.every(r => !r.url.endsWith(".ply")));
  h.unmount();
});
