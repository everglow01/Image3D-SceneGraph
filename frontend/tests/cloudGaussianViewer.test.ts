import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as THREE from "three";
import * as metadata from "../src/gaussianViewerMetadata.ts";
import * as editor from "../src/cloudGaussianEditor.ts";
import { createHookHarness, findNode as find } from "./hookHarness.ts";

const code = ts.transpileModule(readFileSync(new URL("../src/CloudGaussianViewer.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX }
}).outputText;

function harness(available = true, iceServers: RTCIceServer[] = [{
  urls: ["turn:10.186.96.23:8082?transport=udp", "turn:10.186.96.23:8082?transport=tcp"],
  username: "temporary-user", credential: "temporary-credential"
}]) {
  const hooks = createHookHarness(), requests: any[] = [], peers: any[] = [];
  const intervals = new Map<number, () => void>(), timeouts = new Map<number, () => void>();
  let tree: any, timer = 0, revision = 0, protectedCount = 0, imageNode: any, frameCallback: (() => void) | undefined;
  const stage = { getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 500 }), addEventListener() {}, removeEventListener() {} };
  const video = { srcObject: null, play: async () => {}, requestVideoFrameCallback: (fn: () => void) => { frameCallback = fn; } };
  const source = { job_id: "source", asset_role: "scene_splat", label: "Original" };
  const document = () => ({ edit_id: "a".repeat(32), source, revision, visible_count: 8 - revision, can_undo: revision > 0, can_redo: false, versions: [] });
  const jsx = (type: unknown, props: any, key: any) => {
    const node = { type, props, key };
    if (props.ref) {
      if (type === "video") props.ref.current = video;
      else if (type === "img") { imageNode = node; props.ref.current = node; }
      else props.ref.current = stage;
    }
    return node;
  };
  class Controls {
    target = new THREE.Vector3(); enabled = true;
    constructor() { orbit = this; }
    listener?: () => void;
    update() { this.listener?.(); }
    dispose() {}
    addEventListener(_: string, fn: () => void) { this.listener = fn; }
    removeEventListener() { this.listener = undefined; }
  }
  let orbit: Controls;
  class Peer {
    connectionState = "connected"; iceGatheringState = "complete";
    localDescription = { sdp: "offer" }; ontrack: any; channel: any;
    config: RTCConfiguration;
    constructor(config: RTCConfiguration) { this.config = config; peers.push(this); }
    createDataChannel() { return this.channel = { readyState: "open", bufferedAmount: 0, send() {}, onclose() {} }; }
    addTransceiver() {} async createOffer() { return {}; } async setLocalDescription() {}
    async setRemoteDescription() { this.ontrack({ track: {} }); }
    close() { this.connectionState = "closed"; this.channel?.onclose(); }
  }
  const exports: Record<string, any> = {};
  runInNewContext(code, { exports, Error, crypto: globalThis.crypto, RTCPeerConnection: Peer, MediaStream: class {},
    window: { document: { hidden: false }, setInterval: (fn: () => void) => { intervals.set(++timer, fn); return timer; },
      clearInterval: (id: number) => intervals.delete(id), setTimeout: (fn: () => void) => { timeouts.set(++timer, fn); return timer; },
      clearTimeout: (id: number) => timeouts.delete(id) },
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (url: string, options: any) => {
      const body = options.body ? JSON.parse(options.body) : null;
      requests.push({ url, ...options, body });
      let value: any;
      if (url.endsWith("capabilities")) value = { cloud_available: available, reason: available ? null : "未部署" };
      else if (url === "/metadata.json") value = { sh_degree: 3, scene_radius_p95: 1 };
      else if (url.includes("gaussian-edits?")) value = { edits: [] };
      else if (url === "/api/gaussian-edits" || url.startsWith("/api/gaussian-edits/")) value = document();
      else if (url === "/api/gaussian-render-sessions") { protectedCount = 0; value = { session_id: "session", token: "secret", state: "loading" }; }
      else if (url.endsWith("/protection")) { protectedCount = body.kind === "add" ? 1 : 0; value = { protected_count: protectedCount }; }
      else if (url.endsWith("/ice")) value = { iceTransportPolicy: "relay", iceServers };
      else if (url.endsWith("/offer")) value = { type: "answer", sdp: "answer" };
      else if (url.endsWith("/prepare-frame") || url.endsWith("/freeze-frame")) value = { ticket: `ticket-${revision}`, revision, camera_seq: body.sequence, width: body.camera.width, height: body.camera.height, image: "data:image/png;base64,c2FtZQ==" };
      else if (url.endsWith("/selection")) value = { selection_token: "selected", selected_count: 1, visible_count: 8, revision };
      else if (url.endsWith("/preview")) value = { revision, image: "data:image/png;base64,cHJldmlldw==" };
      else if (url.endsWith("/operations")) { revision++; value = document(); }
      else value = { state: "viewing", revision, visible_count: 8 - revision, protected_count: protectedCount };
      return { ok: true, json: async () => value };
    },
    require: (name: string) => {
      if (name === "react") return hooks.react;
      if (name === "react/jsx-runtime") return { jsx, jsxs: jsx };
      if (name === "three") return THREE;
      if (name.includes("OrbitControls")) return { OrbitControls: Controls };
      if (name === "./gaussianViewerMetadata") return metadata;
      if (name === "./cloudGaussianEditor") return editor;
      if (name.endsWith(".css")) return {};
      throw new Error(`Unexpected dependency: ${name}`);
    }
  });
  const flush = (until?: () => boolean) => hooks.flush(() => {
    tree = exports.CloudGaussianViewer({ source, metadataUrl: "/metadata.json", alignmentUrl: null, cameraPathUrl: null });
  }, until);
  return { requests, peers, intervals, timeouts, flush, video,
    moveCamera: () => orbit.update(),
    stage: () => find(tree, n => n.type === "div" && n.props.className === "cloud-stage")?.props,
    svg: () => find(tree, n => n.type === "svg")?.props,
    label: () => find(tree, n => n.type === "span" && n.props.className === "cloud-frame-label")?.props.children,
    tick: async () => { const callbacks = [...timeouts.values()]; timeouts.clear(); for (const fn of callbacks) fn(); await flush(); },
    button: (label: string) => find(tree, n => n.type === "button" && n.props.children === label)?.props,
    protection: () => find(tree, n => n.type === "button" && [n.props.children].flat().join("").startsWith("清空本次会话保护"))?.props,
    tool: () => find(tree, n => n.type === "select" && n.props.value === "rectangle").props,
    selectionMode: () => find(tree, n => n.type === "select" && ["visible", "depth", "through"].includes(n.props.value)).props,
    loadImage: () => imageNode.props.onLoad({ currentTarget: imageNode }),
    imageKey: () => imageNode?.key,
    displayFrame: () => frameCallback?.(),
    image: () => find(tree, n => n.type === "img"),
    error: () => find(tree, n => n.props?.role === "alert")?.props.children,
    unmount: hooks.unmount
  };
}

test("cloud page never fetches PLY and waits for the decoded new generation after edits", async () => {
  const h = harness(); await h.flush();
  h.button("连接云端").onClick(); await h.flush(); h.displayFrame(); await h.flush(); await h.tick();
  h.loadImage(); await h.flush();
  h.button("开始选择当前高清画面").onClick(); await h.flush();
  assert.equal(h.button("重新计算选区").disabled, true);
  h.loadImage(); await h.flush(); h.tool().onChange({ target: { value: "box" } }); await h.flush();
  h.button("应用三维盒").onClick(); await h.flush();
  assert.equal(h.button("确认删除选中项").disabled, true);
  h.button("隔离选中").onClick(); await h.flush(); h.loadImage(); await h.flush();
  assert.equal(h.button("确认删除选中项").disabled, false);
  h.button("确认删除选中项").onClick(); await h.flush();
  const key = h.imageKey(); h.loadImage(); await h.flush();
  h.button("重新渲染当前固定视角").onClick(); await h.flush();
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

test("cloud connections and reconnects use only supplied TURN/TCP servers", async () => {
  const servers = [
    { urls: ["stun:10.186.96.23:8082", "turn:10.186.96.23:8082?transport=udp", "turn:10.186.96.23:8082?transport=tcp"], username: "temporary-user", credential: "temporary-credential" },
    { urls: "turn:10.186.96.23:8082?transport=tcp", username: "second-user", credential: "second-credential" },
    { urls: "turn:10.186.96.23:8082?transport=udp" }
  ];
  const h = harness(true, servers);
  try {
    await h.flush(); h.button("连接云端").onClick(); await h.flush();
    h.button("重新连接视频").onClick(); await h.flush();
    assert.equal(h.peers.length, 2);
    for (const peer of h.peers) {
      assert.equal(peer.config.iceTransportPolicy, "relay");
      assert.deepEqual(JSON.parse(JSON.stringify(peer.config.iceServers)), [
        { ...servers[0], urls: ["turn:10.186.96.23:8082?transport=tcp"] },
        { ...servers[1], urls: ["turn:10.186.96.23:8082?transport=tcp"] }
      ]);
    }
    assert.equal(servers[0].urls.length, 3, "do not mutate the supplied configuration");
  } finally { h.unmount(); }
});

test("missing TURN/TCP configuration fails explicitly without UDP or direct fallback", async () => {
  for (const servers of [[], [{ urls: "turn:10.186.96.23:8082?transport=udp" }], [{ urls: "stun:10.186.96.23:8082?transport=tcp" }]]) {
    const h = harness(true, servers);
    try {
      await h.flush(); h.button("连接云端").onClick(); await h.flush();
      assert.match(h.error(), /未提供 TURN\/TCP/);
      assert.equal(h.peers.length, 0);
      assert.ok(h.requests.every(r => !r.url.endsWith("/offer")));
    } finally { h.unmount(); }
  }
});

test("prepared frame must be displayed and is invalidated on navigation before selection", async () => {
  const h = harness();
  try {
    await h.flush(); h.button("连接云端").onClick(); await h.flush();
    h.displayFrame(); await h.flush(); await h.tick();
    assert.equal(h.button("开始选择当前高清画面").disabled, true);
    h.loadImage(); await h.flush();
    assert.equal(h.button("开始选择当前高清画面").disabled, false);
    h.moveCamera(); await h.flush();
    assert.equal(h.image(), null, "a changed camera removes the prepared frame");
    assert.equal(h.button("开始选择当前高清画面").disabled, true);
    await h.tick(); h.loadImage(); await h.flush();
    assert.equal(h.button("开始选择当前高清画面").disabled, false, JSON.stringify({ requests: h.requests.filter(r => /prepare-frame|freeze-prepared/.test(r.url)).map(r => r.url), label: h.label(), error: h.error(), image: !!h.image(), timeouts: h.timeouts.size }));
    h.button("开始选择当前高清画面").onClick(); await h.flush();
    assert.equal(h.error(), undefined);
    assert.ok(h.requests.some(r => r.url.endsWith("/freeze-prepared")));
    assert.equal(h.requests.filter(r => r.url.endsWith("/freeze-frame")).length, 0);
  } finally { h.unmount(); }
});

test("rectangle drag computes selection, keeps delete locked until isolated preview", async () => {
  const h = harness();
  try {
    await h.flush(); h.button("连接云端").onClick(); await h.flush();
    h.displayFrame(); await h.flush(); await h.tick(); h.loadImage(); await h.flush();
    h.button("开始选择当前高清画面").onClick(); await h.flush();
    const svg = h.svg(), element = { setPointerCapture() {}, releasePointerCapture() {} };
    svg.onPointerDown({ currentTarget: element, pointerId: 1, clientX: 210, clientY: 150, button: 0 });
    await h.flush();
    svg.onPointerMove({ currentTarget: element, pointerId: 1, clientX: 490, clientY: 310, button: -1 });
    await h.flush();
    svg.onPointerUp({ currentTarget: element, pointerId: 1, clientX: 490, clientY: 310, button: 0 });
    await h.flush();
    const request = h.requests.find(r => r.url.endsWith("/selection") && r.body.shape === "polygon");
    assert.equal(request.body.mode, "visible");
    assert.equal(request.body.combine, "replace");
    assert.equal(request.body.coverage, false);
    assert.equal(request.body.layer_tolerance, 0.02);
    assert.equal(request.body.polygon.length, 4);
    assert.equal(h.button("确认删除选中项").disabled, true);
    h.loadImage(); await h.flush();
    h.button("隔离选中").onClick(); await h.flush(); h.loadImage(); await h.flush();
    assert.equal(h.button("确认删除选中项").disabled, false);
    assert.match(String(h.label()), /可选择/);
  } finally { h.unmount(); }
});


test("closing and reopening a cloud session clears the displayed protection count", async () => {
  const h = harness();
  try {
    await h.flush(); h.button("连接云端").onClick(); await h.flush();
    h.displayFrame(); await h.flush(); await h.tick(); h.loadImage(); await h.flush();
    h.button("开始选择当前高清画面").onClick(); await h.flush();
    h.tool().onChange({ target: { value: "box" } }); await h.flush();
    h.button("应用三维盒").onClick(); await h.flush();
    h.button("保护选中项").onClick();
    await h.flush(() => /（1）/.test(h.protection().children.join("")) && !h.button("关闭会话").disabled);
    assert.match(h.protection().children.join(""), /（1）/);
    h.button("关闭会话").onClick(); await h.flush();
    assert.match(h.protection().children.join(""), /（0）/);
    h.button("连接云端").onClick(); await h.flush();
    assert.match(h.protection().children.join(""), /（0）/);
    assert.equal(h.protection().disabled, true);
  } finally { h.unmount(); }
});

test("changing selection mode cancels an armed depth pick", async () => {
  const h = harness();
  try {
    await h.flush(); h.button("连接云端").onClick(); await h.flush();
    h.displayFrame(); await h.flush(); await h.tick(); h.loadImage(); await h.flush();
    h.button("开始选择当前高清画面").onClick(); await h.flush();
    h.selectionMode().onChange({ target: { value: "depth" } }); await h.flush();
    h.button("点击画面拾取深度").onClick(); await h.flush();
    h.selectionMode().onChange({ target: { value: "visible" } }); await h.flush();
    h.svg().onPointerDown({ currentTarget: { setPointerCapture() {} }, pointerId: 1, clientX: 210, clientY: 150, button: 0 });
    await h.flush();
    assert.equal(h.requests.some(r => r.url.endsWith("/depth-pick")), false);
    h.selectionMode().onChange({ target: { value: "depth" } }); await h.flush();
    assert.ok(h.button("点击画面拾取深度"));
  } finally { h.unmount(); }
});

test("missing cloud capability stays unavailable without local fallback", async () => {
  const h = harness(false); await h.flush();
  assert.equal(h.button("连接云端").disabled, true);
  assert.equal(h.peers.length, 0);
  assert.ok(h.requests.every(r => !r.url.endsWith(".ply")));
  h.unmount();
});
