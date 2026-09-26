import "./cloudGaussianEditor.css";
import { useEffect, useRef, useState, type RefObject } from "react";
import type { ViewerLeaveRef } from "./gaussianViewerLeave";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { acceptsFrame, captureView, imagePoint, nativeCamera, operationId, rectangle, referenceView, renderSize, restoreView, type CameraView, type Pixel } from "./cloudGaussianEditor";
import { deriveGaussianViewerFrame, deriveUprightRotation, parseGaussianCameraPath, parseGaussianExportMetadata, signedUprightAxis } from "./gaussianViewerMetadata";

export type CloudSource = { job_id: string; variant_id?: string; asset_role: "scene_splat" | "scene_splat_vggt_filtered"; label: string };
type Props = { leaveRef?: ViewerLeaveRef; source: CloudSource; metadataUrl: string | null; cameraPathUrl: string | null; alignmentUrl: string | null; viewRef?: RefObject<CameraView | null>; viewKey?: string };
type Version = { version: string; revision: number; visible_count: number; exported?: boolean };
type Document = { edit_id: string; source: CloudSource; revision: number; visible_count: number; can_undo: boolean; can_redo: boolean; versions: Version[] };
type Session = { session_id: string; token: string; state: string; revision: number; visible_count: number; protected_count: number; error: string | null };
type Frame = { ticket: string; revision: number; camera_seq: number; width: number; height: number; image: string; render_ms?: number };
type Selection = { selection_token: string; selected_count: number; deletable_count?: number; visible_count: number; revision: number };

async function request<T>(url: string, method = "GET", body?: unknown, token?: string): Promise<T> {
  const response = await fetch(url, {
    method, cache: "no-store", headers: { "Content-Type": "application/json", "X-Image3D-Editor": "1", ...(token ? { "X-Editor-Token": token } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  if (!response.ok) {
    const value = await response.json().catch(() => ({}));
    throw new Error(typeof value.detail === "string" ? value.detail : `请求失败（${response.status}），请检查输入或重新连接`);
  }
  return response.json() as Promise<T>;
}

export function CloudGaussianViewer({ source, metadataUrl, cameraPathUrl, alignmentUrl, viewRef, viewKey, leaveRef }: Props) {
  const stage = useRef<HTMLDivElement>(null), video = useRef<HTMLVideoElement>(null), image = useRef<HTMLImageElement>(null);
  const session = useRef<Session | null>(null), peer = useRef<RTCPeerConnection | null>(null);
  const controls = useRef<OrbitControls | null>(null), camera = useRef(new PerspectiveCamera(55, 1, 0.01, 1000000));
  const upright = useRef(new Matrix4()), sequence = useRef(0), generation = useRef(0), revision = useRef(0);
  const frozenRef = useRef(false), frameRef = useRef<Frame | null>(null), active = useRef({ alive: true });
  const working = useRef(false), drag = useRef<Pixel[]>([]);
  const reference = useRef<{ position: Vector3; target: Vector3; up: Vector3 } | null>(null);
  const idleTimer = useRef<number | null>(null), poseSerial = useRef(0), preparedRef = useRef(false), preparingNow = useRef(false);
  const [preparing, setPreparing] = useState(false), [protectedCount, setProtectedCount] = useState(0);
  const [selectionMode, setSelectionMode] = useState<"visible" | "depth" | "through">("visible");
  const [toolsOpen, setToolsOpen] = useState(true);
  const [frameZoom, setFrameZoom] = useState(1), [frameShift, setFrameShift] = useState<Pixel>([0, 0]);
  const pan = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const [capability, setCapability] = useState<{ cloud_available: boolean; reason: string | null } | null>(null);
  const readyRef = useRef(false);
  const [documents, setDocuments] = useState<Document[]>([]), [document, setDocument] = useState<Document | null>(null);
  const [chosen, setChosen] = useState(""), [version, setVersion] = useState("");
  const [state, setState] = useState("idle"), [error, setError] = useState(""), [busy, setBusy] = useState(false), [ready, setReady] = useState(false);
  const [frame, setFrame] = useState<Frame | null>(null), [display, setDisplay] = useState(""), [imageReady, setImageReady] = useState(false);
  const [imageSerial, setImageSerial] = useState(0);
  const [selection, setSelection] = useState<Selection | null>(null), [polygon, setPolygon] = useState<Pixel[]>([]);
  const selectionSerial = useRef(0);
  const [depthPick, setDepthPick] = useState(false);
  const [tool, setTool] = useState<"rectangle" | "lasso" | "box">("rectangle");
  const [combine, setCombine] = useState("replace"), [coverage, setCoverage] = useState(false);
  const [near, setNear] = useState(0.01), [far, setFar] = useState(100);
  const [layerTolerance, setLayerTolerance] = useState(0.02);
  const [minimum, setMinimum] = useState([-1, -1, -1]), [maximum, setMaximum] = useState([1, 1, 1]);
  const [confirmed, setConfirmed] = useState(false), [exportState, setExportState] = useState("");
  const [previewed, setPreviewed] = useState(false);

  useEffect(() => {
    const leave = async () => {
      if (working.current || session.current) {
        window.alert("请先等待云端操作完成，并明确点击“关闭会话”后再切换模型或进入本地；不会后台抢占云会话。");
        return false;
      }
      return true;
    };
    if (leaveRef) leaveRef.current = leave;
    return () => { if (leaveRef?.current === leave) leaveRef.current = null; };
  }, [leaveRef]);

  function sessionCall<T>(suffix: string, method = "POST", body?: unknown) {
    const s = session.current;
    if (!s) throw new Error("请先连接云端会话");
    return request<T>(`/api/gaussian-render-sessions/${s.session_id}${suffix}`, method, body, s.token);
  }
  async function run(work: () => Promise<void>) {
    if (working.current) return;
    const life = active.current;
    working.current = true; setBusy(true); setError("");
    try { await work(); }
    catch (e) { if (life.alive) setError(e instanceof Error ? e.message : String(e)); }
    finally { working.current = false; if (life.alive) setBusy(false); }
  }
  async function refreshDocuments() {
    const result = await request<{ edits: Document[] }>(`/api/gaussian-edits?job_id=${encodeURIComponent(source.job_id)}`);
    if (!active.current.alive) return;
    setDocuments(result.edits.filter(d => (d.source.variant_id ?? undefined) === source.variant_id && d.source.asset_role === source.asset_role));
  }
  function snapshot() {
    if (!stage.current || !readyRef.current) throw new Error("相机初始化尚未完成");
    const box = stage.current.getBoundingClientRect();
    const [width, height] = renderSize(box.width, box.height);
    return nativeCamera(camera.current, upright.current, width, height);
  }
  function showFrame(value: Frame) {
    frameRef.current = value; setFrame(value); setDisplay(value.image); setImageReady(false); setImageSerial(v => v + 1);
    setFrameZoom(1); setFrameShift([0, 0]);
    setSelection(null); setPolygon([]); setPreviewed(false); setConfirmed(false); setDepthPick(false);
  }
  async function prepareFrame() {
    if (preparingNow.current || !session.current || frozenRef.current || !video.current) return;
    const serial = poseSerial.current, epoch = generation.current;
    preparingNow.current = true; setPreparing(true);
    try {
      const value = await sessionCall<Frame>("/prepare-frame", "POST", { sequence: ++sequence.current, camera: snapshot() });
      if (active.current.alive && serial === poseSerial.current && epoch === generation.current && value.revision === revision.current) {
        preparedRef.current = true; showFrame(value); setState("prepared"); setError("");
      }
    } catch (e) {
      if (active.current.alive && serial === poseSerial.current && !frozenRef.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      preparingNow.current = false; if (active.current.alive) setPreparing(false);
      if (serial !== poseSerial.current && session.current && !frozenRef.current && active.current.alive) schedulePrepare();
    }
  }
  function schedulePrepare() {
    if (idleTimer.current !== null) window.clearTimeout(idleTimer.current);
    idleTimer.current = window.setTimeout(() => { idleTimer.current = null; void prepareFrame(); }, 500);
  }
  function cameraChanged() {
    if (frozenRef.current) return;
    ++poseSerial.current; preparedRef.current = false;
    if (frameRef.current) { frameRef.current = null; setFrame(null); setDisplay(""); setImageReady(false); }
    setState(current => current === "prepared" ? "viewing" : current);
    if (session.current) schedulePrepare();
  }
  async function freezePrepared() {
    const value = frameRef.current;
    if (!value || !preparedRef.current || !imageReady) throw new Error("请等待高清画面显示完成");
    await sessionCall("/freeze-prepared", "POST", { ticket: value.ticket, expected_revision: value.revision });
    frozenRef.current = true; preparedRef.current = false;
    if (controls.current) controls.current.enabled = false;
    setState("editing_frozen"); setSelection(null); setPolygon([]);
  }
  async function fixedFrame() {
    frozenRef.current = true;
    if (controls.current) controls.current.enabled = false;
    setState("editing_frozen"); setImageReady(false);
    const value = await sessionCall<Frame>("/freeze-frame", "POST", { sequence: ++sequence.current, camera: snapshot() });
    if (active.current.alive && value.revision === revision.current) showFrame(value);
  }
  async function connectMedia() {
    if (!video.current) return;
    peer.current?.close();
    const epoch = ++generation.current, expectedRevision = revision.current;
    const config = await sessionCall<RTCConfiguration>("/ice", "GET");
    if (!active.current.alive) return;
    // SDP 路径的 UDP 不可达；首次连接和重连均只使用已下发的 TURN/TCP。
    const iceServers = (config.iceServers ?? []).map(server => ({
      ...server,
      urls: (Array.isArray(server.urls) ? server.urls : [server.urls])
        .filter(url => /^turns?:[^?]+\?transport=tcp$/i.test(url))
    })).filter(server => server.urls.length > 0);
    if (!iceServers.length) throw new Error("服务器未提供 TURN/TCP 地址，无法连接云端视频");
    const pc = new RTCPeerConnection({ ...config, iceServers, iceTransportPolicy: "relay" }); peer.current = pc;
    const channel = pc.createDataChannel("camera", { ordered: false });
    pc.addTransceiver("video", { direction: "recvonly" });
    let lastCamera = "", paused = false;
    const sendCamera = () => {
      if (channel.readyState !== "open" || channel.bufferedAmount > 16384) return;
      if (paused !== window.document.hidden) { paused = window.document.hidden; channel.send(JSON.stringify({ paused })); return; }
      if (frozenRef.current || paused) return;
      const value = snapshot(), text = JSON.stringify(value);
      if (text !== lastCamera) {
        if (lastCamera) cameraChanged();
        channel.send(JSON.stringify({ sequence: ++sequence.current, camera: value })); lastCamera = text;
      }
    };
    const timer = window.setInterval(sendCamera, 34);
    channel.onopen = sendCamera;
    channel.onclose = () => window.clearInterval(timer);
    pc.ontrack = event => {
      const element = video.current;
      if (!element || generation.current !== epoch) return;
      element.srcObject = new MediaStream([event.track]);
      const displayed = () => {
        if (acceptsFrame(generation.current, epoch, revision.current, expectedRevision) && !frozenRef.current && !preparedRef.current) {
          frameRef.current = null; setFrame(null); setDisplay(""); setState("viewing");
          if (!preparedRef.current) schedulePrepare();
        }
      };
      if (typeof element.requestVideoFrameCallback === "function") element.requestVideoFrameCallback(displayed);
      else element.onloadeddata = displayed;
      void element.play().catch(() => setError("浏览器暂停了视频，请点击视频播放"));
    };
    pc.onconnectionstatechange = () => {
      if (peer.current !== pc) return;
      if (["failed", "disconnected"].includes(pc.connectionState)) setError("媒体连接中断，请重新连接；已确认编辑保留");
      if (pc.connectionState === "closed") window.clearInterval(timer);
    };
    try {
      await pc.setLocalDescription(await pc.createOffer());
      if (pc.iceGatheringState !== "complete") await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => { pc.removeEventListener("icegatheringstatechange", check); reject(new Error("TURN 候选收集超时")); }, 15000);
        const check = () => { if (pc.iceGatheringState === "complete") { window.clearTimeout(timeout); pc.removeEventListener("icegatheringstatechange", check); resolve(); } };
        pc.addEventListener("icegatheringstatechange", check); check();
      });
      const answer = await sessionCall<RTCSessionDescriptionInit>("/offer", "POST", { type: "offer", sdp: pc.localDescription?.sdp });
      if (peer.current === pc && active.current.alive) await pc.setRemoteDescription(answer);
      else pc.close();
    } catch (e) { window.clearInterval(timer); pc.close(); throw e; }
  }
  async function open() {
    const life = active.current;
    let doc: Document;
    if (chosen) doc = await request<Document>(`/api/gaussian-edits/${chosen}`);
    else {
      const { label: _label, ...identity } = source;
      doc = await request<Document>("/api/gaussian-edits", "POST", identity);
    }
    if (!life.alive) return;
    setDocument(doc); setChosen(doc.edit_id); setExportState("");
    const s = await request<Session>("/api/gaussian-render-sessions", "POST", { edit_id: doc.edit_id, version: version || null });
    if (!life.alive) { await request(`/api/gaussian-render-sessions/${s.session_id}`, "DELETE", undefined, s.token); return; }
    session.current = s; setState("loading");
    while (life.alive) {
      const status = await sessionCall<Session>("", "GET");
      if (status.state === "error") { setState("error"); throw new Error(status.error ?? "模型加载失败"); }
      if (status.state === "viewing") {
        revision.current = status.revision;
        setProtectedCount(status.protected_count);
        setDocument(d => d ? { ...d, revision: status.revision, visible_count: status.visible_count } : d);
        frozenRef.current = false;
        if (controls.current) controls.current.enabled = true;
        setState("connecting"); await connectMedia(); await refreshDocuments(); return;
      }
      if (status.state !== "loading") throw new Error("会话已关闭");
      await new Promise(resolve => window.setTimeout(resolve, 500));
    }
  }
  async function disconnect() {
    ++poseSerial.current; preparedRef.current = false;
    if (idleTimer.current !== null) window.clearTimeout(idleTimer.current);
    ++generation.current; peer.current?.close(); peer.current = null;
    if (video.current) video.current.srcObject = null;
    if (session.current) await sessionCall("", "DELETE");
    session.current = null; frozenRef.current = false; frameRef.current = null;
    setProtectedCount(0);
    setFrame(null); setDisplay(""); setSelection(null); setDepthPick(false); setState("idle");
    if (controls.current) controls.current.enabled = true;
    await refreshDocuments();
  }
  async function resume() {
    preparedRef.current = false; ++poseSerial.current;
    setImageReady(false); ++generation.current; peer.current?.close();
    await sessionCall("/resume"); frozenRef.current = false;
    if (controls.current) controls.current.enabled = true;
    setState("connecting"); setSelection(null); await connectMedia();
  }
  function binding() {
    if (!frame || !imageReady) throw new Error("请先固定画面，等待高清图显示完成");
    return { ticket: frame.ticket, expected_revision: frame.revision };
  }
  async function select(shape: "polygon" | "box" | "clear", points = polygon, operation = combine) {
    const serial = ++selectionSerial.current;
    const result = await sessionCall<Selection>("/selection", "POST", {
      ...binding(), shape, polygon: points, minimum, maximum, depth_range: [near, far],
      combine: operation, coverage: selectionMode === "through" && coverage, mode: selectionMode, layer_tolerance: layerTolerance
    });
    if (serial !== selectionSerial.current) return;
    setSelection(result); setPreviewed(false); setConfirmed(false); setDisplay(frame!.image);
    if (!result.selected_count || shape === "box") return;
    const highlighted = await sessionCall<{ image: string; revision: number }>("/preview", "POST", {
      ...binding(), selection_token: result.selection_token, mode: "highlight"
    });
    if (serial === selectionSerial.current && highlighted.revision === revision.current) {
      setDisplay(highlighted.image); setImageReady(false); setImageSerial(v => v + 1);
    }
  }
  async function preview(mode: string) {
    if (!selection) return;
    const result = await sessionCall<{ image: string; revision: number }>("/preview", "POST", { ...binding(), selection_token: selection.selection_token, mode });
    if (result.revision === revision.current) { setDisplay(result.image); setImageReady(false); setImageSerial(v => v + 1); setPreviewed(mode !== "highlight"); }
  }
  async function protect(kind: "add" | "remove" | "clear") {
    const result = await sessionCall<{ protected_count: number }>("/protection", "POST", {
      ...binding(), kind, selection_token: kind === "clear" ? null : selection?.selection_token
    });
    setProtectedCount(result.protected_count); setSelection(null); setPreviewed(false); setConfirmed(false);
    setDisplay(frame!.image); setImageReady(false); setImageSerial(v => v + 1);
  }
  async function operate(kind: string) {
    const previousRevision = revision.current;
    let result: Pick<Document, "revision" | "visible_count" | "can_undo" | "can_redo">;
    try {
      result = await sessionCall("/operations", "POST", {
        ...binding(), operation_id: operationId(), kind, selection_token: kind === "delete" ? selection?.selection_token : null, confirm_large: confirmed
      });
    } catch (error) {
      const status = await sessionCall<Session>("", "GET");
      if (status.revision !== previousRevision) {
        revision.current = status.revision;
        setDocument(await request<Document>(`/api/gaussian-edits/${document!.edit_id}`));
        await fixedFrame();
        throw new Error("操作响应中断，但提交已完成；已恢复服务器最新版本，请勿重复删除");
      }
      throw error;
    }
    revision.current = result.revision; setDocument(d => d ? { ...d, ...result } : d);
    setImageReady(false); setSelection(null); setPreviewed(false);
    await fixedFrame();
  }
  async function save() {
    const value = await sessionCall<Version>("/versions", "POST", { expected_revision: revision.current });
    const doc = await request<Document>(`/api/gaussian-edits/${document!.edit_id}`);
    setDocument(doc); setExportState(`已保存 ${value.version}`); await refreshDocuments();
  }
  async function exportVersion(value: string) {
    await sessionCall(`/exports/${value}`);
    setExportState(`正在导出 ${value}`);
    const life = active.current;
    while (life.alive) {
      const result = await request<{ status: string; error?: string }>(`/api/gaussian-edits/${document!.edit_id}/versions/${value}/export`);
      if (result.status === "error") throw new Error(result.error ?? "导出失败");
      if (result.status === "done") { setExportState(value); return; }
      await new Promise(resolve => window.setTimeout(resolve, 1000));
    }
  }

  useEffect(() => {
    const life = { alive: true }; active.current = life;
    let orbit: OrbitControls | null = null;
    const canvas = stage.current!;
    const zoomFrozen = (event: WheelEvent) => {
      if (!frozenRef.current) return;
      event.preventDefault();
      setFrameZoom(v => Math.min(6, Math.max(1, v * (event.deltaY < 0 ? 1.2 : 1 / 1.2))));
    };
    canvas.addEventListener("wheel", zoomFrozen, { passive: false });
    void (async () => {
      try {
        const cap = await request<{ cloud_available: boolean; reason: string | null }>("/api/gaussian-editor/capabilities");
        if (!life.alive) return; setCapability(cap);
        if (!metadataUrl) throw new Error("源模型缺少导出元数据，不能初始化相机");
        const metadata = parseGaussianExportMetadata(await request(metadataUrl));
        const path = cameraPathUrl ? await request(cameraPathUrl).then(parseGaussianCameraPath).catch(() => null) : null;
        const alignment = alignmentUrl ? await request(alignmentUrl).catch(() => null) : null;
        if (!life.alive) return;
        const basis = path ? deriveGaussianViewerFrame(metadata, path) : null;
        const rotation = deriveUprightRotation(metadata, alignment);
        const transform = rotation ? new Matrix4().set(...[...rotation[0], 0, ...rotation[1], 0, ...rotation[2], 0, 0, 0, 0, 1] as Parameters<Matrix4["set"]>) : new Matrix4();
        upright.current = transform;
        const center = new Vector3(...(metadata.scene_center ?? basis?.center ?? [0, 0, 0])).applyMatrix4(transform);
        const up = new Vector3(...(basis?.up ?? [0, 0, 1])).transformDirection(transform);
        camera.current.up.copy(rotation ? new Vector3(...signedUprightAxis(up.toArray())) : up);
        const radius = metadata.scene_radius_p95 ?? 1.5;
        const target = viewRef?.current && viewRef.current.key === viewKey
          ? restoreView(camera.current, viewRef.current)
          : referenceView(camera.current, basis ? path : null, metadata.world_from_normalized, transform, center, radius);
        reference.current = { position: camera.current.position.clone(), target: target.clone(), up: camera.current.up.clone() };
        orbit = new OrbitControls(camera.current, stage.current!); controls.current = orbit;
        orbit.enableDamping = false;
        orbit.rotateSpeed = 0.45; orbit.zoomSpeed = 0.8; orbit.panSpeed = 0.7;
        orbit.screenSpacePanning = true;
        orbit.minDistance = Math.max(radius * 0.08, 0.01);
        orbit.maxDistance = Math.max(radius * 20, 20);
        orbit.target.copy(target); orbit.addEventListener("change", cameraChanged);
        orbit.update(); readyRef.current = true; setReady(true); await refreshDocuments();
      } catch (e) { if (life.alive) setError(String(e)); }
    })();
    const heartbeat = window.setInterval(() => {
      if (session.current) void sessionCall<Session>("", "GET").then(s => {
        if (life.alive && s.state === "error") { setState("error"); setError(s.error ?? "云会话失败"); }
      }).catch(e => { if (life.alive) setError(String(e)); });
    }, 5000);
    return () => {
      life.alive = false; readyRef.current = false; ++generation.current; ++poseSerial.current;
      window.clearInterval(heartbeat); canvas.removeEventListener("wheel", zoomFrozen);
      if (idleTimer.current !== null) window.clearTimeout(idleTimer.current);
      if (viewRef && viewKey && orbit) viewRef.current = captureView(viewKey, camera.current, orbit.target);
      orbit?.removeEventListener("change", cameraChanged); orbit?.dispose(); controls.current = null;
      peer.current?.close();
      const s = session.current; session.current = null;
      if (s) void request(`/api/gaussian-render-sessions/${s.session_id}`, "DELETE", undefined, s.token).catch(() => {});
    };
  }, [metadataUrl, cameraPathUrl, alignmentUrl, source.job_id, source.variant_id, source.asset_role]);

  useEffect(() => {
    setSelection(null); setPreviewed(false); setConfirmed(false); setDepthPick(false);
  }, [tool, selectionMode, near, far, layerTolerance, minimum, maximum, coverage, combine]);

  function pointer(event: ReactPointerEvent<SVGSVGElement>, phase: "start" | "move" | "end") {
    if (!frame || !imageReady || busy || !stage.current) return;
    const box = stage.current.getBoundingClientRect();
    if (phase === "start" && event.button === 2) {
      event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
      pan.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY }; return;
    }
    if (pan.current?.pointerId === event.pointerId) {
      if (phase === "move") {
        setFrameShift(([x, y]) => [x + event.clientX - pan.current!.x, y + event.clientY - pan.current!.y]);
        pan.current.x = event.clientX; pan.current.y = event.clientY;
      } else if (phase === "end") { pan.current = null; event.currentTarget.releasePointerCapture(event.pointerId); }
      return;
    }
    const point = imagePoint(event.clientX, event.clientY, box, frame.width, frame.height, frameZoom, frameShift);
    if (phase === "start") {
      if (event.button !== 0 || !point) return;
      if (depthPick) {
        void run(async () => {
          const result = await sessionCall<{ depth: number }>("/depth-pick", "POST", { ...binding(), pixel: point });
          const span = Math.max(0.02, result.depth * 0.02);
          setNear(Math.max(0.01, result.depth - span)); setFar(result.depth + span); setDepthPick(false);
        });
        return;
      }
      if (tool === "box") return;
      event.currentTarget.setPointerCapture(event.pointerId);
      drag.current = [point]; ++selectionSerial.current;
      setPolygon([]); setSelection(null); setPreviewed(false); setConfirmed(false);
      return;
    }
    if (!drag.current.length) return;
    if (phase === "move" && point) {
      const points = tool === "rectangle" ? rectangle(drag.current[0], point) : [...drag.current, point].slice(0, 128);
      if (tool === "lasso") drag.current = points;
      setPolygon(points);
    }
    if (phase === "end") {
      const points = point ? tool === "rectangle" ? rectangle(drag.current[0], point) : [...drag.current, point].slice(0, 128) : polygon;
      drag.current = []; event.currentTarget.releasePointerCapture(event.pointerId);
      if (points.length >= 3 && (tool !== "rectangle" || Math.abs((points[2][0] - points[0][0]) * (points[2][1] - points[0][1])) >= 1)) {
        setPolygon(points);
        void run(() => select("polygon", points, event.shiftKey ? "add" : event.ctrlKey ? "subtract" : combine));
      }
    }
  }
  const frozen = state === "editing_frozen", writable = !version;
  const selectable = frozen && imageReady && !busy;
  const deletableCount = selection?.deletable_count ?? selection?.selected_count ?? 0;
  const large = !!selection && deletableCount > selection.visible_count / 2;
  const statusLabel: Record<string, string> = { idle: "未连接", loading: "加载模型", connecting: "连接视频", viewing: "云端观看", prepared: "高清已就绪", editing_frozen: "固定帧修剪" };
  return <section className="cloud-editor" aria-label="云端高斯修剪工作区">
    <header className="cloud-editor-heading"><div><strong>{source.label} · 云端修剪</strong><small>源模型只读 · 当前编辑未重新评价 · 不继承碰撞漫游</small></div><span role="status">{state === "prepared" && !imageReady ? "等待高清图显示" : statusLabel[state] ?? state}{busy ? " · 处理中" : ""}</span></header>
    {!capability?.cloud_available && <p className="cloud-notice">{capability?.reason ?? "正在读取云端能力…"}。8082 TCP/UDP 已确认开放，实际 TURN 部署与连通仍须验证。</p>}
    {error && <p className="cloud-error" role="alert">{error}</p>}
    <div className="cloud-toolbar">
      <label>编辑文档<select value={chosen} disabled={!!session.current || busy} onChange={e => { setChosen(e.target.value); setVersion(""); setDocument(documents.find(d => d.edit_id === e.target.value) ?? null); }}><option value="">从 Original 新建副本</option>{documents.map(d => <option key={d.edit_id} value={d.edit_id}>{d.edit_id.slice(0, 8)} · r{d.revision} · {d.visible_count.toLocaleString()}</option>)}</select></label>
      <label>打开版本<select value={version} disabled={!!session.current || busy} onChange={e => setVersion(e.target.value)}><option value="">当前文档（可编辑）</option>{(documents.find(d => d.edit_id === chosen)?.versions ?? []).map(v => <option key={v.version}>{v.version}</option>)}</select></label>
      {!session.current ? <button disabled={busy || !ready || !capability?.cloud_available} onClick={() => void run(open)}>连接云端</button> : <button disabled={busy} onClick={() => void run(disconnect)}>关闭会话</button>}
      {session.current && <button disabled={busy || state === "loading"} onClick={() => void run(resume)}>重新连接视频</button>}
      {frozen ? <button disabled={busy} onClick={() => void run(resume)}>恢复交互观看</button> : <button disabled={busy || !imageReady || !preparedRef.current} title={!imageReady ? "请停稳并等待高清图像显示完成" : "固定这张已呈现的高清图像"} onClick={() => void run(freezePrepared)}>开始选择当前高清画面</button>}
      {session.current && !frozen && <button disabled={busy || preparing} onClick={() => void prepareFrame()}>刷新高清画面</button>}
      {session.current && <button onClick={() => { const value = reference.current; if (!value || !controls.current || frozenRef.current) return; camera.current.position.copy(value.position); camera.current.up.copy(value.up); controls.current.target.copy(value.target); controls.current.update(); }} disabled={busy || frozen}>参考视角</button>}
      <button aria-expanded={toolsOpen} onClick={() => setToolsOpen(v => !v)}>{toolsOpen ? "收起工具" : "展开工具"}</button>
    </div>
    <div className={toolsOpen ? "cloud-workspace" : "cloud-workspace cloud-workspace-wide"}>
      <div ref={stage} className="cloud-stage" aria-label={frozen ? "固定高清画面：左键框选，右键平移，滚轮放大" : "云端画面：左键旋转，右键平移，滚轮缩放"} tabIndex={0} onPointerDown={e => {
        if (!frozen && controls.current) { controls.current.rotateSpeed = e.shiftKey ? 0.12 : 0.45; controls.current.panSpeed = e.shiftKey ? 0.2 : 0.7; }
      }} onPointerUp={() => {
        if (controls.current) { controls.current.rotateSpeed = 0.45; controls.current.panSpeed = 0.7; }
      }} onPointerCancel={() => {
        if (controls.current) { controls.current.rotateSpeed = 0.45; controls.current.panSpeed = 0.7; }
      }} onKeyDown={e => {
        if (frozenRef.current || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "-"].includes(e.key)) return;
        e.preventDefault(); const target = controls.current?.target; if (!target) return;
        const offset = camera.current.position.clone().sub(target);
        if (e.key === "+" || e.key === "-") offset.multiplyScalar(e.key === "+" ? 0.9 : 1.1);
        else offset.applyAxisAngle(e.key === "ArrowLeft" || e.key === "ArrowRight" ? camera.current.up : new Vector3(1, 0, 0).applyQuaternion(camera.current.quaternion), e.key === "ArrowLeft" || e.key === "ArrowUp" ? 0.08 : -0.08);
        camera.current.position.copy(target).add(offset); controls.current?.update();
      }}>
        <video ref={video} autoPlay playsInline muted controls={false} onClick={() => void video.current?.play()} />
        {display && <div className="cloud-frame-layer" style={{ transform: `translate(${frameShift[0]}px, ${frameShift[1]}px) scale(${frameZoom})` }}>
          <img key={imageSerial} ref={image} src={display} alt="绑定当前相机与编辑版本的高清固定帧" onLoad={event => { if (image.current === event.currentTarget && frameRef.current && frameRef.current.revision === revision.current) setImageReady(true); }} />
          {frozen && frame && <svg viewBox={`0 0 ${frame.width} ${frame.height}`} aria-label="固定帧选区" onContextMenu={e => e.preventDefault()} onPointerDown={e => pointer(e, "start")} onPointerMove={e => pointer(e, "move")} onPointerUp={e => pointer(e, "end")} onPointerCancel={() => { drag.current = []; pan.current = null; setPolygon([]); }}><polygon points={polygon.map(p => p.join(",")).join(" ")} /></svg>}
        </div>}
        {!display && state !== "viewing" && <div className="cloud-placeholder">{state === "idle" ? "连接后由服务器渲染；此视口不下载高斯 PLY" : "等待服务器画面…"}</div>}
        <span className="cloud-frame-label">{frozen ? `固定 r${frame?.revision ?? revision.current} · ${imageReady ? "可选择：左键拖框；右键平移；滚轮放大" : "请等待高清图片"}` : preparedRef.current && imageReady ? "高清画面已就绪：点击上方开始选择（固定的就是当前图）" : preparing ? "正在准备高清画面，请停稳相机…" : "导航：左键旋转 / 右键平移 / 滚轮缩放 · 松开后准备高清图"}{frame?.render_ms ? ` · 渲染 ${Math.round(frame.render_ms)}ms` : ""}</span>
      </div>
      {toolsOpen && <aside className="cloud-tools" aria-label="修剪工具">
        <strong>先选准，再删除</strong>
        <small>{frozen ? "左键拖框自动计算；在放大图片上右键拖动可平移" : "停稳相机，等高清画面就绪后点击「开始选择」"}</small>
        <label>选择工具<select value={tool} onChange={e => { const next = e.target.value as typeof tool; setTool(next); setSelectionMode(next === "box" ? "through" : "visible"); setPolygon([]); }}><option value="rectangle">矩形</option><option value="lasso">套索</option><option value="box">三维轴对齐盒（穿透体积）</option></select></label>
        <label>选择深度<select value={selectionMode} disabled={tool === "box"} onChange={e => setSelectionMode(e.target.value as typeof selectionMode)}><option value="visible">仅可见贡献（优先保护后景）</option><option value="depth">指定深度薄层</option><option value="through">穿透选择（包含后景）</option></select></label>
        <label>组合<select value={combine} onChange={e => setCombine(e.target.value)}><option value="replace">新建选区</option><option value="add">增加到选区</option><option value="subtract" disabled={!selection?.selected_count}>从已有选区移除（不是删除高斯）</option></select></label>
        {tool !== "box" ? <>
          {selectionMode === "visible" && <label>表层厚度（相机深度比例）<input type="number" min="0" max="0.2" step="0.005" value={layerTolerance} onChange={e => setLayerTolerance(Math.max(0, Math.min(0.2, Number(e.target.value))))} /></label>}
          {selectionMode === "depth" && <button disabled={!selectable} onClick={() => setDepthPick(true)}>{depthPick ? "请点击高清图中的目标" : "点击画面拾取深度"}</button>}
          <label>近深度（normalized，非米）<input type="number" min="0.01" step="0.01" value={near} onChange={e => setNear(Number(e.target.value))} /></label>
          <label>远深度（normalized，非米）<input type="number" min="0.02" step="0.01" value={far} onChange={e => setFar(Number(e.target.value))} /></label>
          {selectionMode === "through" && <label className="cloud-check"><input type="checkbox" checked={coverage} onChange={e => setCoverage(e.target.checked)} />扩大覆盖候选（可能多选背景）</label>}
        </> : <>{["X", "Y", "Z"].map((axis, i) => <div className="cloud-axis" key={axis}><label>{axis} 最小<input type="number" step="0.1" value={minimum[i]} onChange={e => setMinimum(v => v.map((x, j) => j === i ? Number(e.target.value) : x))} /></label><label>{axis} 最大<input type="number" step="0.1" value={maximum[i]} onChange={e => setMaximum(v => v.map((x, j) => j === i ? Number(e.target.value) : x))} /></label></div>)}</>}
        <small>{tool === "box" ? "三维盒使用原始 normalized 坐标。" : selectionMode === "visible" ? "只取足够不透明的主表层；弱于5%的悬浮前层可能被跳过，不能保证后景绝不被选中。修剪弱悬浮项请用深度薄层，多视角检查后再删。" : "深度/穿透模式按投影选择，必须检查隔离和删除后预览。"}</small>
        <button disabled={!selectable || (tool !== "box" && polygon.length < 3)} onClick={() => void run(() => select(tool === "box" ? "box" : "polygon"))}>{tool === "box" ? "应用三维盒" : "重新计算选区"}</button>
        <button disabled={!selectable} onClick={() => void run(() => select("clear"))}>清空选择</button>
        <output aria-live="polite">{busy ? "正在计算/预览…" : selection ? `选中 ${selection.selected_count.toLocaleString()}，可删除 ${deletableCount.toLocaleString()} / ${selection.visible_count.toLocaleString()}` : polygon.length ? "已画框，尚未计算（或条件已变化）" : `保留 ${document?.visible_count.toLocaleString() ?? "—"} 高斯`}</output>
        {selection?.selected_count === 0 && <small>没有命中；检查组合是否为「新建选区」、深度及选区位置。</small>}
        <div className="cloud-preview-buttons">{[["highlight", "叠色参考（不可直接删除）"], ["isolated", "隔离选中"], ["after_delete", "删除后预览"]].map(([mode, label]) => <button key={mode} disabled={!selectable || !deletableCount} onClick={() => void run(() => preview(mode))}>{label}</button>)}</div>
        <small>叠色仅是图像合成参考；确认删除前必须检查隔离或删除后预览，必要时换视角复查。</small>
        {selection?.selected_count ? <div className="cloud-preview-buttons"><button disabled={!selectable} onClick={() => void run(() => protect("add"))}>保护选中项</button><button disabled={!selectable || !protectedCount} onClick={() => void run(() => protect("remove"))}>解除选中保护</button></div> : null}
        <button disabled={!selectable || !protectedCount} onClick={() => void run(() => protect("clear"))}>清空本次会话保护（{protectedCount.toLocaleString()}）</button>
        <small>保护项不会被本次会话的删除操作移除；关闭会话后保护集合清空，已保存的编辑历史不变。</small>
        {large && <label className="cloud-check"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />确认删除超过一半的可见高斯</label>}
        <button className="cloud-delete" disabled={!selectable || !writable || !previewed || !deletableCount || deletableCount === selection?.visible_count || (large && !confirmed)} title={!selection ? "请先画框并计算" : !previewed ? "请先隔离或查看删除后预览" : !deletableCount ? "选区为空或全被保护" : ""} onClick={() => void run(() => operate("delete"))}>确认删除选中项</button>
        <div className="cloud-preview-buttons"><button disabled={!selectable || !writable || !document?.can_undo} onClick={() => void run(() => operate("undo"))}>撤销</button><button disabled={!selectable || !writable || !document?.can_redo} onClick={() => void run(() => operate("redo"))}>重做</button></div>
        {frozen && <button disabled={busy} onClick={() => void run(fixedFrame)}>重新渲染当前固定视角</button>}
      </aside>}
    </div>
    <footer className="cloud-versions"><button disabled={busy || !session.current || !writable || state === "loading"} onClick={() => void run(save)}>保存不可变版本</button><span role="status">{exportState.startsWith("v") ? "导出完成" : exportState}</span>{document?.versions.map(v => <span className="cloud-version" key={v.version}><span>{v.version} · {v.visible_count.toLocaleString()}</span><button disabled={busy || !session.current} onClick={() => void run(() => exportVersion(v.version))}>导出</button>{(v.exported || exportState === v.version) && <><a href={`/api/gaussian-edits/${document.edit_id}/versions/${v.version}/assets/bundle.zip`}>下载 ZIP</a><a href={`/api/gaussian-edits/${document.edit_id}/versions/${v.version}/assets/scene.ply`}>PLY</a></>}</span>)}</footer>
  </section>;
}
