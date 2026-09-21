import "./cloudGaussianEditor.css";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { acceptsFrame, imagePoint, nativeCamera, operationId, rectangle, renderSize, type Pixel } from "./cloudGaussianEditor";
import { deriveGaussianViewerFrame, deriveUprightRotation, parseGaussianCameraPath, parseGaussianExportMetadata, signedUprightAxis } from "./gaussianViewerMetadata";

export type CloudSource = { job_id: string; variant_id?: string; asset_role: "scene_splat" | "scene_splat_vggt_filtered"; label: string };
type Props = { source: CloudSource; metadataUrl: string | null; cameraPathUrl: string | null; alignmentUrl: string | null };
type Version = { version: string; revision: number; visible_count: number; exported?: boolean };
type Document = { edit_id: string; source: CloudSource; revision: number; visible_count: number; can_undo: boolean; can_redo: boolean; versions: Version[] };
type Session = { session_id: string; token: string; state: string; revision: number; visible_count: number; error: string | null };
type Frame = { ticket: string; revision: number; camera_seq: number; width: number; height: number; image: string };
type Selection = { selection_token: string; selected_count: number; visible_count: number; revision: number };

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

export function CloudGaussianViewer({ source, metadataUrl, cameraPathUrl, alignmentUrl }: Props) {
  const stage = useRef<HTMLDivElement>(null), video = useRef<HTMLVideoElement>(null), image = useRef<HTMLImageElement>(null);
  const session = useRef<Session | null>(null), peer = useRef<RTCPeerConnection | null>(null);
  const controls = useRef<OrbitControls | null>(null), camera = useRef(new PerspectiveCamera(55, 1, 0.01, 1000000));
  const upright = useRef(new Matrix4()), sequence = useRef(0), generation = useRef(0), revision = useRef(0);
  const frozenRef = useRef(false), frameRef = useRef<Frame | null>(null), active = useRef({ alive: true });
  const working = useRef(false), drag = useRef<Pixel[]>([]);
  const [capability, setCapability] = useState<{ cloud_available: boolean; reason: string | null } | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]), [document, setDocument] = useState<Document | null>(null);
  const [chosen, setChosen] = useState(""), [version, setVersion] = useState("");
  const [state, setState] = useState("idle"), [error, setError] = useState(""), [busy, setBusy] = useState(false), [ready, setReady] = useState(false);
  const [frame, setFrame] = useState<Frame | null>(null), [display, setDisplay] = useState(""), [imageReady, setImageReady] = useState(false);
  const [imageSerial, setImageSerial] = useState(0);
  const [selection, setSelection] = useState<Selection | null>(null), [polygon, setPolygon] = useState<Pixel[]>([]);
  const [tool, setTool] = useState<"rectangle" | "lasso" | "box">("rectangle");
  const [combine, setCombine] = useState("replace"), [coverage, setCoverage] = useState(false);
  const [near, setNear] = useState(0.01), [far, setFar] = useState(100);
  const [minimum, setMinimum] = useState([-1, -1, -1]), [maximum, setMaximum] = useState([1, 1, 1]);
  const [confirmed, setConfirmed] = useState(false), [exportState, setExportState] = useState("");
  const [previewed, setPreviewed] = useState(false);

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
    if (!stage.current || !ready) throw new Error("相机初始化尚未完成");
    const box = stage.current.getBoundingClientRect();
    const [width, height] = renderSize(box.width, box.height);
    return nativeCamera(camera.current, upright.current, width, height);
  }
  function showFrame(value: Frame) {
    frameRef.current = value; setFrame(value); setDisplay(value.image); setImageReady(false); setImageSerial(v => v + 1);
    setSelection(null); setPolygon([]); setPreviewed(false); setConfirmed(false);
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
    const pc = new RTCPeerConnection(config); peer.current = pc;
    const channel = pc.createDataChannel("camera", { ordered: false });
    pc.addTransceiver("video", { direction: "recvonly" });
    let lastCamera = "", paused = false;
    const sendCamera = () => {
      if (channel.readyState !== "open" || channel.bufferedAmount > 16384) return;
      if (paused !== window.document.hidden) { paused = window.document.hidden; channel.send(JSON.stringify({ paused })); return; }
      if (frozenRef.current || paused) return;
      const value = snapshot(), text = JSON.stringify(value);
      if (text !== lastCamera) { channel.send(JSON.stringify({ sequence: ++sequence.current, camera: value })); lastCamera = text; }
    };
    const timer = window.setInterval(sendCamera, 34);
    channel.onopen = sendCamera;
    channel.onclose = () => window.clearInterval(timer);
    pc.ontrack = event => {
      const element = video.current;
      if (!element || generation.current !== epoch) return;
      element.srcObject = new MediaStream([event.track]);
      const displayed = () => {
        if (acceptsFrame(generation.current, epoch, revision.current, expectedRevision) && !frozenRef.current) {
          frameRef.current = null; setFrame(null); setDisplay(""); setState("viewing");
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
    ++generation.current; peer.current?.close(); peer.current = null;
    if (video.current) video.current.srcObject = null;
    if (session.current) await sessionCall("", "DELETE");
    session.current = null; frozenRef.current = false; frameRef.current = null;
    setFrame(null); setDisplay(""); setSelection(null); setState("idle");
    if (controls.current) controls.current.enabled = true;
    await refreshDocuments();
  }
  async function resume() {
    setImageReady(false); ++generation.current; peer.current?.close();
    await sessionCall("/resume"); frozenRef.current = false;
    if (controls.current) controls.current.enabled = true;
    setState("connecting"); setSelection(null); await connectMedia();
  }
  function binding() {
    if (!frame || !imageReady) throw new Error("请先固定画面，等待高清图显示完成");
    return { ticket: frame.ticket, expected_revision: frame.revision };
  }
  async function select(shape: "polygon" | "box" | "clear") {
    const result = await sessionCall<Selection>("/selection", "POST", { ...binding(), shape, polygon, minimum, maximum, depth_range: [near, far], combine, coverage });
    setSelection(result); setPreviewed(false); setConfirmed(false); setDisplay(frame!.image);
  }
  async function preview(mode: string) {
    if (!selection) return;
    const result = await sessionCall<{ image: string; revision: number }>("/preview", "POST", { ...binding(), selection_token: selection.selection_token, mode });
    if (result.revision === revision.current) { setDisplay(result.image); setImageReady(false); setImageSerial(v => v + 1); setPreviewed(true); }
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
    const orbit = new OrbitControls(camera.current, stage.current!); controls.current = orbit;
    orbit.enableDamping = false;
    const observer = new ResizeObserver(() => {
      if (frameRef.current) {
        setImageReady(false); frameRef.current = null;
        if (session.current) void sessionCall("/invalidate-frame").catch(() => {});
      }
    });
    observer.observe(stage.current!);
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
        camera.current.position.copy(center).addScaledVector(new Vector3(0.75, 1, 0.45).normalize(), radius * 3);
        orbit.target.copy(center); orbit.update(); setReady(true); await refreshDocuments();
      } catch (e) { if (life.alive) setError(String(e)); }
    })();
    const heartbeat = window.setInterval(() => {
      if (session.current) void sessionCall<Session>("", "GET").then(s => {
        if (life.alive && s.state === "error") { setState("error"); setError(s.error ?? "云会话失败"); }
      }).catch(e => { if (life.alive) setError(String(e)); });
    }, 5000);
    return () => {
      life.alive = false; ++generation.current; window.clearInterval(heartbeat); observer.disconnect(); orbit.dispose();
      peer.current?.close();
      const s = session.current; session.current = null;
      if (s) void request(`/api/gaussian-render-sessions/${s.session_id}`, "DELETE", undefined, s.token).catch(() => {});
    };
  }, [metadataUrl, cameraPathUrl, alignmentUrl, source.job_id, source.variant_id, source.asset_role]);

  useEffect(() => {
    setSelection(null); setPreviewed(false); setConfirmed(false);
  }, [tool, near, far, minimum, maximum, coverage, combine]);

  function pointer(event: ReactPointerEvent<SVGSVGElement>, phase: "start" | "move" | "end") {
    if (!frame || !imageReady || busy || tool === "box" || !stage.current) return;
    const point = imagePoint(event.clientX, event.clientY, stage.current.getBoundingClientRect(), frame.width, frame.height);
    if (phase === "start" && point) { event.currentTarget.setPointerCapture(event.pointerId); drag.current = [point]; setPolygon([]); setSelection(null); setPreviewed(false); setConfirmed(false); }
    else if (drag.current.length && point) {
      if (tool === "rectangle") setPolygon(rectangle(drag.current[0], point));
      else if (drag.current.length < 128) { drag.current.push(point); setPolygon([...drag.current]); }
    }
    if (phase === "end") { drag.current = []; event.currentTarget.releasePointerCapture(event.pointerId); }
  }
  const frozen = state === "editing_frozen", writable = !version;
  const selectable = frozen && imageReady && !busy;
  const large = !!selection && selection.selected_count > selection.visible_count / 2;
  const statusLabel: Record<string, string> = { idle: "未连接", loading: "加载模型", connecting: "连接视频", viewing: "云端观看", editing_frozen: "固定帧修剪" };
  return <section className="cloud-editor" aria-label="云端高斯修剪工作区">
    <header className="cloud-editor-heading"><div><strong>{source.label} · 云端修剪</strong><small>源模型只读 · 当前编辑未重新评价 · 不继承碰撞漫游</small></div><span role="status">{statusLabel[state] ?? state}{busy ? " · 处理中" : ""}</span></header>
    {!capability?.cloud_available && <p className="cloud-notice">{capability?.reason ?? "正在读取云端能力…"}。8082 TCP/UDP 已确认开放，实际 TURN 部署与连通仍须验证。</p>}
    {error && <p className="cloud-error" role="alert">{error}</p>}
    <div className="cloud-toolbar">
      <label>编辑文档<select value={chosen} disabled={!!session.current || busy} onChange={e => { setChosen(e.target.value); setVersion(""); setDocument(documents.find(d => d.edit_id === e.target.value) ?? null); }}><option value="">从 Original 新建副本</option>{documents.map(d => <option key={d.edit_id} value={d.edit_id}>{d.edit_id.slice(0, 8)} · r{d.revision} · {d.visible_count.toLocaleString()}</option>)}</select></label>
      <label>打开版本<select value={version} disabled={!!session.current || busy} onChange={e => setVersion(e.target.value)}><option value="">当前文档（可编辑）</option>{(documents.find(d => d.edit_id === chosen)?.versions ?? []).map(v => <option key={v.version}>{v.version}</option>)}</select></label>
      {!session.current ? <button disabled={busy || !ready || !capability?.cloud_available} onClick={() => void run(open)}>连接云端</button> : <button disabled={busy} onClick={() => void run(disconnect)}>关闭会话</button>}
      {session.current && <button disabled={busy || state === "loading"} onClick={() => void run(resume)}>重新连接视频</button>}
      <button disabled={busy || !session.current || state === "loading"} onClick={() => void run(frozen ? resume : fixedFrame)}>{frozen ? "恢复交互观看" : "固定高清画面，开始修剪"}</button>
    </div>
    <div className="cloud-workspace">
      <div ref={stage} className="cloud-stage" aria-label="云端画面；左键旋转，右键平移，滚轮缩放" tabIndex={0} onKeyDown={e => {
        if (frozenRef.current || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "-"].includes(e.key)) return;
        e.preventDefault(); const target = controls.current?.target; if (!target) return;
        const offset = camera.current.position.clone().sub(target);
        if (e.key === "+" || e.key === "-") offset.multiplyScalar(e.key === "+" ? 0.9 : 1.1);
        else offset.applyAxisAngle(e.key === "ArrowLeft" || e.key === "ArrowRight" ? camera.current.up : new Vector3(1, 0, 0).applyQuaternion(camera.current.quaternion), e.key === "ArrowLeft" || e.key === "ArrowUp" ? 0.08 : -0.08);
        camera.current.position.copy(target).add(offset); controls.current?.update();
      }}>
        <video ref={video} autoPlay playsInline muted controls={false} onClick={() => void video.current?.play()} />
        {display && <img key={imageSerial} ref={image} src={display} alt="绑定当前相机与编辑版本的高清固定帧" onLoad={event => { if (image.current === event.currentTarget && frameRef.current && frameRef.current.revision === revision.current) setImageReady(true); }} />}
        {frozen && frame && <svg viewBox={`0 0 ${frame.width} ${frame.height}`} aria-label="固定帧选区" onPointerDown={e => pointer(e, "start")} onPointerMove={e => pointer(e, "move")} onPointerUp={e => pointer(e, "end")} onPointerCancel={() => { drag.current = []; setPolygon([]); }}><polygon points={polygon.map(p => p.join(",")).join(" ")} /></svg>}
        {!display && state !== "viewing" && <div className="cloud-placeholder">{state === "idle" ? "连接后由服务器渲染；此视口不下载高斯 PLY" : "等待服务器画面…"}</div>}
        <span className="cloud-frame-label">{frozen ? `固定帧 · r${frame?.revision ?? revision.current} · ${imageReady ? "可选择" : "请重新固定或等待图像"}` : "左键旋转 / 右键平移 / 滚轮缩放 · 方向键与 +/- 同样可用"}</span>
      </div>
      <aside className="cloud-tools" aria-label="修剪工具">
        <strong>先选准，再删除</strong>
        <label>选择工具<select value={tool} onChange={e => { setTool(e.target.value as typeof tool); setPolygon([]); }}><option value="rectangle">矩形</option><option value="lasso">套索</option><option value="box">三维轴对齐盒</option></select></label>
        <label>组合<select value={combine} onChange={e => setCombine(e.target.value)}><option value="replace">替换选择</option><option value="add">增加选择</option><option value="subtract">减去选择</option></select></label>
        {tool !== "box" ? <><label>近深度<input type="number" min="0.01" step="0.1" value={near} onChange={e => setNear(Number(e.target.value))} /></label><label>远深度<input type="number" min="0.02" step="0.1" value={far} onChange={e => setFar(Number(e.target.value))} /></label><label className="cloud-check"><input type="checkbox" checked={coverage} onChange={e => setCoverage(e.target.checked)} />保守覆盖范围候选</label></> : <>{["X", "Y", "Z"].map((axis, i) => <div className="cloud-axis" key={axis}><label>{axis} 最小<input type="number" step="0.1" value={minimum[i]} onChange={e => setMinimum(v => v.map((x, j) => j === i ? Number(e.target.value) : x))} /></label><label>{axis} 最大<input type="number" step="0.1" value={maximum[i]} onChange={e => setMaximum(v => v.map((x, j) => j === i ? Number(e.target.value) : x))} /></label></div>)}</>}
        <small>{tool === "box" ? "三维盒使用原始 normalized 坐标，非摆正后的显示轴。" : "深度为相机 +Z，非米制；会选中区域内的前后层。覆盖候选可能多选，并非精确可见面拾取。"}</small>
        <button disabled={!selectable || (tool !== "box" && polygon.length < 3)} onClick={() => void run(() => select(tool === "box" ? "box" : "polygon"))}>计算选择</button>
        <button disabled={!selectable} onClick={() => void run(() => select("clear"))}>清空选择</button>
        <output>{selection ? `选中 ${selection.selected_count.toLocaleString()} / ${selection.visible_count.toLocaleString()}` : `保留 ${document?.visible_count.toLocaleString() ?? "—"} 高斯`}</output>
        <div className="cloud-preview-buttons">{[["highlight", "叠色参考"], ["isolated", "隔离选中"], ["after_delete", "删除后预览"]].map(([mode, label]) => <button key={mode} disabled={!selectable || !selection?.selected_count} onClick={() => void run(() => preview(mode))}>{label}</button>)}</div>
        <small>叠色是选中项图像合成参考，不是遮挡正确的表面高亮；暗色项请用隔离检查。</small>
        {large && <label className="cloud-check"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />确认删除超过一半的可见高斯</label>}
        <button className="cloud-delete" disabled={!selectable || !writable || !previewed || !selection?.selected_count || selection.selected_count === selection.visible_count || (large && !confirmed)} onClick={() => void run(() => operate("delete"))}>确认删除选中项</button>
        <div className="cloud-preview-buttons"><button disabled={!selectable || !writable || !document?.can_undo} onClick={() => void run(() => operate("undo"))}>撤销</button><button disabled={!selectable || !writable || !document?.can_redo} onClick={() => void run(() => operate("redo"))}>重做</button></div>
        {frozen && <button disabled={busy} onClick={() => void run(fixedFrame)}>重新固定高清画面</button>}
      </aside>
    </div>
    <footer className="cloud-versions"><button disabled={busy || !session.current || !writable || state === "loading"} onClick={() => void run(save)}>保存不可变版本</button><span role="status">{exportState.startsWith("v") ? "导出完成" : exportState}</span>{document?.versions.map(v => <span className="cloud-version" key={v.version}><span>{v.version} · {v.visible_count.toLocaleString()}</span><button disabled={busy || !session.current} onClick={() => void run(() => exportVersion(v.version))}>导出</button>{(v.exported || exportState === v.version) && <><a href={`/api/gaussian-edits/${document.edit_id}/versions/${v.version}/assets/bundle.zip`}>下载 ZIP</a><a href={`/api/gaussian-edits/${document.edit_id}/versions/${v.version}/assets/scene.ply`}>PLY</a></>}</span>)}</footer>
  </section>;
}
