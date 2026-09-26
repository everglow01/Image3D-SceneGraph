import { useEffect, useRef, useState, type RefObject } from "react";
import type { CloudSource } from "./CloudGaussianViewer";
import type { SparkPageViewer } from "./SparkPageViewer";
import type { LocalEditorHandle } from "./gaussianViewerLeave";
import { LocalGaussianEditor } from "./LocalGaussianEditor";
import { localDocumentRequest, matchesLocalDocument, validateLocalDocument, type LoadedLocalSource, type LocalEditDocument } from "./localGaussianDocument";
import "./localGaussianEditor.css";

export function LocalGaussianDocumentPanel({ viewer, source, loaded, sourceUrl, metadataUrl, handleRef }: {
  viewer: SparkPageViewer; source: CloudSource; loaded: LoadedLocalSource; sourceUrl: string; metadataUrl: string;
  handleRef: RefObject<LocalEditorHandle | null>;
}) {
  const [documents, setDocuments] = useState<LocalEditDocument[]>([]), [chosen, setChosen] = useState("");
  const [opened, setOpened] = useState<LocalEditDocument | null>(null), [error, setError] = useState("");
  const [busy, setBusy] = useState(false), [uncertain, setUncertain] = useState(false);
  const child = useRef<LocalEditorHandle | null>(null), working = useRef(false);
  const life = useRef<AbortController | null>(null);
  const key = JSON.stringify([source.job_id, source.variant_id, source.asset_role, sourceUrl, metadataUrl]);
  async function list(signal: AbortSignal) {
    const result = await localDocumentRequest<{ edits: LocalEditDocument[] }>(`/api/gaussian-edits?job_id=${encodeURIComponent(source.job_id)}`, signal);
    if (!Array.isArray(result.edits)) throw new Error("编辑文档列表无效");
    if (!signal.aborted) setDocuments(result.edits.filter(d => matchesLocalDocument(d, source)));
  }
  useEffect(() => {
    const abort = new AbortController(); life.current = abort;
    let closing: Promise<void> | null = null;
    const control: LocalEditorHandle = {
      async leave() {
        if (closing) { await closing; return true; }
        if (working.current) { window.alert("正在读取或创建编辑文档，请等待后再切换。"); return false; }
        if (child.current && !await child.current.leave()) return false;
        await control.dispose(); return true;
      },
      dispose() {
        abort.abort();
        return closing ??= child.current?.dispose() ?? Promise.resolve();
      }
    };
    handleRef.current = control;
    void list(abort.signal).catch(e => { if (!abort.signal.aborted) setError(e instanceof Error ? e.message : String(e)); });
    return () => {
      // Retain the settled/closing handle so parent cleanup can await it regardless of effect order.
      void control.dispose().catch(e => console.error("编辑文档关闭失败", e));
    };
  }, [key, handleRef]);
  async function run(action: (signal: AbortSignal) => Promise<void>) {
    const signal = life.current?.signal;
    if (!signal || signal.aborted || working.current) return;
    working.current = true; setBusy(true); setError("");
    try { await action(signal); }
    catch (e) { if (!signal.aborted) setError(e instanceof Error ? e.message : String(e)); }
    finally { working.current = false; if (!signal.aborted) setBusy(false); }
  }
  async function open(signal: AbortSignal) {
    // Compare the small metadata with the exact response used for this viewer, not a newly invented source identity.
    const response = await fetch(metadataUrl, { signal: AbortSignal.any([signal, AbortSignal.timeout(60_000)]), cache: "no-store" });
    if (!response.ok || await response.text() !== loaded.metadataText) throw new Error("模型元数据已变化或无法复核，请重新加载模型");
    let doc: LocalEditDocument;
    if (chosen) doc = await localDocumentRequest(`/api/gaussian-edits/${chosen}`, signal);
    else {
      if (uncertain) throw new Error("上次创建结果未知，请刷新列表并选择已有文档，勿重复创建");
      setUncertain(true);
      doc = await localDocumentRequest("/api/gaussian-edits", signal, source);
    }
    validateLocalDocument(doc, source, loaded, sourceUrl, metadataUrl);
    if (!signal.aborted) { setChosen(doc.edit_id); setOpened(doc); }
  }
  if (opened) return <LocalGaussianEditor viewer={viewer} sourceSha256={opened.source.ply_sha256} handleRef={child}
    documentBinding={{ editId: opened.edit_id, metadataSha256: opened.source.metadata_sha256, gaussianCount: opened.source.gaussian_count }} />;
  return <section className="local-gaussian-editor" aria-label="本地编辑文档">
    <strong>本地编辑 · 实验</strong>
    <p>原始 SH3 · 本机渲染与选择 · CPU 保存／导出，不连接云视频。请选择当前文档，或从原模型新建非破坏副本。</p>
    {error && <p role="alert">{error}</p>}
    <label>编辑文档 <select disabled={busy} value={chosen} onChange={e => setChosen(e.target.value)}>
      <option value="">从原模型新建副本</option>
      {documents.map(d => <option key={d.edit_id} value={d.edit_id}>{d.edit_id.slice(0, 8)} · r{d.revision} · {d.visible_count.toLocaleString()} 高斯</option>)}
    </select></label>
    <div className="local-editor-actions">
      <button type="button" disabled={busy || (!chosen && uncertain)} onClick={() => void run(open)}>{busy ? "正在准备文档…" : "打开本地编辑"}</button>
      <button type="button" disabled={busy} onClick={() => void run(list)}>刷新文档列表</button>
    </div>
    {uncertain && !chosen && <p>创建请求已发送，若响应丢失请刷新列表核对并选择已创建文档；不会自动重复创建。</p>}
    <p>原模型不会改写。编辑期间禁用漫游；退出后恢复原模型浏览，不把修剪结果当作已有碰撞导航。</p>
  </section>;
}
