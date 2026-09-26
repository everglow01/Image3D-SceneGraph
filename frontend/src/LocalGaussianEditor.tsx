import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import type { LocalEditorHandle } from "./gaussianViewerLeave";
import { createPortal } from "react-dom";
import type { SparkLocalEdit, SparkPageViewer } from "./SparkPageViewer.ts";
import { LocalGaussianInteraction, type LocalTool } from "./localGaussianInteraction.ts";
import { LocalSelectionClient } from "./localGaussianSelectionClient.ts";
import type { SelectionOperation } from "./localGaussianEditing.ts";
import type { LocalSelectionMode } from "./localGaussianSelection.ts";
import { LocalGaussianPersistence } from "./localGaussianPersistence.ts";
import { LocalGaussianSavePanel } from "./LocalGaussianSavePanel.tsx";
import "./localGaussianEditor.css";

export function LocalGaussianEditor({ viewer, sourceSha256, documentBinding, handleRef }: {
  viewer: SparkPageViewer; sourceSha256: string; documentBinding?: { editId: string; metadataSha256: string; gaussianCount?: number };
  handleRef?: RefObject<LocalEditorHandle | null>;
}) {
  const [editor, setEditor] = useState<LocalGaussianInteraction | null>(null);
  const [persistence, setPersistence] = useState<LocalGaussianPersistence | null>(null);
  const editId = documentBinding?.editId, metadataSha256 = documentBinding?.metadataSha256;
  const [error, setError] = useState("");
  const [, redraw] = useState(0);
  const lifecycle = useRef<Promise<void>>(Promise.resolve());
  const overlay = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const abort = new AbortController();
    let handle: SparkLocalEdit | undefined, client: LocalSelectionClient | undefined, session: LocalGaussianInteraction | undefined;
    let storage: LocalGaussianPersistence | undefined;
    const notify = () => { if (!abort.signal.aborted) redraw(n => n + 1); };
    const release = async () => {
      const oldStorage = storage, oldSession = session, oldHandle = handle, oldClient = client;
      storage = undefined; session = undefined; handle = undefined; client = undefined;
      try { await oldStorage?.dispose(); }
      finally {
        if (oldSession) await oldSession.dispose();
        else { oldClient?.dispose(); if (oldHandle) await viewer.endLocalEdit(oldHandle); }
      }
    };
    setEditor(null); setPersistence(null); setError("");
    const initialized = lifecycle.current.then(async () => {
      abort.signal.throwIfAborted();
      handle = await viewer.beginLocalEdit(sourceSha256, abort.signal);
      abort.signal.throwIfAborted();
      if (documentBinding?.gaussianCount !== undefined && handle.source.count !== documentBinding.gaussianCount) {
        throw new Error("已加载模型数量与编辑文档不符，未开启编辑");
      }
      client = new LocalSelectionClient(new Worker(new URL("./gaussianSelection.worker.ts", import.meta.url), { type: "module" }), sourceSha256, handle.source);
      await client.ready; abort.signal.throwIfAborted();
      session = new LocalGaussianInteraction(viewer, handle, client, notify);
      session.inputEnabled = !editId;
      if (editId !== undefined) storage = new LocalGaussianPersistence(session.state, editId, metadataSha256 ?? "", notify);
      setEditor(session); setPersistence(storage ?? null);
      if (storage) {
        try { await storage.open(); }
        catch (e) { if (!abort.signal.aborted) session.report(e); }
      }
      abort.signal.throwIfAborted();
      session.inputEnabled = true;
      await session.refresh();
    }).catch(async e => {
      if (!abort.signal.aborted) setError(e instanceof Error ? e.message : String(e));
      await release();
    });
    lifecycle.current = initialized.catch(() => {});
    let closing: Promise<void> | null = null;
    const dispose = () => {
      if (closing) return closing;
      abort.abort(); client?.dispose();
      if (session) { session.inputEnabled = false; session.cancel(); }
      closing = initialized.catch(() => {}).then(release);
      lifecycle.current = closing.catch(() => {});
      return closing;
    };
    const control: LocalEditorHandle = { dispose, async leave() {
      if (storage?.busy) {
        window.alert("文档操作尚未完成，请等待后再离开；当前仍可下载草稿。"); return false;
      }
      const unsaved = storage ? storage.hasUnsavedWork : !!session?.state.contentRevision;
      if (unsaved && !window.confirm("有未保存修改、待确认保存或仅保存在本地的保护状态。请取消并保存版本／下载草稿；确定将放弃本地状态并离开。")) return false;
      await dispose(); return true;
    } };
    if (handleRef) handleRef.current = control;
    const warn = (event: BeforeUnloadEvent) => {
      if (!storage && session?.state.contentRevision) { event.preventDefault(); event.returnValue = ""; }
    };
    window.addEventListener("beforeunload", warn);
    return () => {
      window.removeEventListener("beforeunload", warn);
      // The owner may run its cleanup after ours; retain the idempotent close handle until replaced.
      void dispose().catch(e => { console.error("本地编辑资源释放失败", e); });
    };
  }, [viewer, sourceSha256, editId, metadataSha256, documentBinding?.gaussianCount, handleRef]);

  useEffect(() => {
    let frame = 0;
    const position = () => {
      const node = overlay.current, canvas = viewer.renderer.domElement;
      if (node) {
        const rect = canvas.getBoundingClientRect();
        node.style.left = `${rect.left}px`; node.style.top = `${rect.top}px`;
        node.style.width = `${rect.width}px`; node.style.height = `${rect.height}px`;
        node.setAttribute("viewBox", `0 0 ${canvas.width} ${canvas.height}`);
      }
      frame = requestAnimationFrame(position);
    };
    frame = requestAnimationFrame(position);
    return () => cancelAnimationFrame(frame);
  }, [viewer]);

  const revision = editor?.state.revision;
  const counts = useMemo(() => editor?.state.counts, [editor, revision]);
  const run = (action: () => Promise<unknown>) => { void action().catch(e => editor?.report(e)); };
  const toolNames: [LocalTool, string][] = [["navigate", "导航"], ["rectangle", "矩形"], ["lasso", "套索"], ["pick", "拾取薄层深度"], ["box", "三维盒"]];
  return <section className="local-gaussian-editor" aria-label="本地编辑" onKeyDown={e => editor?.keyDown(e.nativeEvent)}>
    <strong>本地编辑 · 实验</strong>
    <p>{documentBinding ? "读回初始基线后，本地交互不等待服务器；保存版本与导出分开。" : "未绑定编辑文档，仅内存编辑，卸载组件会丢失修改。"}表层选择不是物体分割。</p>
    {!editor && !error && <p role="status">正在准备源行几何与选择 Worker…</p>}
    {(error || editor?.error) && <p role="alert">{error || editor?.error}</p>}
    <fieldset disabled={!editor?.editable}>
      <legend>选择与导航</legend>
      <div className="local-editor-actions">{toolNames.map(([value, label]) => <button key={value} type="button"
        aria-pressed={editor?.tool === value} onClick={() => editor && run(() => editor.changeTool(value))}>{label}</button>)}</div>
      <label>选择范围 <select value={editor?.mode ?? "surface"} disabled={editor?.tool === "box"}
        onChange={e => editor && run(() => editor.setRange(e.target.value as Exclude<LocalSelectionMode, "box">))}>
        <option value="surface">表层贡献（默认）</option><option value="depth">指定深度薄层（中心）</option><option value="through">穿透中心（会选中后方）</option>
      </select></label>
      <label>选集操作 <select value={editor?.operation ?? "replace"} onChange={e => {
        const operation = e.target.value as SelectionOperation;
        if (editor) run(() => editor.act(() => { editor.operation = operation; }));
      }}><option value="replace">新建</option><option value="add">追加</option><option value="subtract" disabled={!counts?.selected}>减选</option></select></label>
      {editor?.mode === "through" && <p>穿透不检查遮挡，可能包含背景；请多角度检查。</p>}
      {editor?.mode === "depth" && <div className="local-editor-depth">
        <p>深度为源归一化任意单位，不是米。可信锚点：{editor.anchor?.toPrecision(5) ?? "未拾取／相机已变化"}</p>
        {([0, 1] as const).map(i => <label key={i}>{i === 0 ? "近端" : "远端"}<input type="number" min="0.01" max="1000000" step="0.01"
          value={Number.isFinite(editor.depthRange[i]) ? editor.depthRange[i] : ""} onChange={e => {
            const value = e.target.valueAsNumber;
            run(() => editor.act(() => { editor.depthRange[i] = value; editor.anchor = null; }));
          }} /></label>)}
      </div>}
      {editor?.tool === "box" && <div className="local-editor-actions">
        <button type="button" onClick={() => editor.setBoxMode("translate")}>移动盒</button>
        <button type="button" onClick={() => editor.setBoxMode("scale")}>调整盒大小</button>
        <button type="button" onClick={() => run(() => editor.selectBox())}>选择盒内中心</button>
        <p>盒轴跟随源 normalized 坐标，显示摆正不会改变盒选语义。</p>
      </div>}
      <p>左拖选择；Alt＋左拖或“导航”工具转动视角，右键平移。拖选时暂锁导航，松开即释放。</p>
    </fieldset>
    {editor && counts && <>
      <p role="status">{editor.viewingHistory ? "当前编辑状态（非正在查看的历史版本）：" : ""}可见 {counts.visible.toLocaleString()} · 选中 {counts.selected.toLocaleString()} · 受保护 {counts.protected.toLocaleString()} · 可删除 {counts.deletable.toLocaleString()}
        {editor.selecting ? " · 正在选择…" : editor.displayBusy ? " · 正在更新画面…" : ""}</p>
      <fieldset disabled={!editor.editable}><legend>非破坏操作</legend>
        <div className="local-editor-actions">
          <button type="button" disabled={!counts.deletable || editor.state.previewMode === "original"} onClick={() => run(() => editor.deleteSelection())}>隐藏选中项（Delete）</button>
          <button type="button" disabled={!editor.state.canUndo} onClick={() => run(() => editor.act(() => editor.state.undo()))}>撤销</button>
          <button type="button" disabled={!editor.state.canRedo} onClick={() => run(() => editor.act(() => editor.state.redo()))}>重做</button>
        </div>
        <details><summary>保护、隔离与前后对照</summary>
        <div className="local-editor-actions">
          <button type="button" disabled={!counts.selected} onClick={() => run(() => editor.act(() => editor.state.protectSelected(true)))}>保护选中项</button>
          <button type="button" disabled={!counts.selected} onClick={() => run(() => editor.act(() => editor.state.protectSelected(false)))}>解除保护</button>
          <button type="button" onClick={() => run(() => editor.act(() => editor.state.clearSelection()))}>清空选集</button>
        </div>
        <label><input type="checkbox" checked={editor.highlight} onChange={e => {
          const checked = e.target.checked; run(() => editor.act(() => { editor.highlight = checked; }));
        }} />显示高亮（选中橙色／保护蓝色；选中优先）</label>
        <div className="local-editor-actions">
          <button type="button" disabled={!counts.selected} aria-pressed={editor.state.previewMode === "isolate"} onClick={() => run(() => editor.preview("isolate"))}>隔离选集</button>
          <button type="button" disabled={!counts.selected} aria-pressed={editor.state.previewMode === "deletion"} onClick={() => run(() => editor.preview("deletion"))}>删除后预览</button>
          <button type="button" aria-pressed={editor.state.previewMode === "original"} onClick={() => run(() => editor.preview("original"))}>原始对照</button>
          <button type="button" aria-pressed={editor.state.previewMode === "none"} onClick={() => run(() => editor.preview("none"))}>返回编辑画面</button>
        </div>
        <p>预览不修改可见状态；Ctrl/Cmd＋Z 撤销，Shift＋Ctrl/Cmd＋Z 重做，Esc 取消／清空。历史最多100步，不含选集与临时预览。</p>
        </details>
      </fieldset>
    </>}
    {editor && persistence && <LocalGaussianSavePanel key={`${editId}:${sourceSha256}`} editor={editor} persistence={persistence} viewer={viewer} />}
    {createPortal(<svg ref={overlay} className="local-editor-selection" aria-hidden="true">
      {editor && editor.polygon.length > 1 && <polygon points={editor.polygon.map(p => p.join(",")).join(" ")} />}
    </svg>, document.body)}
  </section>;
}
