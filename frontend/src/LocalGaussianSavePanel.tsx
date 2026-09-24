import { useEffect, useRef, useState } from "react";
import type { SparkPageViewer } from "./SparkPageViewer.ts";
import type { LocalGaussianInteraction } from "./localGaussianInteraction.ts";
import type { LocalGaussianPersistence } from "./localGaussianPersistence.ts";
import { captureView, restoreView } from "./cloudGaussianEditor.ts";
import { LOCAL_DRAFT_LIMIT } from "./localGaussianDraft.ts";

export function LocalGaussianSavePanel({ editor, persistence: sync, viewer }: {
  editor: LocalGaussianInteraction; persistence: LocalGaussianPersistence; viewer: SparkPageViewer;
}) {
  const [error, setError] = useState(""), [notice, setNotice] = useState("");
  const [version, setVersion] = useState(""), [historical, setHistorical] = useState("");
  const [busy, setBusy] = useState(false), [, redraw] = useState(0);
  const working = useRef(false), alive = useRef(false), pollingFailed = useRef(false);
  useEffect(() => {
    alive.current = true;
    const warn = (event: BeforeUnloadEvent) => {
      if (sync.hasUnsavedWork) { event.preventDefault(); event.returnValue = ""; }
    };
    window.addEventListener("beforeunload", warn);
    const timer = window.setInterval(() => {
      if (!sync.busy && !working.current && sync.exportState === "running" && !pollingFailed.current) {
        void sync.pollExport().catch(e => {
          if (alive.current) { pollingFailed.current = true; setError(e instanceof Error ? e.message : String(e)); }
        }).finally(() => { if (alive.current) redraw(n => n + 1); });
      }
    }, 1500);
    return () => { alive.current = false; window.removeEventListener("beforeunload", warn); window.clearInterval(timer); };
  }, [sync]);

  async function run(action: () => Promise<void>) {
    if (working.current || sync.busy) return;
    working.current = true; setBusy(true); setError(""); setNotice("");
    try { await action(); }
    catch (e) { if (alive.current) setError(e instanceof Error ? e.message : String(e)); }
    finally { working.current = false; if (alive.current) setBusy(false); }
  }
  function downloadDraft() {
    try {
      const camera = viewer.controls ? captureView(editor.state.sourceSha256, viewer.camera, viewer.controls.target) : null;
      const text = sync.draft(camera);
      const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      const link = document.createElement("a"); link.href = url; link.download = `${sync.editId}-local-draft.json`;
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setNotice("已请求下载草稿，请确认浏览器已保存文件。草稿不是 PLY，也不包含模型或完整撤销历史。");
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }
  async function importDraft(file: File) {
    if (file.size > LOCAL_DRAFT_LIMIT) throw new Error("草稿超过 2 MiB 上限");
    const generation = editor.state.contentRevision, text = await file.text();
    if (!alive.current) return;
    if (generation !== editor.state.contentRevision) throw new Error("读取文件期间本地状态已变化，未导入");
    if (!window.confirm("导入将替换当前本地可见／保护状态并清空临时撤销历史；尚未保存的内容请先下载草稿。继续？")) return;
    editor.cancel();
    const result = await sync.restoreDraft(text);
    if (!alive.current) return;
    if (result.camera && viewer.controls) {
      viewer.controls.target.copy(restoreView(viewer.camera, result.camera)); viewer.controls.update();
    }
    await editor.refresh();
    if (alive.current) setNotice(result.conflict ? "草稿已恢复供本地查看／编辑，但服务器基线不同，已停止同步。" :
      result.offline ? "草稿已离线恢复；保存前必须重新核对服务器基线。" : "草稿已恢复；尚未提交服务器。保护只属于草稿，不写入修剪版本。");
  }
  const selected = version || sync.lastVersion?.version || sync.versions.at(-1)?.version || "";
  const locked = busy || sync.busy, mutable = editor.editable;
  return <fieldset aria-label="本地保存与恢复"><legend>保存、导出与恢复</legend>
    <p role="status">{sync.conflict ? "版本冲突 · 同步已停止" : sync.needsRetry ? "保存待确认 · 重试将提交原快照" : sync.dirty ? "当前可见修改未保存" : "当前可见状态已持久化"}
      {sync.busy ? " · 正在处理文档…" : ""} · 服务器 revision：{sync.baseRevision ?? "未知"}</p>
    <p>文档：{sync.editId}</p>
    {error && <p role="alert">{error}</p>}{notice && <p role="status">{notice}</p>}
    <div className="local-editor-actions">
      <button type="button" disabled={locked || !mutable || sync.conflict || sync.needsBaselineCheck || sync.baseRevision === null}
        onClick={() => void run(async () => {
          const saved = await sync.save(() => window.confirm("此快照相对服务器将隐藏超过 50% 的可见高斯，确认保存？"));
          if (alive.current && saved) { setVersion(saved.version); setNotice(`已保存 ${saved.version}；${sync.dirty ? "较新的本地修改仍未保存。" : "本地撤销／重做保留。"}`); }
        })}>{sync.needsRetry ? "重试原保存" : "保存版本"}</button>
      <button type="button" onClick={downloadDraft}>下载当前编辑草稿</button>
      <label>导入草稿 <input type="file" accept=".json,application/json" disabled={locked || !mutable || sync.needsRetry}
        onChange={event => {
          const file = event.target.files?.[0]; event.target.value = "";
          if (file) void run(() => importDraft(file));
        }} /></label>
      {sync.needsBaselineCheck && !sync.conflict && <button type="button" disabled={locked}
        onClick={() => void run(async () => { await sync.verifyDraftBase(); if (alive.current) setNotice("服务器基线一致，可以保存草稿修改。"); })}>重新核对草稿基线</button>}
      {sync.baseRevision === null && editor.state.contentRevision === 0 && <button type="button" disabled={locked}
        onClick={() => void run(async () => { await sync.open(); await editor.refresh(); })}>重试读取文档</button>}
    </div>
    <p>保存不包含保护锁；保护需随草稿下载。断网可继续编辑和撤销。没有自动草稿备份，崩溃前未保存／未下载的内容可能丢失；离页提醒仅尽力保护。</p>
    <label>已保存版本 <select value={selected} disabled={locked} onChange={event => setVersion(event.target.value)}>
      {!sync.versions.length && <option value="">尚无版本</option>}
      {sync.versions.map(v => <option key={v.version} value={v.version}>{v.version} · {v.visible_count.toLocaleString()} 高斯</option>)}
    </select></label>
    <div className="local-editor-actions">
      <button type="button" disabled={locked || !selected || !editor.ready} onClick={() => void run(async () => {
        const mask = await sync.readVersion(selected);
        if (!alive.current) return;
        await editor.showHistoricalMask(mask); if (alive.current) setHistorical(selected);
      })}>只读查看此版本</button>
      <button type="button" disabled={locked || !historical} onClick={() => void run(async () => {
        await editor.showHistoricalMask(null); if (alive.current) setHistorical("");
      })}>返回当前编辑</button>
      <button type="button" disabled={locked || !selected || sync.exportState === "running"} onClick={() => void run(async () => {
        await sync.startExport(selected); pollingFailed.current = false;
      })}>导出此版本 PLY／ZIP</button>
      <button type="button" disabled={locked || !sync.exportVersion} onClick={() => void run(async () => {
        await sync.pollExport(); pollingFailed.current = false;
      })}>检查导出状态</button>
      {sync.versions.find(v => v.version === selected)?.exported && <>
        <a href={sync.downloadUrl(selected, "scene.ply")}>下载 PLY</a><a href={sync.downloadUrl(selected, "bundle.zip")}>下载 ZIP</a>
      </>}
    </div>
    {historical && <p role="status">正在只读查看 {historical}，选择／删除／保存已禁用；当前本地编辑和历史保持不变。返回当前编辑后再继续修改。</p>}
    {sync.exportVersion && <p role="status">{sync.exportVersion} 导出：{({ running: "进行中", done: "完成", error: "失败", not_exported: "尚未完成，可重新启动" } as Record<string, string>)[sync.exportState] ?? "未知"}</p>}
    <p>导出只处理所选已保存版本，不会自动保存新修改；保留原始属性与行序，不继承重建质量评分或导航有效性。</p>
  </fieldset>;
}
