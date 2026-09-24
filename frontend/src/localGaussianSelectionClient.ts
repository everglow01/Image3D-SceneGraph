import type { P0SelectionResult } from "./localGaussianP0Selection.ts";
import type { P0Source } from "./localGaussianP0Source.ts";
import type { P0WorkerMessage, P0WorkerReply } from "./localGaussianP0Worker.ts";
import type { LocalSelectionRequest } from "./localGaussianSelection.ts";

export class LocalSelectionClient {
  readonly ready: Promise<void>;
  private sequence = 0;
  private disposed = false;
  private initialized = false;
  private rejectReady!: (error: Error) => void;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private pending: { request: LocalSelectionRequest; resolve: (r: P0SelectionResult | null) => void; reject: (e: Error) => void } | null = null;

  private readonly worker: Worker;
  private readonly sha256: string;

  constructor(worker: Worker, sha256: string, source: P0Source) {
    this.worker = worker; this.sha256 = sha256;
    this.ready = new Promise<void>((resolve, reject) => {
      this.rejectReady = reject;
      this.timer = setTimeout(() => this.fail(new Error("选择 Worker 准备超时")), 6000);
      worker.onmessage = (event: MessageEvent<P0WorkerReply>) => {
        if (this.disposed) return;
        const reply = event.data;
        if (reply.type === "ready") {
          if (reply.sourceSha256 !== this.sha256 || reply.modelGeneration !== 1) return this.fail(new Error("Worker 源身份不符"));
          this.clearTimer(); this.initialized = true; resolve();
        } else if (reply.type === "error") {
          if (reply.sequence === null || reply.sequence === this.pending?.request.sequence) this.fail(new Error(reply.message));
        } else if (reply.type === "result") {
          const pending = this.pending, r = reply.result;
          if (!pending || r.sequence !== pending.request.sequence) return;
          if (r.sourceSha256 !== this.sha256 || r.modelGeneration !== 1 || r.cameraGeneration !== pending.request.cameraGeneration) {
            this.fail(new Error("Worker 返回过期源或相机")); return;
          }
          this.pending = null; this.clearTimer(); pending.resolve(r);
        }
      };
      worker.onerror = e => { this.initialized = false; this.fail(new Error(e.message || "选择 Worker 错误")); };
      worker.onmessageerror = () => { this.initialized = false; this.fail(new Error("选择 Worker 消息无法解码")); };
      try { this.post({ type: "source", source, modelGeneration: 1 }, [source.geometry.buffer as ArrayBuffer]); }
      catch (error) { this.fail(error as Error); }
    });
  }

  private post(message: P0WorkerMessage, transfer: Transferable[] = []) { this.worker.postMessage(message, transfer); }
  private clearTimer() { if (this.timer !== null) clearTimeout(this.timer); this.timer = null; }
  private fail(error: Error) {
    this.clearTimer();
    if (!this.initialized) this.rejectReady(error);
    const pending = this.pending; this.pending = null; pending?.reject(error);
  }

  select(request: Omit<LocalSelectionRequest, "sourceSha256" | "modelGeneration" | "sequence">) {
    if (this.disposed || !this.initialized) return Promise.reject(new Error("选择 Worker 未就绪或已释放"));
    this.cancel();
    const full = { ...request, sourceSha256: this.sha256, modelGeneration: 1, sequence: ++this.sequence };
    return new Promise<P0SelectionResult | null>((resolve, reject) => {
      this.pending = { request: full, resolve, reject };
      this.timer = setTimeout(() => { this.fail(new Error("选择消息超时，未应用任何结果")); this.post({ type: "cancel" }); }, 3000);
      try { this.post({ type: "select-local", request: full }); } catch (error) { this.fail(error as Error); }
    });
  }

  cancel() {
    if (!this.initialized) return;
    const pending = this.pending; this.pending = null; this.clearTimer();
    pending?.resolve(null);
    if (!this.disposed) this.post({ type: "cancel" });
  }

  dispose() {
    if (this.disposed) return;
    this.cancel(); this.disposed = true;
    this.fail(new Error("选择 Worker 已关闭"));
    this.worker.onmessage = this.worker.onerror = this.worker.onmessageerror = null;
    this.worker.terminate();
  }
}
