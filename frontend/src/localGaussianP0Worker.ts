import { validateP0Source, type P0Source } from "./localGaussianP0Source.ts";
import { selectP0Surface, type P0SelectionRequest, type P0SelectionResult } from "./localGaussianP0Selection.ts";
import { selectLocalGaussians, type LocalSelectionRequest } from "./localGaussianSelection.ts";

export type P0WorkerMessage =
  | { type: "source"; source: P0Source; modelGeneration: number }
  | { type: "select"; request: P0SelectionRequest }
  | { type: "select-local"; request: LocalSelectionRequest }
  | { type: "cancel" };
export type P0WorkerReply =
  | { type: "ready"; sourceSha256: string; modelGeneration: number }
  | { type: "result"; result: P0SelectionResult }
  | { type: "error"; sequence: number | null; message: string };

export function createP0WorkerHandler(send: (reply: P0WorkerReply, transfer?: Transferable[]) => void) {
  let source: P0Source | null = null, modelGeneration = -1, serial = 0, lastSequence = -1;
  return async (message: P0WorkerMessage) => {
    const own = ++serial;
    let sequence: number | null = null;
    try {
      if (message.type === "cancel") return;
      if (message.type === "source") {
        source = null; modelGeneration = -1; lastSequence = -1;
        validateP0Source(message.source);
        if (!Number.isSafeInteger(message.modelGeneration) || message.modelGeneration < 0) throw new Error("P0 模型版本不合法");
        source = message.source; modelGeneration = message.modelGeneration;
        send({ type: "ready", sourceSha256: source.sha256, modelGeneration });
        return;
      }
      if (message.type !== "select" && message.type !== "select-local") throw new Error("P0 未知 Worker 请求");
      const request = message.request;
      sequence = request.sequence;
      if (!source || request.modelGeneration !== modelGeneration || request.sequence <= lastSequence) throw new Error("P0 源尚未就绪或选择请求已过期");
      lastSequence = request.sequence;
      const result = message.type === "select-local"
        ? await selectLocalGaussians(source, message.request, () => own !== serial)
        : await selectP0Surface(source, request, () => own !== serial);
      if (own === serial) send({ type: "result", result }, [result.selected.buffer]);
    } catch (error) {
      if (own === serial) send({ type: "error", sequence, message: error instanceof Error ? error.message : String(error) });
    }
  };
}
