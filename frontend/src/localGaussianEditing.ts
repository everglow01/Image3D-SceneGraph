import { p0FullMask, validateP0Mask } from "./localGaussianP0Source.ts";

export type SelectionOperation = "replace" | "add" | "subtract";
export type LocalPreview = "none" | "isolate" | "deletion" | "original";
type Snapshot = { visible: Uint8Array; protected: Uint8Array };
export const LOCAL_HISTORY_STEPS = 100;
export const LOCAL_HISTORY_BYTES = 80 * 1024 * 1024;

export function maskCount(mask: Uint8Array) {
  let count = 0;
  for (let byte of mask) while (byte) { count++; byte &= byte - 1; }
  return count;
}

export class LocalGaussianEditing {
  readonly sourceSha256: string;
  readonly count: number;
  private visible: Uint8Array;
  private selected: Uint8Array;
  private protected: Uint8Array;
  private history: Snapshot[];
  private cursor = 0;
  private preview: LocalPreview = "none";
  private generation = 0;

  constructor(sourceSha256: string, count: number) {
    if (!/^[a-f0-9]{64}$/.test(sourceSha256)) throw new Error("编辑源 SHA 不合法");
    this.sourceSha256 = sourceSha256;
    this.count = count;
    this.visible = p0FullMask(count);
    this.selected = new Uint8Array(this.visible.length);
    this.protected = new Uint8Array(this.visible.length);
    this.history = [this.snapshot()];
  }

  private snapshot(): Snapshot { return { visible: this.visible.slice(), protected: this.protected.slice() }; }
  get masks() { return { ...this.snapshot(), selected: this.selected.slice() }; }
  get revision() { return this.generation; }
  get previewMode() { return this.preview; }
  get canUndo() { return this.cursor > 0; }
  get canRedo() { return this.cursor + 1 < this.history.length; }
  get historyBytes() { return this.history.length * this.visible.byteLength * 2; }
  get counts() {
    return { visible: maskCount(this.visible), selected: maskCount(this.selected),
      protected: maskCount(this.protected), deletable: maskCount(this.deletable()) };
  }

  private deletable() {
    return this.selected.map((v, i) => v & this.visible[i] & ~this.protected[i]);
  }

  select(mask: Uint8Array, operation: SelectionOperation) {
    validateP0Mask(mask, this.count);
    if (!["replace", "add", "subtract"].includes(operation)) throw new Error("未知选集操作");
    if (operation === "subtract" && !maskCount(this.selected)) throw new Error("空选集不能减选");
    for (let i = 0; i < mask.length; i++) {
      const incoming = mask[i] & this.visible[i];
      this.selected[i] = operation === "replace" ? incoming : operation === "add"
        ? this.selected[i] | incoming : this.selected[i] & ~incoming;
    }
    this.preview = "none";
    this.generation++;
  }

  clearSelection() { this.selected.fill(0); this.preview = "none"; this.generation++; }

  private commit(visible: Uint8Array, protectedMask: Uint8Array) {
    if (visible.every((v, i) => v === this.visible[i]) && protectedMask.every((v, i) => v === this.protected[i])) return false;
    this.visible = visible; this.protected = protectedMask;
    this.history.splice(this.cursor + 1);
    this.history.push(this.snapshot());
    while (this.history.length > LOCAL_HISTORY_STEPS + 1 || this.historyBytes > LOCAL_HISTORY_BYTES) this.history.shift();
    this.cursor = this.history.length - 1;
    this.afterHistoryChange();
    return true;
  }

  protectSelected(protect: boolean) {
    const mask = this.protected.map((v, i) => protect ? v | (this.selected[i] & this.visible[i]) : v & ~this.selected[i]);
    return this.commit(this.visible.slice(), mask);
  }

  deleteSelected(confirmedLargeDeletion = false) {
    const removed = this.deletable(), n = maskCount(removed), visibleCount = maskCount(this.visible);
    if (!n) return false;
    if (n === visibleCount) throw new Error("禁止删除全部高斯");
    if (n > visibleCount / 2 && !confirmedLargeDeletion) throw new Error("删除超过当前可见高斯的 50%，需要确认");
    return this.commit(this.visible.map((v, i) => v & ~removed[i]), this.protected.slice());
  }

  private afterHistoryChange() {
    for (let i = 0; i < this.selected.length; i++) this.selected[i] &= this.visible[i];
    this.preview = "none";
    this.generation++;
  }

  undo() { return this.moveHistory(-1); }
  redo() { return this.moveHistory(1); }
  private moveHistory(delta: number) {
    const next = this.cursor + delta;
    if (next < 0 || next >= this.history.length) return false;
    this.cursor = next;
    const entry = this.history[this.cursor];
    this.visible = entry.visible.slice(); this.protected = entry.protected.slice();
    this.afterHistoryChange();
    return true;
  }

  setPreview(mode: LocalPreview) {
    if (!["none", "isolate", "deletion", "original"].includes(mode)) throw new Error("未知预览模式");
    if ((mode === "isolate" || mode === "deletion") && !maskCount(this.selected)) throw new Error("请先选择高斯");
    this.preview = mode; this.generation++;
  }

  get displayMask() {
    if (this.preview === "original") return p0FullMask(this.count);
    if (this.preview === "isolate") return this.visible.map((v, i) => v & this.selected[i]);
    if (this.preview === "deletion") {
      const removed = this.deletable();
      return this.visible.map((v, i) => v & ~removed[i]);
    }
    return this.visible.slice();
  }
}

export function localEditorShortcut(event: {
  key: string; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean; altKey: boolean; isComposing?: boolean;
  target: EventTarget | null;
}): "delete" | "undo" | "redo" | "cancel" | null {
  const target = event.target as HTMLElement | null;
  if (event.isComposing || event.altKey || target?.isContentEditable ||
      target?.closest?.("input, textarea, select, [contenteditable]:not([contenteditable='false']), [role='textbox']")) return null;
  const key = event.key.toLowerCase(), command = event.ctrlKey || event.metaKey;
  if (command && key === "z") return event.shiftKey ? "redo" : "undo";
  if (command && key === "y") return "redo";
  if (!command && key === "delete") return "delete";
  if (!command && key === "escape") return "cancel";
  return null;
}
