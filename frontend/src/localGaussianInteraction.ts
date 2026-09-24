import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { rectangle, type Pixel } from "./cloudGaussianEditor.ts";
import { LocalGaussianEditing, localEditorShortcut, type LocalPreview, type SelectionOperation } from "./localGaussianEditing.ts";
import { LocalSelectionClient } from "./localGaussianSelectionClient.ts";
import type { LocalSelectionMode } from "./localGaussianSelection.ts";
import type { SparkLocalEdit, SparkPageViewer } from "./SparkPageViewer.ts";
import { validateP0Mask } from "./localGaussianP0Source.ts";

export type LocalTool = "navigate" | "rectangle" | "lasso" | "pick" | "box";

export class LocalGaussianInteraction {
  readonly state: LocalGaussianEditing;
  tool: LocalTool = "rectangle";
  mode: LocalSelectionMode = "surface";
  operation: SelectionOperation = "replace";
  depthRange: [number, number] = [0.01, 1];
  anchor: number | null = null;
  highlight = true;
  selecting = false;
  error = "";
  polygon: Pixel[] = [];
  inputEnabled = true;
  private historicalMask: Uint8Array | null = null;
  private disposed = false;
  private failed = false;
  private serial = 0;
  private cameraGeneration = 0;
  private signature: string;
  private frame = 0;
  private dirty = false;
  private rendering: Promise<void> | null = null;
  private drag: { id: number; start: Pixel; points: Pixel[]; controlsEnabled: boolean } | null = null;
  private readonly canvas: HTMLCanvasElement;
  private readonly oldTabIndex: number;
  private readonly box = new THREE.Object3D();
  private boxLines: THREE.LineSegments | null = null;
  private gizmo: TransformControls | null = null;
  private gizmoControlsEnabled: boolean | null = null;

  private readonly viewer: SparkPageViewer;
  private readonly handle: SparkLocalEdit;
  private readonly client: LocalSelectionClient;
  private readonly notify: () => void;

  constructor(viewer: SparkPageViewer, handle: SparkLocalEdit, client: LocalSelectionClient, notify: () => void) {
    this.viewer = viewer; this.handle = handle; this.client = client; this.notify = notify;
    this.state = new LocalGaussianEditing(handle.source.sha256, handle.source.count);
    this.signature = viewer.localEditView(handle).signature;
    this.canvas = viewer.renderer.domElement;
    this.oldTabIndex = this.canvas.tabIndex; this.canvas.tabIndex = 0;
    this.canvas.addEventListener("pointerdown", this.pointerDown, true);
    this.canvas.addEventListener("pointermove", this.pointerMove, true);
    this.canvas.addEventListener("pointerup", this.pointerUp, true);
    this.canvas.addEventListener("pointercancel", this.pointerCancel, true);
    this.canvas.addEventListener("lostpointercapture", this.pointerCancel, true);
    this.canvas.addEventListener("keydown", this.keyDown);
    this.frame = requestAnimationFrame(this.watchView);
  }

  get ready() { return !this.disposed && !this.failed; }
  get editable() { return this.ready && this.inputEnabled && this.historicalMask === null; }
  get viewingHistory() { return this.historicalMask !== null; }

  async showHistoricalMask(mask: Uint8Array | null) {
    if (!this.ready) throw new Error("本地编辑已关闭或显示失败");
    if (mask) validateP0Mask(mask, this.state.count);
    this.cancel(); this.endDrag(); this.restoreGizmoNavigation();
    this.gizmo?.detach(); this.box.visible = false; this.polygon = [];
    this.historicalMask = mask?.slice() ?? null;
    if (!mask && this.tool === "box") this.showBox();
    this.notify(); await this.refresh();
  }
  get displayBusy() { return this.rendering !== null; }

  private watchView = () => {
    if (this.disposed) return;
    try {
      const signature = this.viewer.localEditView(this.handle).signature;
      if (signature !== this.signature) {
        this.signature = signature; this.cameraGeneration++;
        if (this.anchor !== null) this.depthRange = [NaN, NaN];
        this.anchor = null;
        this.cancel(); this.endDrag(); this.notify();
      }
    } catch (error) { this.failed = true; this.cancel(); this.endDrag(); this.report(error); return; }
    this.frame = requestAnimationFrame(this.watchView);
  };

  report(error: unknown) { if (!this.disposed) { this.error = error instanceof Error ? error.message : String(error); this.notify(); } }

  cancel() { this.serial++; this.client.cancel(); this.selecting = false; }

  async refresh() {
    if (!this.ready) return;
    this.dirty = true;
    if (this.rendering) return this.rendering;
    this.rendering = (async () => {
      while (this.dirty && !this.disposed) {
        this.dirty = false;
        const masks = this.state.masks;
        const empty = this.historicalMask ? new Uint8Array(masks.visible.length) : null;
        await this.viewer.updateLocalEdit(this.handle, this.historicalMask ?? this.state.displayMask, empty ?? masks.selected, empty ?? masks.protected,
          !this.historicalMask && this.highlight && this.state.previewMode === "none");
      }
    })().catch(error => { this.failed = true; this.report(error); throw error; }).finally(() => {
      this.rendering = null; if (!this.disposed) this.notify();
    });
    this.notify();
    return this.rendering;
  }

  async act(action: () => unknown) {
    if (!this.editable) throw new Error("当前为只读状态，或编辑已关闭");
    this.cancel(); this.endDrag(); this.error = "";
    action(); this.notify(); await this.refresh();
  }

  async changeTool(tool: LocalTool) {
    await this.act(() => { this.tool = tool; this.polygon = []; this.state.setPreview("none"); });
    if (tool === "box") this.showBox();
    else { this.gizmo?.detach(); this.box.visible = false; this.restoreGizmoNavigation(); }
    this.notify();
  }

  async setRange(mode: Exclude<LocalSelectionMode, "box">) {
    await this.act(() => { this.mode = mode; this.anchor = null; this.state.setPreview("none"); });
  }

  async preview(mode: LocalPreview) { await this.act(() => this.state.setPreview(mode)); }

  async selectPolygon(polygon: Pixel[], pick = false) {
    if (!this.editable || this.state.previewMode !== "none") throw new Error("请先返回编辑画面再选择");
    this.cancel();
    const serial = this.serial, revision = this.state.revision;
    await this.refresh();
    if (!this.ready || serial !== this.serial || revision !== this.state.revision) return;
    const view = this.viewer.localEditView(this.handle);
    if (view.signature !== this.signature) {
      this.signature = view.signature; this.cameraGeneration++;
      if (this.anchor !== null) this.depthRange = [NaN, NaN];
      this.anchor = null;
    }
    const operation = this.operation;
    this.selecting = true; this.error = ""; if (pick) this.anchor = null; this.notify();
    try {
      const result = await this.client.select({ ...view, cameraGeneration: this.cameraGeneration, polygon,
        visible: this.state.masks.visible, layerTolerance: 0.02, mode: pick ? "surface" : this.tool === "box" ? "box" : this.mode,
        depthRange: [...this.depthRange], box: this.tool === "box" ? this.boxBounds() : undefined });
      if (!result || !this.ready || serial !== this.serial || revision !== this.state.revision ||
          view.signature !== this.viewer.localEditView(this.handle).signature) return;
      if (pick) {
        this.anchor = result.surfaceDepth ?? null;
        if (this.anchor === null) throw new Error("此处没有可信表层，请换位置、手动设置深度或使用三维盒；不会吸附背景");
        const thickness = Math.max(0.001, this.anchor * 0.02);
        this.depthRange = [Math.max(0.01, this.anchor - thickness), this.anchor + thickness];
        this.mode = "depth";
      } else {
        this.state.select(result.selected, operation);
        if (!result.selectedCount) this.error = "本次没有命中高斯；未扩大范围或切换为穿透";
        await this.refresh();
      }
    } finally { if (serial === this.serial) { this.selecting = false; this.notify(); } }
  }

  private point(event: PointerEvent): Pixel {
    const rect = this.canvas.getBoundingClientRect(), view = this.viewer.localEditView(this.handle);
    return [Math.max(0, Math.min(view.width, (event.clientX - rect.left) * view.width / rect.width)),
      Math.max(0, Math.min(view.height, (event.clientY - rect.top) * view.height / rect.height))];
  }

  private pointerDown = (event: PointerEvent) => {
    if (!this.editable || event.button !== 0 || event.altKey || this.tool === "navigate" || this.tool === "box") return;
    event.preventDefault(); event.stopImmediatePropagation(); this.canvas.focus();
    if (this.state.previewMode !== "none") { this.report(new Error("请先返回编辑画面")); return; }
    if (this.drag) return;
    const point = this.point(event);
    this.cancel(); this.error = "";
    if (this.tool === "pick") {
      const view = this.viewer.localEditView(this.handle), x = Math.min(view.width - 1, Math.floor(point[0])), y = Math.min(view.height - 1, Math.floor(point[1]));
      void this.selectPolygon(rectangle([x, y], [x + 1, y + 1]), true).catch(e => this.report(e)); return;
    }
    this.drag = { id: event.pointerId, start: point, points: [point], controlsEnabled: this.viewer.controls?.enabled ?? false };
    if (this.viewer.controls) this.viewer.controls.enabled = false;
    this.canvas.setPointerCapture(event.pointerId); this.polygon = []; this.notify();
  };

  private pointerMove = (event: PointerEvent) => {
    if (!this.drag || event.pointerId !== this.drag.id) return;
    event.preventDefault(); event.stopImmediatePropagation();
    const point = this.point(event);
    if (this.tool === "rectangle") this.polygon = rectangle(this.drag.start, point);
    else {
      const last = this.drag.points.at(-1)!;
      if (Math.hypot(point[0] - last[0], point[1] - last[1]) < 2) return;
      if (this.drag.points.length >= 128) {
        this.endDrag(); this.report(new Error("套索超过128顶点，请缩短路径；未简化或扩大选区")); return;
      }
      this.drag.points.push(point); this.polygon = [...this.drag.points];
    }
    this.notify();
  };

  private pointerUp = (event: PointerEvent) => {
    if (!this.drag || event.pointerId !== this.drag.id) return;
    this.pointerMove(event);
    if (!this.drag) return;
    const polygon = this.polygon.map(p => [...p] as Pixel);
    this.endDrag(); this.notify();
    void this.selectPolygon(polygon).catch(e => this.report(e));
  };
  private pointerCancel = (event: PointerEvent) => {
    if (!this.drag || event.pointerId !== this.drag.id) return;
    this.cancel(); this.endDrag(); this.notify();
  };
  private endDrag() {
    if (!this.drag) return;
    const drag = this.drag; this.drag = null; this.polygon = [];
    if (this.viewer.controls) this.viewer.controls.enabled = drag.controlsEnabled;
    if (this.canvas.hasPointerCapture(drag.id)) this.canvas.releasePointerCapture(drag.id);
  }

  async deleteSelection() {
    if (this.state.previewMode === "original") throw new Error("原始对照中不能删除，请先返回编辑画面");
    const counts = this.state.counts;
    let confirmed = false;
    if (counts.deletable > counts.visible / 2 && counts.deletable < counts.visible) {
      confirmed = window.confirm(`将隐藏 ${counts.deletable} / ${counts.visible} 个可见高斯（超过50%），是否继续？`);
      if (!confirmed) return;
    }
    await this.act(() => this.state.deleteSelected(confirmed));
  }

  keyDown = (event: KeyboardEvent) => {
    const action = localEditorShortcut(event);
    if (!action || !this.editable) return;
    event.preventDefault(); event.stopPropagation();
    const work = action === "delete" ? this.deleteSelection() : this.act(() => {
      if (action === "undo") this.state.undo();
      else if (action === "redo") this.state.redo();
      else { this.state.clearSelection(); this.polygon = []; }
    });
    void work.catch(e => this.report(e));
  };

  private showBox() {
    if (!this.gizmo) {
      const cube = new THREE.BoxGeometry(1, 1, 1), geometry = new THREE.EdgesGeometry(cube);
      cube.dispose();
      this.boxLines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x55ccff, depthTest: false }));
      this.box.add(this.boxLines); this.boxLines.renderOrder = 100;
      this.handle.bounds.getCenter(this.box.position); this.handle.bounds.getSize(this.box.scale);
      this.box.scale.max(new THREE.Vector3(0.001, 0.001, 0.001));
      this.handle.sourceRoot.add(this.box);
      this.gizmo = new TransformControls(this.viewer.camera, this.canvas);
      this.gizmo.setSpace("local");
      this.handle.overlay.add(this.gizmo.getHelper());
      this.gizmo.addEventListener("dragging-changed", e => {
        if (e.value) {
          this.cancel(); this.gizmoControlsEnabled = this.viewer.controls?.enabled ?? false;
          if (this.viewer.controls) this.viewer.controls.enabled = false;
        } else this.restoreGizmoNavigation();
        this.notify();
      });
      this.gizmo.addEventListener("objectChange", () => {
        this.box.scale.max(new THREE.Vector3(0.001, 0.001, 0.001)); this.cancel(); this.notify();
      });
    }
    this.box.visible = true; this.gizmo.attach(this.box);
  }
  private restoreGizmoNavigation() {
    if (this.gizmoControlsEnabled === null) return;
    if (this.viewer.controls) this.viewer.controls.enabled = this.gizmoControlsEnabled;
    this.gizmoControlsEnabled = null;
  }
  setBoxMode(mode: "translate" | "scale") { this.cancel(); this.gizmo?.setMode(mode); this.notify(); }
  private boxBounds() {
    const half = this.box.scale.clone().multiplyScalar(0.5);
    return { min: this.box.position.clone().sub(half).toArray() as [number, number, number],
      max: this.box.position.clone().add(half).toArray() as [number, number, number] };
  }
  async selectBox() {
    const view = this.viewer.localEditView(this.handle);
    await this.selectPolygon(rectangle([0, 0], [view.width, view.height]));
  }

  async dispose() {
    if (this.disposed) return;
    this.disposed = true; this.cancel(); this.endDrag(); this.restoreGizmoNavigation();
    cancelAnimationFrame(this.frame); this.client.dispose();
    for (const [type, listener] of [["pointerdown", this.pointerDown], ["pointermove", this.pointerMove],
      ["pointerup", this.pointerUp], ["pointercancel", this.pointerCancel], ["lostpointercapture", this.pointerCancel]] as const) {
      this.canvas.removeEventListener(type, listener, true);
    }
    this.canvas.removeEventListener("keydown", this.keyDown); this.canvas.tabIndex = this.oldTabIndex;
    this.gizmo?.detach(); this.gizmo?.getHelper().removeFromParent(); this.gizmo?.dispose();
    this.box.removeFromParent(); this.boxLines?.geometry.dispose();
    (this.boxLines?.material as THREE.Material | undefined)?.dispose();
    await this.rendering?.catch(() => {});
    await this.viewer.endLocalEdit(this.handle);
  }
}
