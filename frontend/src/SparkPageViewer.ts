import * as THREE from "three";
import { OrbitControls, type ViewerControls } from "@mkkellogg/gaussian-splats-3d";
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { captureP0Source, type P0Source } from "./localGaussianP0Source.ts";
import { LocalGaussianDisplay } from "./localGaussianDisplay.ts";

export type SparkLocalEdit = { source: P0Source; sourceRoot: THREE.Group; overlay: THREE.Group; bounds: THREE.Box3 };

export class SparkPageViewer {
  readonly camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  readonly renderer: THREE.WebGLRenderer;
  controls: ViewerControls | null;
  private readonly orbit: ViewerControls & { dispose(): void };
  private readonly spark: SparkRenderer;
  private readonly resizeObserver: ResizeObserver;
  private readonly abortController = new AbortController();
  private mesh: SplatMesh | null = null;
  private loading: Promise<void> | null = null;
  private pending: Promise<void> | null = null;
  private disposal: Promise<void> | null = null;
  private frame: number | null = null;
  private disposed = false;
  private editQueue: Promise<void> = Promise.resolve();
  private editsPending = 0;
  private editor: { handle: SparkLocalEdit; display: LocalGaussianDisplay } | null = null;

  constructor(
    private readonly mount: HTMLElement,
    private readonly scene: THREE.Scene,
    private readonly onError: (error: unknown) => void
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(1);
    this.renderer.setClearColor(0, 1);
    this.renderer.toneMapping = THREE.NoToneMapping;
    let orbit: (ViewerControls & { dispose(): void }) | undefined;
    let spark: SparkRenderer | undefined;
    let observer: ResizeObserver | undefined;
    try {
      this.orbit = orbit = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls = this.orbit;
      this.spark = spark = new SparkRenderer({
        renderer: this.renderer,
        autoUpdate: false,
        enableLod: false,
        enableDriveLod: false,
        enableLodFetching: false,
        accumExtSplats: true,
        sortRadial: false,
        minSortIntervalMs: 0,
        preBlurAmount: 0.3,
        blurAmount: 0,
        focalAdjustment: 1,
        minAlpha: 1 / 255,
        maxStdDev: 3,
        maxPixelRadius: 1e10
      });
      scene.add(this.spark);
      mount.appendChild(this.renderer.domElement);
      this.renderer.domElement.addEventListener("webglcontextlost", this.contextLost);
      this.resizeObserver = observer = new ResizeObserver(this.resize);
      this.resizeObserver.observe(mount);
      this.resize();
    } catch (error) {
      observer?.disconnect();
      orbit?.dispose();
      spark?.removeFromParent();
      spark?.dispose();
      this.renderer.dispose();
      this.renderer.forceContextLoss();
      this.renderer.domElement.remove();
      throw error;
    }
  }

  private resize = () => {
    if (this.disposed) return;
    const width = Math.max(this.mount.clientWidth, 1);
    const height = Math.max(this.mount.clientHeight, 1);
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  };

  private contextLost = (event: Event) => {
    event.preventDefault();
    this.fail(new Error("Spark WebGL 上下文丢失，请切回旧查看器"));
  };

  private fail(error: unknown) {
    this.stop();
    if (!this.disposed) this.onError(error);
  }

  load(sourceUrl: string, shDegree: number, rotation: THREE.Quaternion | null): Promise<void> {
    this.loading = this.loadModel(sourceUrl, shDegree, rotation);
    return this.loading;
  }

  private async loadModel(sourceUrl: string, shDegree: number, rotation: THREE.Quaternion | null) {
    const response = await fetch(sourceUrl, { signal: this.abortController.signal });
    if (!response.ok || !response.body) throw new Error(`Spark 模型加载失败：${response.status}`);
    if (this.disposed) {
      await response.body.cancel();
      return;
    }
    this.mesh = new SplatMesh({
      stream: response.body,
      fileType: SplatFileType.PLY,
      extSplats: true,
      lod: false,
      enableLod: false
    });
    await this.mesh.initialized;
    if (this.disposed) return;
    if (!this.mesh.numSplats || this.mesh.extSplats?.getNumSh() !== shDegree) {
      throw new Error("Spark 模型数量或 SH 阶数与导出声明不符");
    }
    this.mesh.maxSh = shDegree;
    this.mesh.extSplats.setMaxSh(shDegree);
    if (rotation) this.mesh.quaternion.copy(rotation);
    this.scene.add(this.mesh);
    await this.spark.update({ scene: this.scene, camera: this.camera });
    if (this.disposed) return;
    if (this.spark.display.numSplats !== this.mesh.numSplats || this.spark.sorting || this.spark.sortDirty) {
      throw new Error("Spark 初始全量排序未完成");
    }
  }

  getBoundingBox(): THREE.Box3 {
    if (!this.mesh) throw new Error("Spark 模型尚未加载");
    this.mesh.updateMatrixWorld(true);
    return this.mesh.getBoundingBox().applyMatrix4(this.mesh.matrixWorld);
  }

  start() {
    if (!this.disposed && this.frame === null) this.frame = requestAnimationFrame(this.tick);
  }

  private tick = () => {
    this.frame = null;
    if (this.disposed) return;
    try {
      if (this.controls?.enabled) this.controls.update();
      // A camera update may still be sorting while the previous complete frame is drawn.
      if (!this.pending && !this.editsPending) {
        this.pending = this.spark.update({ scene: this.scene, camera: this.camera })
          .catch((error: unknown) => this.fail(error))
          .finally(() => { this.pending = null; });
      }
      this.renderer.render(this.scene, this.camera);
      const gl = this.renderer.getContext();
      if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw new Error("Spark WebGL 渲染失败");
      this.frame = requestAnimationFrame(this.tick);
    } catch (error) {
      this.fail(error);
    }
  };

  private serializeEdit<T>(action: () => T | Promise<T>): Promise<T> {
    this.editsPending++;
    const work = this.editQueue.then(async () => {
      await this.loading;
      await this.pending;
      if (this.disposed) throw new Error("查看器已释放");
      return action();
    });
    this.editQueue = work.then(() => {}, () => {}).finally(() => { this.editsPending--; });
    return work;
  }

  beginLocalEdit(sha256: string, signal: AbortSignal): Promise<SparkLocalEdit> {
    return this.serializeEdit(async () => {
      if (this.editor || !this.mesh || this.controls !== this.orbit) throw new Error("请在已加载的 Spark 轨道模式进入本地编辑");
      const source = await captureP0Source(this.mesh, sha256, signal);
      signal.throwIfAborted();
      if (this.disposed) throw new Error("查看器已释放");
      this.mesh.updateMatrixWorld(true);
      const sourceRoot = new THREE.Group(), overlay = new THREE.Group();
      sourceRoot.matrixAutoUpdate = false;
      sourceRoot.matrix.copy(this.mesh.matrixWorld);
      const handle = { source, sourceRoot, overlay, bounds: this.mesh.getBoundingBox().clone() };
      const display = new LocalGaussianDisplay(this.mesh);
      this.editor = { handle, display };
      this.scene.add(sourceRoot, overlay);
      return handle;
    });
  }

  localEditView(handle: SparkLocalEdit) {
    if (this.disposed || this.editor?.handle !== handle || !this.mesh) throw new Error("本地编辑源已失效");
    this.scene.updateMatrixWorld(true); this.camera.updateMatrixWorld(true);
    handle.sourceRoot.matrix.copy(this.mesh.matrixWorld);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    const modelToView = new THREE.Matrix4().multiplyMatrices(this.camera.matrixWorldInverse, this.mesh.matrixWorld).elements;
    const projection = this.camera.projectionMatrix.elements.slice();
    return { width: size.x, height: size.y, modelToView, projection,
      signature: JSON.stringify([size.x, size.y, modelToView, projection]) };
  }

  updateLocalEdit(handle: SparkLocalEdit, visible: Uint8Array, selected: Uint8Array, protectedMask: Uint8Array, highlight: boolean) {
    const masks = [visible.slice(), selected.slice(), protectedMask.slice()];
    return this.serializeEdit(async () => {
      if (this.editor?.handle !== handle) throw new Error("本地编辑源已失效");
      this.editor.display.update(masks[0], masks[1], masks[2], highlight);
      await this.spark.update({ scene: this.scene, camera: this.camera });
    });
  }

  private releaseEditor() {
    if (!this.editor) return;
    this.editor.handle.sourceRoot.removeFromParent(); this.editor.handle.overlay.removeFromParent();
    this.editor.display.dispose(); this.editor = null;
  }

  endLocalEdit(handle: SparkLocalEdit) {
    if (this.disposed) return this.disposal ?? Promise.resolve();
    return this.serializeEdit(() => { if (this.editor?.handle === handle) this.releaseEditor(); });
  }

  stop() {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
  }

  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.disposed = true;
    this.stop();
    this.abortController.abort();
    this.resizeObserver.disconnect();
    this.orbit.dispose();
    this.renderer.domElement.removeEventListener("webglcontextlost", this.contextLost);
    this.disposal = (async () => {
      // Decoding and sorting must settle before their textures and workers are released.
      await Promise.allSettled([this.loading, this.pending, this.editQueue]);
      try {
        this.releaseEditor();
        this.mesh?.removeFromParent();
        this.mesh?.dispose();
        this.spark.removeFromParent();
        this.spark.dispose();
      } finally {
        this.renderer.dispose();
        this.renderer.forceContextLoss();
        this.renderer.domElement.remove();
      }
    })();
    return this.disposal;
  }
}
