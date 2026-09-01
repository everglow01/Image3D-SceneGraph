import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import {
  applyAxisSigns as applyCameraAxisSigns,
  cameraLinePositions,
  parseAlignmentTransform,
  parseCameraFrames,
  transformCameraFrames,
  type CameraFrame,
  type Mat4,
  type Vec3
} from "./cameraOverlay";
import { robustCloudBounds } from "./pointCloudBounds";

type PointCloudViewerProps = {
  sourceUrl: string | null;
  camerasUrl?: string | null;
  alignmentDiagnosticsUrl?: string | null;
  pointCloudVariant?: "raw" | "aligned";
  viewArtifactStem?: string;
};

type PointCloudViewState = {
  schemaVersion: 1;
  cameraPosition: [number, number, number];
  controlsTarget: [number, number, number];
  cameraUp: [number, number, number];
  fov: number;
  near: number;
  far: number;
  axisSigns: AxisSigns;
  pointSize: number;
  captureViewport: [number, number];
};

const G1_CAPTURE_VIEWPORT: [number, number] = [1600, 1000];

type AxisSigns = {
  x: 1 | -1;
  y: 1 | -1;
  z: 1 | -1;
};

export function PointCloudViewer({
  sourceUrl,
  camerasUrl = null,
  alignmentDiagnosticsUrl = null,
  pointCloudVariant = "raw",
  viewArtifactStem = "point-cloud-view"
}: PointCloudViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const cameraOverlayRef = useRef<THREE.LineSegments | null>(null);
  const cloudCenterRef = useRef<Vec3>([0, 0, 0]);
  const cloudRadiusRef = useRef(1);
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const [viewerState, setViewerState] = useState("idle");
  const [viewMessage, setViewMessage] = useState("");
  const [artifactName, setArtifactName] = useState(viewArtifactStem);
  const [axisSigns, setAxisSigns] = useState<AxisSigns>({ x: 1, y: 1, z: 1 });
  const [pointSize, setPointSize] = useState(0.035);
  const [showCameras, setShowCameras] = useState(true);
  const [camerasAvailable, setCamerasAvailable] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f7f8);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
    camera.up.set(0, 0, 1);
    camera.position.set(1.8, -2.4, 1.5);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controlsRef.current = controls;

    const grid = new THREE.GridHelper(2, 10, 0x9aa3a7, 0xd2d8dc);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const axes = new THREE.AxesHelper(0.8);
    scene.add(axes);

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    let frame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = window.requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      removeCameraOverlay(scene, cameraOverlayRef);
      scene.traverse((object) => {
        if (object instanceof THREE.Points || object instanceof THREE.Mesh) {
          object.geometry.dispose();
        }
      });
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!scene || !camera || !controls) {
      return;
    }

    if (pointsRef.current) {
      scene.remove(pointsRef.current);
      pointsRef.current.geometry.dispose();
      const material = pointsRef.current.material;
      if (Array.isArray(material)) {
        material.forEach((item) => item.dispose());
      } else {
        material.dispose();
      }
      pointsRef.current = null;
    }

    if (!sourceUrl) {
      setViewerState("idle");
      return;
    }

    let cancelled = false;
    setViewerState("loading");

    const loader = new PLYLoader();
    loader.load(
      sourceUrl,
      (geometry) => {
        if (cancelled) {
          geometry.dispose();
          return;
        }

        geometry.computeVertexNormals();
        const attribute = geometry.getAttribute("position");
        const bounds = robustCloudBounds(attribute.array as ArrayLike<number>, attribute.count);
        const radius = bounds.radius;
        cloudCenterRef.current = bounds.center;
        cloudRadiusRef.current = radius;
        geometry.translate(-bounds.center[0], -bounds.center[1], -bounds.center[2]);

        const hasVertexColors = geometry.hasAttribute("color");
        const material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: hasVertexColors,
          color: hasVertexColors ? 0xffffff : 0x1f6f78
        });

        const points = new THREE.Points(geometry, material);
        applyAxisSigns(points, axisSigns);
        scene.add(points);
        pointsRef.current = points;

        camera.position.set(radius * 1.8, -radius * 2.2, radius * 1.4);
        camera.up.set(0, 0, 1);
        camera.near = Math.max(radius / 100, 0.001);
        camera.far = Math.max(radius * 100, 100);
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        controls.minDistance = radius * 0.02;
        controls.maxDistance = radius * 20;
        controls.update();

        setViewerState("ready");
      },
      undefined,
      () => {
        if (!cancelled) {
          setViewerState("error");
        }
      }
    );

    return () => {
      cancelled = true;
    };
  }, [sourceUrl]);

  useEffect(() => {
    if (pointsRef.current) {
      applyAxisSigns(pointsRef.current, axisSigns);
    }
    if (cameraOverlayRef.current) {
      applyAxisSigns(cameraOverlayRef.current, axisSigns);
    }
  }, [axisSigns]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) {
      return;
    }
    removeCameraOverlay(scene, cameraOverlayRef);
    setCamerasAvailable(false);
    if (!camerasUrl || viewerState !== "ready") {
      return;
    }

    let cancelled = false;
    Promise.all([
      fetchJson(camerasUrl),
      pointCloudVariant === "aligned"
        ? alignmentDiagnosticsUrl
          ? fetchJson(alignmentDiagnosticsUrl)
          : Promise.reject(new Error("Aligned camera overlay requires alignment diagnostics"))
        : Promise.resolve(null)
    ])
      .then(([cameraPayload, alignmentPayload]) => {
        if (cancelled) {
          return;
        }
        const frames = parseCameraFrames(cameraPayload, cloudRadiusRef.current * 0.02);
        const alignment: Mat4 | null = alignmentPayload
          ? parseAlignmentTransform(alignmentPayload)
          : null;
        const transformed = transformCameraFrames(
          frames,
          alignment,
          cloudCenterRef.current
        );
        const overlay = buildCameraOverlay(transformed);
        overlay.visible = showCameras;
        applyAxisSigns(overlay, axisSigns);
        scene.add(overlay);
        cameraOverlayRef.current = overlay;
        setCamerasAvailable(true);
      })
      .catch(() => {
        if (!cancelled) {
          setCamerasAvailable(false);
        }
      });

    return () => {
      cancelled = true;
      removeCameraOverlay(scene, cameraOverlayRef);
    };
  }, [camerasUrl, alignmentDiagnosticsUrl, pointCloudVariant, sourceUrl, viewerState]);

  useEffect(() => {
    if (cameraOverlayRef.current) {
      cameraOverlayRef.current.visible = showCameras;
    }
  }, [showCameras]);

  useEffect(() => {
    const material = pointsRef.current?.material;
    if (material instanceof THREE.PointsMaterial) {
      material.size = pointSize;
      material.needsUpdate = true;
    }
  }, [pointSize]);

  function toggleAxis(axis: keyof AxisSigns) {
    setAxisSigns((current) => ({
      ...current,
      [axis]: current[axis] === 1 ? -1 : 1
    }));
  }

  function currentViewState(): PointCloudViewState | null {
    const container = containerRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!container || !camera || !controls || viewerState !== "ready") {
      return null;
    }
    return {
      schemaVersion: 1,
      cameraPosition: camera.position.toArray() as [number, number, number],
      controlsTarget: controls.target.toArray() as [number, number, number],
      cameraUp: camera.up.toArray() as [number, number, number],
      fov: camera.fov,
      near: camera.near,
      far: camera.far,
      axisSigns,
      pointSize,
      captureViewport: G1_CAPTURE_VIEWPORT
    };
  }

  function exportViewState() {
    const state = currentViewState();
    if (!state) {
      return;
    }
    const blob = new Blob([`${JSON.stringify(state, null, 2)}\n`], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${artifactName || viewArtifactStem}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setViewMessage("视图状态已导出");
  }

  function capturePng() {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    const scene = sceneRef.current;
    if (!renderer || !camera || !scene || viewerState !== "ready") {
      return;
    }
    const originalSize = renderer.getSize(new THREE.Vector2());
    const originalPixelRatio = renderer.getPixelRatio();
    const originalAspect = camera.aspect;
    renderer.setPixelRatio(1);
    renderer.setSize(G1_CAPTURE_VIEWPORT[0], G1_CAPTURE_VIEWPORT[1], false);
    camera.aspect = G1_CAPTURE_VIEWPORT[0] / G1_CAPTURE_VIEWPORT[1];
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    const url = renderer.domElement.toDataURL("image/png");
    renderer.setPixelRatio(originalPixelRatio);
    renderer.setSize(originalSize.x, originalSize.y, false);
    camera.aspect = originalAspect;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${artifactName || viewArtifactStem}.png`;
    anchor.click();
    setViewMessage("已生成 1600×1000 PNG 截图");
  }

  function applyViewState(state: PointCloudViewState) {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls || viewerState !== "ready" || state.schemaVersion !== 1) {
      throw new Error("Point cloud must be ready and view schema must be version 1");
    }
    camera.position.fromArray(state.cameraPosition);
    camera.up.fromArray(state.cameraUp);
    camera.fov = state.fov;
    camera.near = state.near;
    camera.far = state.far;
    camera.updateProjectionMatrix();
    controls.target.fromArray(state.controlsTarget);
    controls.update();
    setAxisSigns(state.axisSigns);
    setPointSize(state.pointSize);
  }

  async function importViewState(file: File | undefined) {
    if (!file) {
      return;
    }
    try {
      applyViewState(JSON.parse(await file.text()) as PointCloudViewState);
      setViewMessage(`已加载 ${file.name}`);
    } catch {
      setViewMessage("视图状态文件无效");
    } finally {
      if (importInputRef.current) {
        importInputRef.current.value = "";
      }
    }
  }

  return (
    <div className="viewer-surface" ref={containerRef}>
      <div className="pointcloud-toolbar" aria-label="点云坐标与视图控制">
        <button
          className={axisSigns.x === -1 ? "viewer-tool-button active" : "viewer-tool-button"}
          onClick={() => toggleAxis("x")}
          type="button"
        >
          X
        </button>
        <button
          className={axisSigns.y === -1 ? "viewer-tool-button active" : "viewer-tool-button"}
          onClick={() => toggleAxis("y")}
          type="button"
        >
          Y
        </button>
        <button
          className={axisSigns.z === -1 ? "viewer-tool-button active" : "viewer-tool-button"}
          onClick={() => toggleAxis("z")}
          type="button"
        >
          Z
        </button>
        <button
          className={showCameras && camerasAvailable ? "viewer-tool-button active" : "viewer-tool-button"}
          disabled={!camerasAvailable}
          onClick={() => setShowCameras((current) => !current)}
          type="button"
        >
          相机位姿
        </button>
        <input
          aria-label="视图资产名称"
          className="viewer-artifact-name"
          onChange={(event) => setArtifactName(event.target.value)}
          value={artifactName}
        />
        <button
          className="viewer-tool-button"
          disabled={viewerState !== "ready"}
          onClick={() => importInputRef.current?.click()}
          type="button"
        >
          加载视图
        </button>
        <button
          className="viewer-tool-button"
          disabled={viewerState !== "ready"}
          onClick={exportViewState}
          type="button"
        >
          保存视图
        </button>
        <button
          className="viewer-tool-button"
          disabled={viewerState !== "ready"}
          onClick={capturePng}
          type="button"
        >
          截图 PNG
        </button>
        <input
          accept="application/json,.json"
          aria-label="加载点云视图状态"
          hidden
          onChange={(event) => void importViewState(event.target.files?.[0])}
          ref={importInputRef}
          type="file"
        />
        {viewMessage && <span className="viewer-tool-message">{viewMessage}</span>}
        <label className="point-size-control">
          <span>点大小</span>
          <input
            aria-label="点大小"
            type="range"
            min={0.01}
            max={0.12}
            step={0.005}
            value={pointSize}
            onChange={(event) => setPointSize(Number(event.target.value))}
          />
        </label>
      </div>
      {viewerState !== "ready" && (
        <div className="viewer-overlay">
          {viewerState === "idle" && "尚未加载点云"}
          {viewerState === "loading" && "正在加载点云"}
          {viewerState === "error" && "点云加载失败"}
        </div>
      )}
    </div>
  );
}

function applyAxisSigns(object: THREE.Object3D, axisSigns: AxisSigns) {
  object.scale.set(axisSigns.x, axisSigns.y, axisSigns.z);
}

function buildCameraOverlay(frames: CameraFrame[]) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(cameraLinePositions(frames), 3));
  const material = new THREE.LineBasicMaterial({
    color: 0xe4572e,
    transparent: true,
    opacity: 0.82,
    depthTest: false
  });
  const overlay = new THREE.LineSegments(geometry, material);
  overlay.renderOrder = 2;
  return overlay;
}

function removeCameraOverlay(
  scene: THREE.Scene,
  ref: { current: THREE.LineSegments | null }
) {
  if (!ref.current) {
    return;
  }
  scene.remove(ref.current);
  ref.current.geometry.dispose();
  const material = ref.current.material;
  if (Array.isArray(material)) {
    material.forEach((item) => item.dispose());
  } else {
    material.dispose();
  }
  ref.current = null;
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}
