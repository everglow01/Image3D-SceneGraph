import { useEffect, useRef, useState } from "react";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import * as THREE from "three";
import {
  parseContentLength,
  parseGaussianExportMetadata,
  viewerAlphaThreshold
} from "./gaussianViewerMetadata";

type GaussianSplatViewerProps = {
  sourceUrl: string | null;
  metadataUrl: string | null;
};

type ViewPreset = "fit" | "top" | "front" | "side";

type SceneFrame = {
  center: THREE.Vector3;
  radius: number;
};

const FALLBACK_FRAME: SceneFrame = {
  center: new THREE.Vector3(0, 0, 0),
  radius: 1.5
};

const VIEW_PRESETS: Record<ViewPreset, { label: string; direction: THREE.Vector3; up: THREE.Vector3 }> = {
  fit: {
    label: "Reset",
    direction: new THREE.Vector3(0.75, 1, 0.45),
    up: new THREE.Vector3(0, 0, 1)
  },
  top: {
    label: "Top",
    direction: new THREE.Vector3(0, 0, 1),
    up: new THREE.Vector3(0, 1, 0)
  },
  front: {
    label: "Front",
    direction: new THREE.Vector3(0, 1, 0.08),
    up: new THREE.Vector3(0, 0, 1)
  },
  side: {
    label: "Side",
    direction: new THREE.Vector3(1, 0, 0.08),
    up: new THREE.Vector3(0, 0, 1)
  }
};

function configureControls(viewer: GaussianSplats3D.Viewer, frame: SceneFrame) {
  const controls = viewer.controls;
  if (!controls) {
    return;
  }

  controls.rotateSpeed = 0.45;
  controls.zoomSpeed = 0.8;
  controls.panSpeed = 0.7;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.zoomToCursor = false;
  controls.minDistance = Math.max(frame.radius * 0.08, 0.01);
  controls.maxDistance = Math.max(frame.radius * 20, 20);
  controls.minPolarAngle = 0.01;
  controls.maxPolarAngle = Math.PI - 0.01;
  controls.target.copy(frame.center);
  controls.update();
}

function getSceneFrame(
  viewer: GaussianSplats3D.Viewer,
  sceneCenter: [number, number, number] | null,
  sceneRadius: number | null
): SceneFrame {
  const box = viewer.getSplatMesh().computeBoundingBox(true);
  const center = sceneCenter ? new THREE.Vector3(...sceneCenter) : new THREE.Vector3();
  const size = new THREE.Vector3();
  if (!sceneCenter) {
    box.getCenter(center);
  }
  box.getSize(size);

  if (!Number.isFinite(center.x) || !Number.isFinite(size.length()) || size.length() <= 0) {
    return FALLBACK_FRAME;
  }

  return {
    center,
    radius: sceneRadius ?? Math.max(size.length() * 0.5, 0.5)
  };
}

function applyViewPreset(
  viewer: GaussianSplats3D.Viewer,
  frame: SceneFrame,
  preset: ViewPreset,
  upMultiplier: 1 | -1
) {
  const config = VIEW_PRESETS[preset];
  const direction = config.direction.clone().normalize();
  const up = config.up.clone().multiplyScalar(upMultiplier);
  const distance = Math.max(frame.radius * 2.4, 1.5);
  const camera = viewer.camera;
  const controls = viewer.controls;

  camera.position.copy(frame.center).add(direction.multiplyScalar(distance));
  camera.up.copy(up).normalize();
  camera.near = Math.max(frame.radius / 100, 0.01);
  camera.far = Math.max(frame.radius * 100, 100);
  camera.lookAt(frame.center);
  camera.updateProjectionMatrix();

  if (controls) {
    controls.target.copy(frame.center);
    controls.saveState();
    controls.update();
  }

  viewer.forceRenderNextFrame?.();
}

export function GaussianSplatViewer({ sourceUrl, metadataUrl }: GaussianSplatViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<GaussianSplats3D.Viewer | null>(null);
  const sceneFrameRef = useRef<SceneFrame>(FALLBACK_FRAME);
  const upMultiplierRef = useRef<1 | -1>(1);
  const [viewerState, setViewerState] = useState("idle");
  const [assetBytes, setAssetBytes] = useState<number | null>(null);

  const setView = (preset: ViewPreset) => {
    const viewer = viewerRef.current;
    if (!viewer || viewerState !== "ready") {
      return;
    }
    applyViewPreset(viewer, sceneFrameRef.current, preset, upMultiplierRef.current);
  };

  const flipView = () => {
    const viewer = viewerRef.current;
    if (!viewer || viewerState !== "ready") {
      return;
    }
    const frame = sceneFrameRef.current;
    const direction = viewer.camera.position.clone().sub(frame.center);
    upMultiplierRef.current = upMultiplierRef.current === 1 ? -1 : 1;
    viewer.camera.position.copy(frame.center).sub(direction);
    viewer.camera.up.multiplyScalar(-1).normalize();
    viewer.camera.lookAt(frame.center);
    if (viewer.controls) {
      viewer.controls.target.copy(frame.center);
      viewer.controls.update();
      viewer.controls.saveState();
    }
    viewer.forceRenderNextFrame?.();
  };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !sourceUrl || !metadataUrl) {
      setViewerState("idle");
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    setViewerState("loading");
    setAssetBytes(null);
    mount.replaceChildren();
    void fetch(sourceUrl, { method: "HEAD", signal: controller.signal })
      .then((response) => {
        const bytes = parseContentLength(response.headers.get("content-length"));
        if (!cancelled && bytes !== null) {
          setAssetBytes(bytes);
        }
      })
      .catch(() => {
        // Asset size is optional; the viewer still owns the actual GET and error state.
      });

    let viewer: GaussianSplats3D.Viewer | null = null;
    const load = async () => {
      try {
        const response = await fetch(metadataUrl, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Gaussian export metadata request failed: ${response.status}`);
        }
        const metadata = parseGaussianExportMetadata(await response.json());
        if (cancelled) {
          return;
        }
        viewer = new GaussianSplats3D.Viewer({
          rootElement: mount,
          cameraUp: [0, 0, 1],
          initialCameraPosition: [0, 1.2, 3],
          initialCameraLookAt: [0, 0, 0],
          sharedMemoryForWorkers: false,
          sphericalHarmonicsDegree: metadata.sh_degree,
          ignoreDevicePixelRatio: true,
          integerBasedSort: false
        });
        viewerRef.current = viewer;
        await viewer.addSplatScene(sourceUrl, {
          showLoadingUI: true,
          progressiveLoad: true,
          splatAlphaRemovalThreshold: viewerAlphaThreshold(metadata.viewer_minimum_opacity)
        });
        if (cancelled) {
          return;
        }
        sceneFrameRef.current = getSceneFrame(
          viewer,
          metadata.scene_center,
          metadata.scene_radius_p95
        );
        configureControls(viewer, sceneFrameRef.current);
        applyViewPreset(viewer, sceneFrameRef.current, "fit", upMultiplierRef.current);
        viewer.start();
        setViewerState("ready");
      } catch {
        if (!cancelled) {
          setViewerState("error");
        }
      }
    };
    void load();

    return () => {
      cancelled = true;
      controller.abort();
      const activeViewer = viewerRef.current;
      viewerRef.current = null;
      if (activeViewer) {
        activeViewer.stop();
        void activeViewer.dispose().catch(() => {
          // The viewer owns its internal DOM and may already have removed it.
        });
      }
      mount.replaceChildren();
    };
  }, [sourceUrl, metadataUrl]);

  return (
    <div className="viewer-surface">
      <div className="splat-root" ref={mountRef} />
      {sourceUrl && (
        <div className="splat-toolbar" aria-label="Gaussian splat camera controls">
          {(Object.keys(VIEW_PRESETS) as ViewPreset[]).map((preset) => (
            <button
              className="viewer-tool-button"
              disabled={viewerState !== "ready"}
              key={preset}
              onClick={() => setView(preset)}
              type="button"
            >
              {VIEW_PRESETS[preset].label}
            </button>
          ))}
          <button className="viewer-tool-button" disabled={viewerState !== "ready"} onClick={flipView} type="button">
            Flip
          </button>
        </div>
      )}
      {viewerState === "ready" && (
        <div className="viewer-hint">
          Canonical normalized coordinates · arbitrary units
          {assetBytes === null ? "" : ` · ${(assetBytes / 1_048_576).toFixed(1)} MiB`}
          {" · Drag orbit · Shift/right drag pan · Wheel zoom · Click geometry to set pivot"}
        </div>
      )}
      {viewerState !== "ready" && (
        <div className="viewer-overlay">
          {viewerState === "idle" && "No Gaussian splat"}
          {viewerState === "loading" && "Loading Gaussian splat"}
          {viewerState === "error" && "Failed to load Gaussian splat"}
        </div>
      )}
    </div>
  );
}
