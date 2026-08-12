import { useEffect, useRef, useState } from "react";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Capsule } from "three/examples/jsm/math/Capsule.js";
import { Octree } from "three/examples/jsm/math/Octree.js";
import {
  deriveGaussianViewerFrame,
  parseContentLength,
  parseGaussianCameraPath,
  parseGaussianExportMetadata,
  viewerAlphaThreshold
} from "./gaussianViewerMetadata";
import {
  clampPitchRadians,
  defaultWalkSettings,
  parseNavigationContract,
  parseWalkSettings,
  planarWalkVelocity,
  pointInNavigationBoundary,
  type NavigationContract,
  type WalkSettings
} from "./walkNavigation";

type GaussianSplatViewerProps = {
  sourceUrl: string | null;
  metadataUrl: string | null;
  cameraPathUrl: string | null;
  collisionMeshUrl: string | null;
  navigationUrl: string | null;
  navigationStatus: string | null;
  navigationReason: string | null;
};

type ViewPreset = "fit" | "top" | "front" | "side";
type ViewerState = "idle" | "loading" | "ready" | "error";
type NavigationState = "idle" | "loading" | "ready" | "error";
type ViewerMode = "orbit" | "walk";

type SceneFrame = {
  center: THREE.Vector3;
  radius: number;
  up: THREE.Vector3;
};

type WalkRuntime = {
  contract: NavigationContract;
  octree: Octree;
  capsule: Capsule;
  spawnCapsule: Capsule;
  safeCapsule: Capsule;
  up: THREE.Vector3;
  basisU: THREE.Vector3;
  basisV: THREE.Vector3;
  baseForward: THREE.Vector3;
  baseRight: THREE.Vector3;
  yaw: number;
  pitch: number;
  keys: Set<string>;
  debugRoot: THREE.Group;
  collisionRoot: THREE.Object3D;
  capsuleHelper: THREE.Mesh;
  contactHelper: THREE.ArrowHelper;
  orbitControls: GaussianSplats3D.ViewerControls | null;
  animationFrame: number | null;
  previousTime: number | null;
  accumulator: number;
  boundaryHintTimer: number | null;
};

const FALLBACK_FRAME: SceneFrame = {
  center: new THREE.Vector3(0, 0, 0),
  radius: 1.5,
  up: new THREE.Vector3(0, 0, 1)
};
const WALK_SETTINGS_KEY = "image3d.walk-settings.v1";
const FIXED_STEP_SECONDS = 1 / 120;
const MAX_FRAME_SECONDS = 0.1;

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
  sceneRadius: number | null,
  cameraFrame: { center: [number, number, number]; up: [number, number, number] } | null
): SceneFrame {
  const box = viewer.getSplatMesh().computeBoundingBox(true);
  const center = cameraFrame
    ? new THREE.Vector3(...cameraFrame.center)
    : sceneCenter
      ? new THREE.Vector3(...sceneCenter)
      : new THREE.Vector3();
  const size = new THREE.Vector3();
  if (!cameraFrame && !sceneCenter) {
    box.getCenter(center);
  }
  box.getSize(size);

  if (!Number.isFinite(center.x) || !Number.isFinite(size.length()) || size.length() <= 0) {
    return FALLBACK_FRAME;
  }

  return {
    center,
    radius: sceneRadius ?? Math.max(size.length() * 0.5, 0.5),
    up: cameraFrame ? new THREE.Vector3(...cameraFrame.up) : new THREE.Vector3(0, 0, 1)
  };
}

function applyViewPreset(viewer: GaussianSplats3D.Viewer, frame: SceneFrame, preset: ViewPreset) {
  const config = VIEW_PRESETS[preset];
  const orientation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), frame.up);
  const direction = config.direction.clone().applyQuaternion(orientation).normalize();
  const up = config.up.clone().applyQuaternion(orientation).normalize();
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

export function GaussianSplatViewer({
  sourceUrl,
  metadataUrl,
  cameraPathUrl,
  collisionMeshUrl,
  navigationUrl,
  navigationStatus,
  navigationReason
}: GaussianSplatViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<GaussianSplats3D.Viewer | null>(null);
  const sceneFrameRef = useRef<SceneFrame>(FALLBACK_FRAME);
  const walkRuntimeRef = useRef<WalkRuntime | null>(null);
  const settingsRef = useRef<WalkSettings | null>(null);
  const viewerModeRef = useRef<ViewerMode>("orbit");
  const debugVisibleRef = useRef(false);
  const [viewerState, setViewerState] = useState<ViewerState>("idle");
  const [navigationState, setNavigationState] = useState<NavigationState>("idle");
  const [viewerMode, setViewerMode] = useState<ViewerMode>("orbit");
  const [assetBytes, setAssetBytes] = useState<number | null>(null);
  const [settings, setSettings] = useState<WalkSettings | null>(null);
  const [debugVisible, setDebugVisible] = useState(false);
  const [walkMessage, setWalkMessage] = useState("Orbit mode");
  const [boundaryHint, setBoundaryHint] = useState(false);

  const setView = (preset: ViewPreset) => {
    const viewer = viewerRef.current;
    if (!viewer || viewerState !== "ready" || viewerModeRef.current !== "orbit") {
      return;
    }
    applyViewPreset(viewer, sceneFrameRef.current, preset);
  };

  const enterWalk = () => {
    const viewer = viewerRef.current;
    const runtime = walkRuntimeRef.current;
    if (!viewer || !runtime || navigationState !== "ready") {
      return;
    }
    const canvas = viewer.renderer?.domElement;
    if (!canvas) {
      setWalkMessage("Pointer Lock is unavailable");
      return;
    }
    const request = canvas.requestPointerLock();
    if (request && "catch" in request) {
      void request.catch(() => setWalkMessage("Pointer Lock request was denied"));
    }
  };

  const exitWalk = () => {
    if (document.pointerLockElement) {
      document.exitPointerLock();
    } else {
      leaveWalkMode(
        viewerRef.current,
        walkRuntimeRef.current,
        viewerModeRef,
        setViewerMode,
        setWalkMessage
      );
    }
  };

  const resetWalk = () => {
    const viewer = viewerRef.current;
    const runtime = walkRuntimeRef.current;
    if (!viewer || !runtime) {
      return;
    }
    runtime.capsule.copy(runtime.spawnCapsule);
    runtime.safeCapsule.copy(runtime.spawnCapsule);
    runtime.yaw = 0;
    runtime.pitch = 0;
    runtime.keys.clear();
    applyWalkCamera(viewer, runtime, settingsRef.current);
    viewer.forceRenderNextFrame?.();
    setWalkMessage("Returned to the safe spawn");
  };

  const updateSettings = (next: WalkSettings) => {
    const runtime = walkRuntimeRef.current;
    if (!runtime) {
      return;
    }
    settingsRef.current = next;
    setSettings(next);
    localStorage.setItem(WALK_SETTINGS_KEY, JSON.stringify(next));
    const viewer = viewerRef.current;
    if (viewer && viewerModeRef.current === "walk") {
      applyWalkCamera(viewer, runtime, next);
      viewer.forceRenderNextFrame?.();
    }
  };

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    debugVisibleRef.current = debugVisible;
    const runtime = walkRuntimeRef.current;
    if (runtime) {
      runtime.debugRoot.visible = debugVisible;
      viewerRef.current?.forceRenderNextFrame?.();
    }
  }, [debugVisible]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !sourceUrl || !metadataUrl) {
      setViewerState("idle");
      setNavigationState("idle");
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    setViewerState("loading");
    setNavigationState(navigationUrl && collisionMeshUrl ? "loading" : "idle");
    setViewerMode("orbit");
    viewerModeRef.current = "orbit";
    setAssetBytes(null);
    setSettings(null);
    setWalkMessage("Orbit mode");
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
    let auxiliaryScene: THREE.Scene | null = null;
    const load = async () => {
      try {
        const response = await fetch(metadataUrl, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Gaussian export metadata request failed: ${response.status}`);
        }
        const metadata = parseGaussianExportMetadata(await response.json());
        const cameraFrame = cameraPathUrl
          ? await fetch(cameraPathUrl, { signal: controller.signal })
              .then(async (cameraResponse) => {
                if (!cameraResponse.ok) {
                  throw new Error(`Gaussian camera path request failed: ${cameraResponse.status}`);
                }
                return deriveGaussianViewerFrame(metadata, parseGaussianCameraPath(await cameraResponse.json()));
              })
              .catch(() => null)
          : null;
        if (cancelled) {
          return;
        }
        auxiliaryScene = new THREE.Scene();
        const cameraUp = cameraFrame?.up ?? [0, 0, 1];
        const cameraCenter = cameraFrame?.center ?? metadata.scene_center ?? [0, 0, 0];
        viewer = new GaussianSplats3D.Viewer({
          rootElement: mount,
          cameraUp,
          initialCameraPosition: [cameraCenter[0], cameraCenter[1] + 1.2, cameraCenter[2] + 3],
          initialCameraLookAt: cameraCenter,
          sharedMemoryForWorkers: false,
          sphericalHarmonicsDegree: metadata.sh_degree,
          ignoreDevicePixelRatio: true,
          integerBasedSort: false,
          renderMode: GaussianSplats3D.RenderMode.OnChange,
          threeScene: auxiliaryScene
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
        sceneFrameRef.current = getSceneFrame(viewer, metadata.scene_center, metadata.scene_radius_p95, cameraFrame);
        configureControls(viewer, sceneFrameRef.current);
        applyViewPreset(viewer, sceneFrameRef.current, "fit");
        viewer.start();
        setViewerState("ready");

        if (navigationUrl && collisionMeshUrl) {
          try {
            const runtime = await loadWalkRuntime(
              navigationUrl,
              collisionMeshUrl,
              auxiliaryScene,
              controller.signal
            );
            if (cancelled) {
              disposeWalkRuntime(runtime);
              return;
            }
            walkRuntimeRef.current = runtime;
            runtime.debugRoot.visible = debugVisibleRef.current;
            const nextSettings = parseWalkSettings(localStorage.getItem(WALK_SETTINGS_KEY), runtime.contract);
            settingsRef.current = nextSettings;
            setSettings(nextSettings);
            installWalkHandlers(
              viewer,
              runtime,
              viewerModeRef,
              settingsRef,
              setViewerMode,
              setWalkMessage,
              setBoundaryHint
            );
            setNavigationState("ready");
            setWalkMessage("Walk ready · Enter Walk to lock the pointer");
          } catch (error) {
            if (!cancelled) {
              setNavigationState("error");
              setWalkMessage(error instanceof Error ? error.message : "Navigation assets failed to load");
            }
          }
        }
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
      if (document.pointerLockElement === viewer?.renderer?.domElement) {
        document.exitPointerLock();
      }
      const runtime = walkRuntimeRef.current;
      walkRuntimeRef.current = null;
      if (runtime) {
        uninstallWalkHandlers(runtime);
        disposeWalkRuntime(runtime);
      }
      const activeViewer = viewerRef.current;
      viewerRef.current = null;
      if (activeViewer) {
        activeViewer.stop();
        void activeViewer.dispose().catch(() => {
          // The viewer owns its internal DOM and may already have removed it.
        });
      }
      auxiliaryScene?.clear();
      mount.replaceChildren();
    };
  }, [sourceUrl, metadataUrl, cameraPathUrl, collisionMeshUrl, navigationUrl, navigationStatus]);

  const walkReady = viewerState === "ready" && navigationState === "ready";
  const unavailableMessage =
    navigationStatus === "available" && navigationState === "error"
      ? walkMessage
      : navigationReason
        ? `Walk unavailable: ${navigationReason.replaceAll("_", " ")}`
        : navigationStatus && navigationStatus !== "available"
          ? `Walk ${navigationStatus.replaceAll("_", " ")}`
          : "Walk assets not generated";

  return (
    <div className="viewer-surface">
      <div className="splat-root" ref={mountRef} />
      {sourceUrl && (
        <div className="splat-toolbar" aria-label="Gaussian splat controls">
          {viewerMode === "orbit" &&
            (Object.keys(VIEW_PRESETS) as ViewPreset[]).map((preset) => (
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
          <button
            className={viewerMode === "walk" ? "viewer-tool-button active" : "viewer-tool-button"}
            disabled={!walkReady}
            onClick={viewerMode === "walk" ? exitWalk : enterWalk}
            type="button"
          >
            {viewerMode === "walk" ? "Exit Walk" : "Enter Walk"}
          </button>
          <button className="viewer-tool-button" disabled={!walkReady} onClick={resetWalk} type="button">
            Reset Walk
          </button>
          <button
            className={debugVisible ? "viewer-tool-button active" : "viewer-tool-button"}
            disabled={!walkReady}
            onClick={() => setDebugVisible((current) => !current)}
            type="button"
          >
            Debug
          </button>
        </div>
      )}
      {settings && walkReady && (
        <div className="walk-settings" aria-label="Walk settings">
          <label>
            <span>Speed {settings.speedHeightRatio.toFixed(1)}H/s</span>
            <input
              min={0.4}
              max={1.2}
              step={0.1}
              type="range"
              value={settings.speedHeightRatio}
              onChange={(event) => updateSettings({ ...settings, speedHeightRatio: Number(event.target.value) })}
            />
          </label>
          <label>
            <span>FOV {settings.fovDegrees.toFixed(0)}°</span>
            <input
              min={50}
              max={90}
              step={1}
              type="range"
              value={settings.fovDegrees}
              onChange={(event) => updateSettings({ ...settings, fovDegrees: Number(event.target.value) })}
            />
          </label>
          <label>
            <span>Sensitivity {settings.sensitivity.toFixed(4)}</span>
            <input
              min={0.0005}
              max={0.005}
              step={0.0005}
              type="range"
              value={settings.sensitivity}
              onChange={(event) => updateSettings({ ...settings, sensitivity: Number(event.target.value) })}
            />
          </label>
          <button
            className="viewer-tool-button"
            type="button"
            onClick={() => updateSettings(defaultWalkSettings(walkRuntimeRef.current!.contract))}
          >
            Defaults
          </button>
        </div>
      )}
      {navigationState === "ready" && viewerMode === "orbit" && (
        <div className="walk-ready-hint">{walkMessage}</div>
      )}
      {boundaryHint && <div className="boundary-hint">Navigation limit reached</div>}
      <div className="sr-only" aria-live="polite">
        {walkMessage}
      </div>
      {viewerState === "ready" && (
        <div className="viewer-hint">
          {viewerMode === "walk"
            ? "Walk · WASD/arrows move · Mouse look · Esc exits"
            : "Canonical normalized coordinates · arbitrary units"}
          {assetBytes === null ? "" : ` · ${(assetBytes / 1_048_576).toFixed(1)} MiB`}
          {viewerMode === "orbit" ? " · Drag orbit · Shift/right drag pan · Wheel zoom" : ""}
          {navigationState !== "ready" ? ` · ${unavailableMessage}` : ""}
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

async function loadWalkRuntime(
  navigationUrl: string,
  collisionMeshUrl: string,
  scene: THREE.Scene,
  signal: AbortSignal
): Promise<WalkRuntime> {
  const [navigationResponse, collisionResponse] = await Promise.all([
    fetch(navigationUrl, { signal }),
    fetch(collisionMeshUrl, { signal })
  ]);
  if (!navigationResponse.ok) {
    throw new Error(`Navigation contract request failed: ${navigationResponse.status}`);
  }
  if (!collisionResponse.ok) {
    throw new Error(`Collision mesh request failed: ${collisionResponse.status}`);
  }

  const contract = parseNavigationContract(await navigationResponse.json());
  const collisionBuffer = await collisionResponse.arrayBuffer();
  if (collisionBuffer.byteLength !== contract.collision.bytes) {
    throw new Error("Collision mesh size does not match the navigation contract");
  }
  const collisionHash = await crypto.subtle.digest("SHA-256", collisionBuffer);
  if (hexDigest(collisionHash) !== contract.collision.sha256) {
    throw new Error("Collision mesh hash does not match the navigation contract");
  }

  const gltf = await new GLTFLoader().parseAsync(collisionBuffer, collisionMeshUrl.slice(0, collisionMeshUrl.lastIndexOf("/") + 1));
  const collisionRoot = gltf.scene;
  let triangles = 0;
  collisionRoot.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) {
      return;
    }
    const positions = object.geometry.getAttribute("position");
    triangles += object.geometry.index ? object.geometry.index.count / 3 : positions.count / 3;
    disposeMaterial(object.material);
    object.material = new THREE.MeshBasicMaterial({
      color: 0xe06b3c,
      depthWrite: false,
      opacity: 0.4,
      transparent: true,
      wireframe: true
    });
  });
  if (triangles !== contract.collision.triangles) {
    disposeObject(collisionRoot);
    throw new Error("Collision triangle count does not match the navigation contract");
  }

  const octree = new Octree().fromGraphNode(collisionRoot);
  const up = new THREE.Vector3(...contract.up).normalize();
  const basisU = new THREE.Vector3(...contract.floor.basis_u).normalize();
  const basisV = new THREE.Vector3(...contract.floor.basis_v).normalize();
  const baseForward = new THREE.Vector3(...contract.spawn.look_direction)
    .addScaledVector(up, -new THREE.Vector3(...contract.spawn.look_direction).dot(up))
    .normalize();
  if (!Number.isFinite(baseForward.x) || baseForward.lengthSq() < 0.99) {
    disposeObject(collisionRoot);
    throw new Error("Navigation spawn direction is invalid");
  }
  const baseRight = new THREE.Vector3().crossVectors(baseForward, up).normalize();
  const floorPosition = new THREE.Vector3(...contract.spawn.floor_position);
  const radius = contract.player.capsule_radius;
  const capsule = new Capsule(
    floorPosition.clone().addScaledVector(up, radius),
    floorPosition.clone().addScaledVector(up, contract.player.capsule_total_height - radius),
    radius
  );
  if (octree.capsuleIntersect(capsule)) {
    disposeObject(collisionRoot);
    throw new Error("Navigation spawn intersects the collision mesh");
  }

  const debugRoot = new THREE.Group();
  debugRoot.name = "navigation-debug";
  debugRoot.add(collisionRoot);
  addBoundaryDebug(debugRoot, contract);
  const capsuleHelper = new THREE.Mesh(
    new THREE.CapsuleGeometry(radius, Math.max(contract.player.capsule_total_height - 2 * radius, 0.001), 8, 16),
    new THREE.MeshBasicMaterial({ color: 0x247f89, wireframe: true })
  );
  debugRoot.add(capsuleHelper);
  const spawnHelper = new THREE.AxesHelper(contract.estimated_eye_height * 0.35);
  spawnHelper.position.copy(new THREE.Vector3(...contract.spawn.eye_position));
  debugRoot.add(spawnHelper);
  const contactHelper = new THREE.ArrowHelper(up, floorPosition, contract.estimated_eye_height * 0.3, 0xc74440);
  contactHelper.visible = false;
  debugRoot.add(contactHelper);
  scene.add(debugRoot);

  const runtime: WalkRuntime = {
    contract,
    octree,
    capsule,
    spawnCapsule: capsule.clone(),
    safeCapsule: capsule.clone(),
    up,
    basisU,
    basisV,
    baseForward,
    baseRight,
    yaw: 0,
    pitch: 0,
    keys: new Set(),
    debugRoot,
    collisionRoot,
    capsuleHelper,
    contactHelper,
    orbitControls: null,
    animationFrame: null,
    previousTime: null,
    accumulator: 0,
    boundaryHintTimer: null
  };
  updateDebugHelpers(runtime);
  return runtime;
}

function installWalkHandlers(
  viewer: GaussianSplats3D.Viewer,
  runtime: WalkRuntime,
  modeRef: { current: ViewerMode },
  activeSettingsRef: { current: WalkSettings | null },
  setViewerMode: (mode: ViewerMode) => void,
  setWalkMessage: (message: string) => void,
  setBoundaryHint: (visible: boolean) => void
) {
  const canvas = viewer.renderer?.domElement;
  if (!canvas) {
    throw new Error("Gaussian renderer does not expose a Pointer Lock canvas");
  }

  const clearKeys = () => runtime.keys.clear();
  const keyChange = (event: KeyboardEvent, pressed: boolean) => {
    if (viewerModeRefValue() !== "walk" || !isMovementKey(event.code)) {
      return;
    }
    event.preventDefault();
    if (pressed) {
      runtime.keys.add(event.code);
    } else {
      runtime.keys.delete(event.code);
    }
  };
  const onKeyDown = (event: KeyboardEvent) => keyChange(event, true);
  const onKeyUp = (event: KeyboardEvent) => keyChange(event, false);
  const onMouseMove = (event: MouseEvent) => {
    if (document.pointerLockElement !== canvas) {
      return;
    }
    const settings = settingsRefValue();
    runtime.yaw += event.movementX * settings.sensitivity;
    runtime.pitch = clampPitchRadians(
      runtime.pitch - event.movementY * settings.sensitivity,
      runtime.contract.controls.pitch_degrees.minimum,
      runtime.contract.controls.pitch_degrees.maximum
    );
    applyWalkCamera(viewer, runtime, settings);
    viewer.forceRenderNextFrame?.();
  };
  const onPointerLockChange = () => {
    if (document.pointerLockElement === canvas) {
      modeRef.current = "walk";
      setViewerMode("walk");
      const controls = viewer.controls;
      if (controls) {
        controls.enabled = false;
        controls.enableDamping = false;
        runtime.orbitControls = controls;
        viewer.controls = null;
      }
      clearKeys();
      runtime.previousTime = null;
      runtime.accumulator = 0;
      applyWalkCamera(viewer, runtime, settingsRefValue());
      setWalkMessage("Walk active · WASD or arrows move · Esc exits");
      runtime.animationFrame = window.requestAnimationFrame(tick);
    } else {
      leaveWalkMode(viewer, runtime, modeRef, setViewerMode, setWalkMessage);
    }
  };
  const onPointerLockError = () => setWalkMessage("Pointer Lock request was denied");
  const onVisibilityChange = () => {
    if (document.hidden) {
      clearKeys();
    }
  };
  const tick = (time: number) => {
    if (document.pointerLockElement !== canvas) {
      runtime.animationFrame = null;
      return;
    }
    const frameSeconds = runtime.previousTime === null ? 0 : Math.min((time - runtime.previousTime) / 1000, MAX_FRAME_SECONDS);
    runtime.previousTime = time;
    runtime.accumulator += frameSeconds;
    let moved = false;
    while (runtime.accumulator >= FIXED_STEP_SECONDS) {
      moved = stepWalk(runtime, settingsRefValue(), FIXED_STEP_SECONDS, setBoundaryHint) || moved;
      runtime.accumulator -= FIXED_STEP_SECONDS;
    }
    if (moved) {
      applyWalkCamera(viewer, runtime, settingsRefValue());
      updateDebugHelpers(runtime);
      viewer.forceRenderNextFrame?.();
    }
    runtime.animationFrame = window.requestAnimationFrame(tick);
  };

  runtimeHandlers.set(runtime, {
    onKeyDown,
    onKeyUp,
    onMouseMove,
    onPointerLockChange,
    onPointerLockError,
    clearKeys,
    onVisibilityChange
  });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  document.addEventListener("mousemove", onMouseMove);
  window.addEventListener("blur", clearKeys);
  document.addEventListener("visibilitychange", onVisibilityChange);
  document.addEventListener("pointerlockchange", onPointerLockChange);
  document.addEventListener("pointerlockerror", onPointerLockError);

  function viewerModeRefValue() {
    return modeRef.current;
  }
  function settingsRefValue() {
    return activeSettingsRef.current ?? defaultWalkSettings(runtime.contract);
  }
}

const runtimeHandlers = new WeakMap<
  WalkRuntime,
  {
    onKeyDown: (event: KeyboardEvent) => void;
    onKeyUp: (event: KeyboardEvent) => void;
    onMouseMove: (event: MouseEvent) => void;
    onPointerLockChange: () => void;
    onPointerLockError: () => void;
    clearKeys: () => void;
    onVisibilityChange: () => void;
  }
>();

function uninstallWalkHandlers(runtime: WalkRuntime) {
  const handlers = runtimeHandlers.get(runtime);
  if (!handlers) {
    return;
  }
  window.removeEventListener("keydown", handlers.onKeyDown);
  window.removeEventListener("keyup", handlers.onKeyUp);
  document.removeEventListener("mousemove", handlers.onMouseMove);
  window.removeEventListener("blur", handlers.clearKeys);
  document.removeEventListener("visibilitychange", handlers.onVisibilityChange);
  document.removeEventListener("pointerlockchange", handlers.onPointerLockChange);
  document.removeEventListener("pointerlockerror", handlers.onPointerLockError);
  runtimeHandlers.delete(runtime);
}

function leaveWalkMode(
  viewer: GaussianSplats3D.Viewer | null,
  runtime: WalkRuntime | null,
  modeRef: { current: ViewerMode },
  setViewerMode: (mode: ViewerMode) => void,
  setWalkMessage: (message: string) => void
) {
  modeRef.current = "orbit";
  setViewerMode("orbit");
  if (!viewer || !runtime) {
    return;
  }
  runtime.keys.clear();
  if (runtime.animationFrame !== null) {
    window.cancelAnimationFrame(runtime.animationFrame);
    runtime.animationFrame = null;
  }
  const controls = runtime.orbitControls ?? viewer.controls;
  if (controls) {
    viewer.controls = controls;
    controls.enabled = true;
    controls.target.copy(viewer.camera.position).addScaledVector(walkDirection(runtime), runtime.contract.estimated_eye_height);
    controls.update();
    controls.enableDamping = true;
    controls.saveState();
  }
  viewer.forceRenderNextFrame?.();
  setWalkMessage("Orbit mode · Walk position preserved");
}

function stepWalk(
  runtime: WalkRuntime,
  settings: WalkSettings,
  deltaSeconds: number,
  setBoundaryHint: (visible: boolean) => void
): boolean {
  const forwardInput = Number(runtime.keys.has("KeyW") || runtime.keys.has("ArrowUp")) - Number(runtime.keys.has("KeyS") || runtime.keys.has("ArrowDown"));
  const strafeInput = Number(runtime.keys.has("KeyD") || runtime.keys.has("ArrowRight")) - Number(runtime.keys.has("KeyA") || runtime.keys.has("ArrowLeft"));
  if (forwardInput === 0 && strafeInput === 0) {
    return false;
  }

  const direction = walkDirection(runtime);
  const speed = settings.speedHeightRatio * runtime.contract.estimated_eye_height;
  const velocityValues = planarWalkVelocity(
    [direction.x, direction.y, direction.z],
    [runtime.up.x, runtime.up.y, runtime.up.z],
    forwardInput,
    strafeInput,
    speed
  );
  const displacement = new THREE.Vector3(...velocityValues).multiplyScalar(deltaSeconds);
  runtime.capsule.translate(displacement);

  runtime.contactHelper.visible = false;
  for (let iteration = 0; iteration < 4; iteration += 1) {
    const collision = runtime.octree.capsuleIntersect(runtime.capsule);
    if (!collision) {
      break;
    }
    runtime.capsule.translate(collision.normal.clone().multiplyScalar(collision.depth));
    runtime.contactHelper.position.copy(runtime.capsule.start);
    runtime.contactHelper.setDirection(collision.normal);
    runtime.contactHelper.visible = true;
  }

  const floorPoint = runtime.capsule.start.clone().addScaledVector(runtime.up, -runtime.capsule.radius);
  const relative = floorPoint.sub(new THREE.Vector3(...runtime.contract.floor.origin));
  const uv: [number, number] = [relative.dot(runtime.basisU), relative.dot(runtime.basisV)];
  const finite = [...runtime.capsule.start.toArray(), ...runtime.capsule.end.toArray()].every(Number.isFinite);
  if (!finite || !pointInNavigationBoundary(uv, runtime.contract.boundary)) {
    runtime.capsule.copy(runtime.safeCapsule);
    showBoundaryHint(runtime, setBoundaryHint);
    return true;
  }

  runtime.safeCapsule.copy(runtime.capsule);
  return true;
}

function showBoundaryHint(runtime: WalkRuntime, setBoundaryHint: (visible: boolean) => void) {
  setBoundaryHint(true);
  if (runtime.boundaryHintTimer !== null) {
    window.clearTimeout(runtime.boundaryHintTimer);
  }
  runtime.boundaryHintTimer = window.setTimeout(() => {
    setBoundaryHint(false);
    runtime.boundaryHintTimer = null;
  }, 1200);
}

function applyWalkCamera(viewer: GaussianSplats3D.Viewer, runtime: WalkRuntime, settings: WalkSettings | null) {
  if (!(viewer.camera instanceof THREE.PerspectiveCamera)) {
    return;
  }
  const activeSettings = settings ?? defaultWalkSettings(runtime.contract);
  const eye = runtime.capsule.start
    .clone()
    .addScaledVector(runtime.up, runtime.contract.estimated_eye_height - runtime.capsule.radius);
  const direction = walkDirection(runtime);
  viewer.camera.position.copy(eye);
  viewer.camera.up.copy(runtime.up);
  viewer.camera.lookAt(eye.clone().add(direction));
  viewer.camera.fov = activeSettings.fovDegrees;
  viewer.camera.near = Math.max(runtime.contract.player.capsule_radius * 0.05, 0.001);
  viewer.camera.updateProjectionMatrix();
  if (viewer.controls) {
    viewer.controls.target
      .copy(eye)
      .addScaledVector(direction, runtime.contract.estimated_eye_height);
    viewer.controls.update();
  }
}

function walkDirection(runtime: WalkRuntime): THREE.Vector3 {
  const horizontal = runtime.baseForward
    .clone()
    .multiplyScalar(Math.cos(runtime.yaw))
    .addScaledVector(runtime.baseRight, Math.sin(runtime.yaw))
    .normalize();
  return horizontal.multiplyScalar(Math.cos(runtime.pitch)).addScaledVector(runtime.up, Math.sin(runtime.pitch)).normalize();
}

function updateDebugHelpers(runtime: WalkRuntime) {
  const center = runtime.capsule.getCenter(new THREE.Vector3());
  runtime.capsuleHelper.position.copy(center);
  runtime.capsuleHelper.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), runtime.up);
}

function addBoundaryDebug(root: THREE.Group, contract: NavigationContract) {
  const origin = new THREE.Vector3(...contract.floor.origin).addScaledVector(new THREE.Vector3(...contract.up), 0.003);
  const basisU = new THREE.Vector3(...contract.floor.basis_u);
  const basisV = new THREE.Vector3(...contract.floor.basis_v);
  for (const polygon of [contract.boundary.outer, ...contract.boundary.holes]) {
    const points = polygon.map(([u, v]) => origin.clone().addScaledVector(basisU, u).addScaledVector(basisV, v));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    root.add(new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color: 0x2a83b8 })));
  }
}

function isMovementKey(code: string) {
  return ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(code);
}

function hexDigest(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function disposeWalkRuntime(runtime: WalkRuntime) {
  if (runtime.animationFrame !== null) {
    window.cancelAnimationFrame(runtime.animationFrame);
  }
  if (runtime.boundaryHintTimer !== null) {
    window.clearTimeout(runtime.boundaryHintTimer);
  }
  runtime.octree.clear();
  runtime.debugRoot.removeFromParent();
  disposeObject(runtime.debugRoot);
}

function disposeObject(root: THREE.Object3D) {
  root.traverse((object) => {
    if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.LineLoop) {
      object.geometry.dispose();
      disposeMaterial(object.material);
    }
  });
}

function disposeMaterial(material: THREE.Material | THREE.Material[]) {
  if (Array.isArray(material)) {
    material.forEach((item) => item.dispose());
  } else {
    material.dispose();
  }
}
