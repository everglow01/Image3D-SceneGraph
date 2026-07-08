import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

type MeshViewerProps = {
  sourceUrl: string | null;
};

type AxisSigns = {
  x: 1 | -1;
  y: 1 | -1;
  z: 1 | -1;
};

export function MeshViewer({ sourceUrl }: MeshViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRootRef = useRef<THREE.Object3D | null>(null);
  const [viewerState, setViewerState] = useState("idle");
  const [axisSigns, setAxisSigns] = useState<AxisSigns>({ x: 1, y: 1, z: 1 });

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

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controlsRef.current = controls;

    const ambient = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambient);
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(3, -4, 5);
    scene.add(keyLight);

    const grid = new THREE.GridHelper(2, 10, 0x9aa3a7, 0xd2d8dc);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(0.8));

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
      clearMeshRoot(scene, meshRootRef.current);
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          disposeMaterial(object.material);
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

    clearMeshRoot(scene, meshRootRef.current);
    meshRootRef.current = null;

    if (!sourceUrl) {
      setViewerState("idle");
      return;
    }

    let cancelled = false;
    setViewerState("loading");

    const loader = new GLTFLoader();
    loader.load(
      sourceUrl,
      (gltf) => {
        if (cancelled) {
          clearMeshRoot(null, gltf.scene);
          return;
        }

        const root = gltf.scene;
        normalizeMeshMaterials(root);
        applyAxisSigns(root, axisSigns);
        scene.add(root);
        meshRootRef.current = root;

        const frame = getObjectFrame(root);
        camera.position.set(frame.radius * 1.8, -frame.radius * 2.2, frame.radius * 1.4);
        camera.up.set(0, 0, 1);
        camera.near = Math.max(frame.radius / 100, 0.001);
        camera.far = Math.max(frame.radius * 100, 100);
        camera.updateProjectionMatrix();
        controls.target.copy(frame.center);
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
    if (meshRootRef.current) {
      applyAxisSigns(meshRootRef.current, axisSigns);
    }
  }, [axisSigns]);

  function toggleAxis(axis: keyof AxisSigns) {
    setAxisSigns((current) => ({
      ...current,
      [axis]: current[axis] === 1 ? -1 : 1
    }));
  }

  return (
    <div className="viewer-surface" ref={containerRef}>
      <div className="pointcloud-toolbar" aria-label="Mesh coordinate controls">
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
      </div>
      {viewerState === "ready" && <div className="viewer-hint">Drag rotate · Wheel zoom · Right drag pan</div>}
      {viewerState !== "ready" && (
        <div className="viewer-overlay">
          {viewerState === "idle" && "No mesh"}
          {viewerState === "loading" && "Loading mesh"}
          {viewerState === "error" && "Failed to load mesh"}
        </div>
      )}
    </div>
  );
}

function getObjectFrame(root: THREE.Object3D) {
  const box = new THREE.Box3().setFromObject(root);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.length() * 0.5, 0.5);
  return { center, radius };
}

function normalizeMeshMaterials(root: THREE.Object3D) {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) {
      return;
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if ("side" in material) {
        material.side = THREE.DoubleSide;
      }
    }
  });
}

function applyAxisSigns(object: THREE.Object3D, axisSigns: AxisSigns) {
  object.scale.set(axisSigns.x, axisSigns.y, axisSigns.z);
}

function clearMeshRoot(scene: THREE.Scene | null, root: THREE.Object3D | null) {
  if (!root) {
    return;
  }
  scene?.remove(root);
  root.traverse((object) => {
    if (object instanceof THREE.Mesh) {
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
