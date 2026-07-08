import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

type PointCloudViewerProps = {
  sourceUrl: string | null;
};

type AxisSigns = {
  x: 1 | -1;
  y: 1 | -1;
  z: 1 | -1;
};

export function PointCloudViewer({ sourceUrl }: PointCloudViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const [viewerState, setViewerState] = useState("idle");
  const [axisSigns, setAxisSigns] = useState<AxisSigns>({ x: 1, y: -1, z: -1 });
  const [pointSize, setPointSize] = useState(0.035);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f7f8);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
    camera.position.set(1.8, 1.5, 2.4);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controlsRef.current = controls;

    const grid = new THREE.GridHelper(2, 10, 0x9aa3a7, 0xd2d8dc);
    grid.position.y = -0.6;
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
        geometry.computeBoundingSphere();
        geometry.center();

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

        const radius = geometry.boundingSphere?.radius ?? 1;
        camera.position.set(radius * 1.8, radius * 1.4, radius * 2.2);
        camera.near = Math.max(radius / 100, 0.001);
        camera.far = Math.max(radius * 100, 100);
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
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
  }, [axisSigns]);

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

  return (
    <div className="viewer-surface" ref={containerRef}>
      <div className="pointcloud-toolbar" aria-label="Point cloud coordinate controls">
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
        <label className="point-size-control">
          <span>Point</span>
          <input
            aria-label="Point size"
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
          {viewerState === "idle" && "No point cloud"}
          {viewerState === "loading" && "Loading point cloud"}
          {viewerState === "error" && "Failed to load point cloud"}
        </div>
      )}
    </div>
  );
}

function applyAxisSigns(points: THREE.Points, axisSigns: AxisSigns) {
  points.scale.set(axisSigns.x, axisSigns.y, axisSigns.z);
}
