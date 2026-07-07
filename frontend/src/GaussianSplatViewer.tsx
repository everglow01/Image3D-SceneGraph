import { useEffect, useRef, useState } from "react";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

type GaussianSplatViewerProps = {
  sourceUrl: string | null;
};

export function GaussianSplatViewer({ sourceUrl }: GaussianSplatViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<GaussianSplats3D.Viewer | null>(null);
  const [viewerState, setViewerState] = useState("idle");

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !sourceUrl) {
      setViewerState("idle");
      return;
    }

    let cancelled = false;
    setViewerState("loading");
    mount.replaceChildren();

    const viewer = new GaussianSplats3D.Viewer({
      rootElement: mount,
      cameraUp: [0, 1, 0],
      initialCameraPosition: [0, 1.2, 3],
      initialCameraLookAt: [0, 0, 0],
      sharedMemoryForWorkers: false,
      sphericalHarmonicsDegree: 0,
      ignoreDevicePixelRatio: true,
      integerBasedSort: false
    });
    viewerRef.current = viewer;

    viewer
      .addSplatScene(sourceUrl, {
        showLoadingUI: true,
        progressiveLoad: true,
        splatAlphaRemovalThreshold: 5
      })
      .then(() => {
        if (cancelled) {
          return;
        }
        viewer.start();
        setViewerState("ready");
      })
      .catch(() => {
        if (!cancelled) {
          setViewerState("error");
        }
      });

    return () => {
      cancelled = true;
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
  }, [sourceUrl]);

  return (
    <div className="viewer-surface">
      <div className="splat-root" ref={mountRef} />
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
