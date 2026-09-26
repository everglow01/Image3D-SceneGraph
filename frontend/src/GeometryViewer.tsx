import { useEffect, useRef, useState } from "react";
import type { CameraView } from "./cloudGaussianEditor";
import { CloudGaussianViewer, type CloudSource } from "./CloudGaussianViewer";
import { GaussianSplatViewer } from "./GaussianSplatViewer";
import { MeshViewer } from "./MeshViewer";
import { PointCloudViewer } from "./PointCloudViewer";
import type { SfmInspectionTab } from "./sfmDiagnostics";
import { leaveGaussianViewer, type ViewerLeaveRef } from "./gaussianViewerLeave";

type GeometryViewerProps = {
  cloudSource?: CloudSource;
  onCloudModeChange?: (cloud: boolean) => void;
  onLocalModeChange?: (editing: boolean) => void;
  leaveRef?: ViewerLeaveRef;
  pointCloudUrl: string | null;
  camerasUrl: string | null;
  alignmentDiagnosticsUrl: string | null;
  pointCloudVariant: "raw" | "aligned";
  meshUrl: string | null;
  splatUrl: string | null;
  browserSplatUrl?: string | null;
  splatMetadataUrl: string | null;
  splatCameraPathUrl: string | null;
  jobId: string | null;
  sfmDiagnosticsUrl: string | null;
  inspectionRequest: { id: number; tab: SfmInspectionTab } | null;
  onInspectionStateChange: (tab: SfmInspectionTab | null) => void;
  collisionMeshUrl: string | null;
  navigationUrl: string | null;
  navigationStatus: string | null;
  navigationReason: string | null;
};

export function GeometryViewer({
  cloudSource,
  onCloudModeChange,
  onLocalModeChange,
  leaveRef,
  pointCloudUrl,
  camerasUrl,
  alignmentDiagnosticsUrl,
  pointCloudVariant,
  meshUrl,
  splatUrl,
  browserSplatUrl,
  splatMetadataUrl,
  splatCameraPathUrl,
  jobId,
  sfmDiagnosticsUrl,
  inspectionRequest,
  onInspectionStateChange,
  collisionMeshUrl,
  navigationUrl,
  navigationStatus,
  navigationReason
}: GeometryViewerProps) {
  const [cloud, setCloud] = useState(false);
  const viewRef = useRef<CameraView | null>(null);
  const modelRelease = useRef<Promise<void>>(Promise.resolve());
  const ownGate = useRef<(() => Promise<boolean>) | null>(null);
  const gate = leaveRef ?? ownGate;
  const switching = useRef(false);
  async function switchCloud(next: boolean) {
    if (next === cloud || switching.current) return;
    switching.current = true;
    try {
      if (!await leaveGaussianViewer(gate)) return;
      onInspectionStateChange(null); setCloud(next);
    } finally { switching.current = false; }
  }
  useEffect(() => {
    onCloudModeChange?.(!!(cloud && splatUrl && cloudSource));
    return () => onCloudModeChange?.(false);
  }, [cloud, splatUrl, !!cloudSource, onCloudModeChange]);
  if (splatUrl) {
    return <>
      {cloudSource && <div className="variant-toggle cloud-mode-toggle" role="group" aria-label="渲染位置">
        <button type="button" aria-pressed={!cloud} className={!cloud ? "active" : ""} onClick={() => void switchCloud(false)}>本地查看</button>
        <button type="button" aria-pressed={cloud} className={cloud ? "active" : ""} onClick={() => void switchCloud(true)}>云端查看与修剪</button>
      </div>}
      {cloud && cloudSource ? <CloudGaussianViewer key={splatUrl} leaveRef={gate} viewRef={viewRef} viewKey={splatUrl} source={cloudSource} metadataUrl={splatMetadataUrl} cameraPathUrl={splatCameraPathUrl} alignmentUrl={alignmentDiagnosticsUrl} /> : <GaussianSplatViewer
        editSource={cloudSource}
        releaseRef={modelRelease}
        leaveRef={gate}
        onLocalModeChange={onLocalModeChange}
        viewRef={viewRef}
        sourceUrl={splatUrl}
        browserSourceUrl={browserSplatUrl}
        metadataUrl={splatMetadataUrl}
        cameraPathUrl={splatCameraPathUrl}
        alignmentUrl={alignmentDiagnosticsUrl}
        jobId={jobId}
        sfmDiagnosticsUrl={sfmDiagnosticsUrl}
        inspectionRequest={inspectionRequest}
        onInspectionStateChange={onInspectionStateChange}
        collisionMeshUrl={collisionMeshUrl}
        navigationUrl={navigationUrl}
        navigationStatus={navigationStatus}
        navigationReason={navigationReason}
      />}
    </>;
  }
  if (meshUrl) {
    return <MeshViewer sourceUrl={meshUrl} />;
  }
  return (
    <PointCloudViewer
      sourceUrl={pointCloudUrl}
      camerasUrl={camerasUrl}
      alignmentDiagnosticsUrl={alignmentDiagnosticsUrl}
      pointCloudVariant={pointCloudVariant}
    />
  );
}
