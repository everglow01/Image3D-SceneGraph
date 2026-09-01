import { GaussianSplatViewer } from "./GaussianSplatViewer";
import { MeshViewer } from "./MeshViewer";
import { PointCloudViewer } from "./PointCloudViewer";
import type { SfmInspectionTab } from "./sfmDiagnostics";

type GeometryViewerProps = {
  pointCloudUrl: string | null;
  camerasUrl: string | null;
  alignmentDiagnosticsUrl: string | null;
  pointCloudVariant: "raw" | "aligned";
  meshUrl: string | null;
  splatUrl: string | null;
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
  pointCloudUrl,
  camerasUrl,
  alignmentDiagnosticsUrl,
  pointCloudVariant,
  meshUrl,
  splatUrl,
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
  if (splatUrl) {
    return (
      <GaussianSplatViewer
        sourceUrl={splatUrl}
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
      />
    );
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
