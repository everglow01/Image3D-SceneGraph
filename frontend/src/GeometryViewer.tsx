import { GaussianSplatViewer } from "./GaussianSplatViewer";
import { MeshViewer } from "./MeshViewer";
import { PointCloudViewer } from "./PointCloudViewer";

type GeometryViewerProps = {
  pointCloudUrl: string | null;
  camerasUrl: string | null;
  alignmentDiagnosticsUrl: string | null;
  pointCloudVariant: "raw" | "aligned";
  meshUrl: string | null;
  splatUrl: string | null;
  splatMetadataUrl: string | null;
  splatCameraPathUrl: string | null;
};

export function GeometryViewer({
  pointCloudUrl,
  camerasUrl,
  alignmentDiagnosticsUrl,
  pointCloudVariant,
  meshUrl,
  splatUrl,
  splatMetadataUrl,
  splatCameraPathUrl
}: GeometryViewerProps) {
  if (splatUrl) {
    return (
      <GaussianSplatViewer
        sourceUrl={splatUrl}
        metadataUrl={splatMetadataUrl}
        cameraPathUrl={splatCameraPathUrl}
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
