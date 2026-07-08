import { GaussianSplatViewer } from "./GaussianSplatViewer";
import { MeshViewer } from "./MeshViewer";
import { PointCloudViewer } from "./PointCloudViewer";

type GeometryViewerProps = {
  pointCloudUrl: string | null;
  meshUrl: string | null;
  splatUrl: string | null;
};

export function GeometryViewer({ pointCloudUrl, meshUrl, splatUrl }: GeometryViewerProps) {
  if (splatUrl) {
    return <GaussianSplatViewer sourceUrl={splatUrl} />;
  }
  if (meshUrl) {
    return <MeshViewer sourceUrl={meshUrl} />;
  }
  return <PointCloudViewer sourceUrl={pointCloudUrl} />;
}
