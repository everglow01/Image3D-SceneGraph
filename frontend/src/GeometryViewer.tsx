import { GaussianSplatViewer } from "./GaussianSplatViewer";
import { PointCloudViewer } from "./PointCloudViewer";

type GeometryViewerProps = {
  pointCloudUrl: string | null;
  splatUrl: string | null;
};

export function GeometryViewer({ pointCloudUrl, splatUrl }: GeometryViewerProps) {
  if (splatUrl) {
    return <GaussianSplatViewer sourceUrl={splatUrl} />;
  }
  return <PointCloudViewer sourceUrl={pointCloudUrl} />;
}
