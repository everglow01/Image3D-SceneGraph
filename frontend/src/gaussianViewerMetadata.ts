export type GaussianExportMetadata = {
  sh_degree: number;
  viewer_minimum_opacity: number;
  scene_center: [number, number, number] | null;
  scene_radius_p95: number | null;
};

export function parseContentLength(value: string | null): number | null {
  if (value === null || value.trim() === "") {
    return null;
  }
  const bytes = Number(value);
  return Number.isFinite(bytes) && bytes >= 0 ? bytes : null;
}

export function parseGaussianExportMetadata(value: unknown): GaussianExportMetadata {
  if (!value || typeof value !== "object") {
    throw new Error("Gaussian export metadata must be an object");
  }
  const record = value as Record<string, unknown>;
  const shDegree = record.sh_degree;
  const minimumOpacity = record.viewer_minimum_opacity ?? 0.005;
  const sceneCenter = record.scene_center ?? null;
  const sceneRadius = record.scene_radius_p95 ?? null;
  if (!Number.isInteger(shDegree) || Number(shDegree) < 0 || Number(shDegree) > 3) {
    throw new Error("Gaussian export SH degree is invalid");
  }
  if (
    typeof minimumOpacity !== "number" ||
    !Number.isFinite(minimumOpacity) ||
    minimumOpacity < 0 ||
    minimumOpacity >= 1
  ) {
    throw new Error("Gaussian export opacity threshold is invalid");
  }
  if (
    sceneCenter !== null &&
    (!Array.isArray(sceneCenter) ||
      sceneCenter.length !== 3 ||
      sceneCenter.some((coordinate) => typeof coordinate !== "number" || !Number.isFinite(coordinate)))
  ) {
    throw new Error("Gaussian export scene center is invalid");
  }
  if (
    sceneRadius !== null &&
    (typeof sceneRadius !== "number" || !Number.isFinite(sceneRadius) || sceneRadius <= 0)
  ) {
    throw new Error("Gaussian export scene radius is invalid");
  }
  return {
    sh_degree: Number(shDegree),
    viewer_minimum_opacity: minimumOpacity,
    scene_center: sceneCenter === null ? null : [sceneCenter[0], sceneCenter[1], sceneCenter[2]],
    scene_radius_p95: sceneRadius
  };
}

export function viewerAlphaThreshold(minimumOpacity: number): number {
  return Math.floor(minimumOpacity * 255);
}
