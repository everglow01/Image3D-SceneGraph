export type Vec3 = [number, number, number];
export type Mat4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number]
];

export type GaussianExportMetadata = {
  sh_degree: number;
  viewer_minimum_opacity: number;
  scene_center: Vec3 | null;
  scene_radius_p95: number | null;
  world_from_normalized: Mat4 | null;
};

export type GaussianCameraPath = {
  keyframes: Array<{
    center_normalized: Vec3;
    world_from_camera: Mat4;
  }>;
};

export type GaussianViewerFrame = {
  center: Vec3;
  up: Vec3;
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
  const worldFromNormalized = record.world_from_normalized ?? null;
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
  if (worldFromNormalized !== null && !isMat4(worldFromNormalized)) {
    throw new Error("Gaussian export world transform is invalid");
  }
  return {
    sh_degree: Number(shDegree),
    viewer_minimum_opacity: minimumOpacity,
    scene_center: sceneCenter === null ? null : [sceneCenter[0], sceneCenter[1], sceneCenter[2]],
    scene_radius_p95: sceneRadius,
    world_from_normalized: worldFromNormalized
  };
}

export function parseGaussianCameraPath(value: unknown): GaussianCameraPath {
  if (!value || typeof value !== "object") {
    throw new Error("Gaussian camera path must be an object");
  }
  const record = value as Record<string, unknown>;
  const keyframes = record.keyframes;
  if (record.schema_version !== 1 || record.coordinate_frame !== "normalized") {
    throw new Error("Gaussian camera path schema is invalid");
  }
  if (!Array.isArray(keyframes) || keyframes.length < 2) {
    throw new Error("Gaussian camera path requires at least two keyframes");
  }
  return {
    keyframes: keyframes.map((value) => {
      if (!value || typeof value !== "object") {
        throw new Error("Gaussian camera keyframe must be an object");
      }
      const keyframe = value as Record<string, unknown>;
      if (!isVec3(keyframe.center_normalized) || !isMat4(keyframe.world_from_camera)) {
        throw new Error("Gaussian camera keyframe is invalid");
      }
      return {
        center_normalized: keyframe.center_normalized,
        world_from_camera: keyframe.world_from_camera
      };
    })
  };
}

export function deriveGaussianViewerFrame(
  metadata: GaussianExportMetadata,
  cameraPath: GaussianCameraPath
): GaussianViewerFrame | null {
  if (!metadata.world_from_normalized) {
    return null;
  }
  const normalizedFromWorld = invertAffine(metadata.world_from_normalized);
  if (!normalizedFromWorld) {
    return null;
  }
  const center: Vec3 = [0, 0, 0];
  const up: Vec3 = [0, 0, 0];
  for (const keyframe of cameraPath.keyframes) {
    const cameraUp: Vec3 = [0, 0, 0];
    for (let axis = 0; axis < 3; axis += 1) {
      center[axis] += keyframe.center_normalized[axis];
      cameraUp[axis] = -(
        normalizedFromWorld[axis][0] * keyframe.world_from_camera[0][1] +
        normalizedFromWorld[axis][1] * keyframe.world_from_camera[1][1] +
        normalizedFromWorld[axis][2] * keyframe.world_from_camera[2][1]
      );
    }
    const cameraUpLength = Math.hypot(...cameraUp);
    if (!Number.isFinite(cameraUpLength) || cameraUpLength < 1e-6) {
      return null;
    }
    for (let axis = 0; axis < 3; axis += 1) {
      up[axis] += cameraUp[axis] / cameraUpLength;
    }
  }
  for (let axis = 0; axis < 3; axis += 1) {
    center[axis] /= cameraPath.keyframes.length;
  }
  const length = Math.hypot(...up);
  if (!Number.isFinite(length) || length < 1e-6) {
    return null;
  }
  return { center, up: [up[0] / length, up[1] / length, up[2] / length] };
}

function isVec3(value: unknown): value is Vec3 {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate))
  );
}

function isMat4(value: unknown): value is Mat4 {
  return Array.isArray(value) && value.length === 4 && value.every((row) => Array.isArray(row) && row.length === 4 && row.every((entry) => typeof entry === "number" && Number.isFinite(entry)));
}

function invertAffine(matrix: Mat4): Mat4 | null {
  const a = matrix[0][0];
  const b = matrix[0][1];
  const c = matrix[0][2];
  const d = matrix[1][0];
  const e = matrix[1][1];
  const f = matrix[1][2];
  const g = matrix[2][0];
  const h = matrix[2][1];
  const i = matrix[2][2];
  const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) {
    return null;
  }
  const inverse = 1 / determinant;
  const rotation = [
    [(e * i - f * h) * inverse, (c * h - b * i) * inverse, (b * f - c * e) * inverse],
    [(f * g - d * i) * inverse, (a * i - c * g) * inverse, (c * d - a * f) * inverse],
    [(d * h - e * g) * inverse, (b * g - a * h) * inverse, (a * e - b * d) * inverse]
  ];
  const translation = matrix.map((row) => row[3]).slice(0, 3);
  return [
    [rotation[0][0], rotation[0][1], rotation[0][2], -dot(rotation[0], translation)],
    [rotation[1][0], rotation[1][1], rotation[1][2], -dot(rotation[1], translation)],
    [rotation[2][0], rotation[2][1], rotation[2][2], -dot(rotation[2], translation)],
    [0, 0, 0, 1]
  ];
}

function dot(left: number[], right: number[]): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

export function viewerAlphaThreshold(minimumOpacity: number): number {
  return Math.floor(minimumOpacity * 255);
}
