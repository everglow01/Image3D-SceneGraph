export type Vec3 = [number, number, number];

export type SfmImage = {
  frame_uid: string;
  colmap_image_id: number;
  name: string;
  path: string;
  sha256: string;
  width: number;
  height: number;
  registered: boolean;
  split: "train" | "validation" | "test" | null;
  source_time_seconds: number | null;
  feature_count: number;
  center_normalized: Vec3 | null;
  forward_normalized: Vec3 | null;
  up_normalized: Vec3 | null;
  horizontal_fov_degrees: number | null;
  vertical_fov_degrees: number | null;
};

export type SfmRun = {
  run_id: string;
  detector: { name: string; implementation: string; version: string };
  matcher: { name: string; implementation: string; version: string };
  feature_index_path: string;
  pair_index_path: string;
};

export type SfmDiagnostics = {
  default_run_id: string;
  images: SfmImage[];
  runs: SfmRun[];
};

export type FeatureIndexEntry = {
  image_id: number;
  feature_count: number;
  detail_shard: string;
};

export type PairIndexEntry = {
  pair_key: string;
  image_ids: [number, number];
  candidate_match_count: number;
  inlier_count: number;
  geometric_config: number;
  detail_shard: string;
};

export type SfmPairDetail = {
  inliers: Array<[number, number]>;
  outliers: Array<[number, number]>;
};

export type RankedSfmImage = {
  image: SfmImage;
  distance: number;
  angleDegrees: number;
  score: number;
  frustumRelation: "aligned" | "edge" | "outside";
  confidence: "good" | "limited" | "poor";
};

export type RankingMode = "view" | "position";

export function parseSfmDiagnostics(value: unknown): SfmDiagnostics {
  const record = object(value, "SfM diagnostics");
  if (
    record.schema_version !== 1 ||
    record.profile !== "sfm_frontend_diagnostics_v1" ||
    record.coordinate_frame !== "normalized" ||
    record.camera_convention !== "opencv" ||
    record.world_units !== "arbitrary"
  ) {
    throw new Error("SfM diagnostics schema is unsupported");
  }
  const runs = array(record.runs, "SfM runs").map(parseRun);
  const defaultRunId = text(record.default_run_id, "default run ID");
  if (!runs.some((run) => run.run_id === defaultRunId)) {
    throw new Error("SfM default run is missing");
  }
  const images = array(record.images, "SfM images").map(parseImage);
  const ids = new Set(images.map((image) => image.colmap_image_id));
  if (ids.size !== images.length) {
    throw new Error("SfM image IDs are not unique");
  }
  return { default_run_id: defaultRunId, runs, images };
}

export function parseFeatureIndex(value: unknown): FeatureIndexEntry[] {
  const record = object(value, "feature index");
  if (record.schema_version !== 1) {
    throw new Error("Feature index schema is unsupported");
  }
  return array(record.images, "feature index images").map((value) => {
    const entry = object(value, "feature index entry");
    return {
      image_id: positiveInteger(entry.image_id, "feature image ID"),
      feature_count: nonNegativeInteger(entry.feature_count, "feature count"),
      detail_shard: assetPath(entry.detail_shard)
    };
  });
}

export function parsePairIndex(value: unknown): PairIndexEntry[] {
  const record = object(value, "pair index");
  if (record.schema_version !== 1) {
    throw new Error("Pair index schema is unsupported");
  }
  return array(record.pairs, "pair index pairs").map((value) => {
    const entry = object(value, "pair index entry");
    const imageIds = array(entry.image_ids, "pair image IDs");
    if (imageIds.length !== 2) {
      throw new Error("Pair image IDs have the wrong length");
    }
    const left = positiveInteger(imageIds[0], "left image ID");
    const right = positiveInteger(imageIds[1], "right image ID");
    if (left >= right) {
      throw new Error("Pair image IDs are not ordered");
    }
    return {
      pair_key: text(entry.pair_key, "pair key"),
      image_ids: [left, right],
      candidate_match_count: nonNegativeInteger(entry.candidate_match_count, "candidate matches"),
      inlier_count: nonNegativeInteger(entry.inlier_count, "inliers"),
      geometric_config: nonNegativeInteger(entry.geometric_config, "geometric config"),
      detail_shard: assetPath(entry.detail_shard)
    };
  });
}

export function parseFeatureShard(value: unknown, imageId: number): Array<[number, number]> {
  const record = object(value, "feature shard");
  if (record.schema_version !== 1) {
    throw new Error("Feature shard schema is unsupported");
  }
  const images = object(record.images, "feature shard images");
  const image = object(images[String(imageId)], "feature shard image");
  return array(image.points, "feature points").map((point) => coordinatePair(point, "feature point"));
}

export function parsePairShard(value: unknown, key: string): SfmPairDetail {
  const record = object(value, "pair shard");
  if (record.schema_version !== 1) {
    throw new Error("Pair shard schema is unsupported");
  }
  const pairs = object(record.pairs, "pair shard pairs");
  const pair = object(pairs[key], "pair shard detail");
  return {
    inliers: array(pair.inliers, "pair inliers").map((match) => integerPair(match, "inlier")),
    outliers: array(pair.outliers, "pair outliers").map((match) => integerPair(match, "outlier"))
  };
}

export function rankSfmImages(
  diagnostics: SfmDiagnostics,
  queryCenter: Vec3,
  queryForward: Vec3,
  mode: RankingMode,
  limit = 3
): RankedSfmImage[] {
  const center = finiteVec3(queryCenter, "query center");
  const forward = unit(finiteVec3(queryForward, "query forward"));
  const ranked = diagnostics.images
    .filter(
      (image): image is SfmImage & {
        center_normalized: Vec3;
        forward_normalized: Vec3;
        horizontal_fov_degrees: number;
        vertical_fov_degrees: number;
      } =>
        image.registered &&
        image.center_normalized !== null &&
        image.forward_normalized !== null &&
        image.horizontal_fov_degrees !== null &&
        image.vertical_fov_degrees !== null
    )
    .map((image) => {
      const distance = length(subtract(image.center_normalized, center));
      const cameraForward = unit(image.forward_normalized);
      const angleDegrees = radiansToDegrees(
        Math.acos(clamp(dot(cameraForward, forward), -1, 1))
      );
      const score =
        mode === "position"
          ? distance
          : 0.5 * clamp(distance / 2, 0, 1) + 0.5 * (angleDegrees / 180);
      const halfNarrowFov = Math.min(
        image.horizontal_fov_degrees,
        image.vertical_fov_degrees
      ) / 2;
      const halfDiagonalFov =
        Math.hypot(image.horizontal_fov_degrees, image.vertical_fov_degrees) / 2;
      const frustumRelation =
        angleDegrees <= halfNarrowFov
          ? "aligned"
          : angleDegrees <= halfDiagonalFov
            ? "edge"
            : "outside";
      const confidence =
        distance <= 0.35 && angleDegrees <= 30
          ? "good"
          : distance <= 0.8 && angleDegrees <= 60
            ? "limited"
            : "poor";
      return { image, distance, angleDegrees, score, frustumRelation, confidence } satisfies RankedSfmImage;
    });
  return ranked
    .sort((left, right) => left.score - right.score || left.image.colmap_image_id - right.image.colmap_image_id)
    .slice(0, Math.max(0, Math.floor(limit)));
}

export function pairKey(left: number, right: number): string {
  const first = Math.min(left, right);
  const second = Math.max(left, right);
  return `${first}-${second}`;
}

export function assetUrl(jobId: string, path: string): string {
  if (!jobId || jobId.includes("/") || jobId.includes("\\")) {
    throw new Error("Job ID is invalid");
  }
  return `/api/jobs/${encodeURIComponent(jobId)}/assets/${assetPath(path)
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;
}

function parseRun(value: unknown): SfmRun {
  const run = object(value, "SfM run");
  return {
    run_id: text(run.run_id, "run ID"),
    detector: parseAlgorithm(run.detector, "detector"),
    matcher: parseAlgorithm(run.matcher, "matcher"),
    feature_index_path: assetPath(run.feature_index_path),
    pair_index_path: assetPath(run.pair_index_path)
  };
}

function parseAlgorithm(value: unknown, label: string) {
  const record = object(value, label);
  return {
    name: text(record.name, `${label} name`),
    implementation: text(record.implementation, `${label} implementation`),
    version: text(record.version, `${label} version`)
  };
}

function parseImage(value: unknown): SfmImage {
  const image = object(value, "SfM image");
  const registered = boolean(image.registered, "registered");
  const split = image.split;
  if (split !== null && split !== "train" && split !== "validation" && split !== "test") {
    throw new Error("SfM image split is invalid");
  }
  const center = optionalVec3(image.center_normalized, "camera center");
  const forward = optionalVec3(image.forward_normalized, "camera forward");
  const up = optionalVec3(image.up_normalized, "camera up");
  const horizontalFov = optionalPositiveNumber(image.horizontal_fov_degrees, "horizontal FOV");
  const verticalFov = optionalPositiveNumber(image.vertical_fov_degrees, "vertical FOV");
  if (registered !== (center !== null && forward !== null && up !== null && horizontalFov !== null && verticalFov !== null)) {
    throw new Error("SfM image registration and pose are inconsistent");
  }
  return {
    frame_uid: sha256(image.frame_uid, "frame UID"),
    colmap_image_id: positiveInteger(image.colmap_image_id, "COLMAP image ID"),
    name: text(image.name, "image name"),
    path: assetPath(image.path),
    sha256: sha256(image.sha256, "image SHA-256"),
    width: positiveInteger(image.width, "image width"),
    height: positiveInteger(image.height, "image height"),
    registered,
    split,
    source_time_seconds: optionalNonNegativeNumber(image.source_time_seconds, "source time"),
    feature_count: nonNegativeInteger(image.feature_count, "feature count"),
    center_normalized: center,
    forward_normalized: forward,
    up_normalized: up,
    horizontal_fov_degrees: horizontalFov,
    vertical_fov_degrees: verticalFov
  };
}

function assetPath(value: unknown): string {
  const path = text(value, "asset path");
  const segments = path.replaceAll("\\", "/").split("/");
  if (
    path.startsWith("/") ||
    segments.some((segment) => !segment || segment === "." || segment === "..") ||
    path.includes("?") ||
    path.includes("#")
  ) {
    throw new Error("Asset path is unsafe");
  }
  return segments.join("/");
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || !value) {
    throw new Error(`${label} must be non-empty text`);
  }
  return value;
}

function boolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be boolean`);
  }
  return value;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
}

function positiveInteger(value: unknown, label: string): number {
  const number = finite(value, label);
  if (!Number.isInteger(number) || number <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return number;
}

function nonNegativeInteger(value: unknown, label: string): number {
  const number = finite(value, label);
  if (!Number.isInteger(number) || number < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return number;
}

function optionalNonNegativeNumber(value: unknown, label: string): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const number = finite(value, label);
  if (number < 0) {
    throw new Error(`${label} must not be negative`);
  }
  return number;
}

function optionalPositiveNumber(value: unknown, label: string): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const number = finite(value, label);
  if (number <= 0 || number >= 180) {
    throw new Error(`${label} is outside its valid range`);
  }
  return number;
}

function coordinatePair(value: unknown, label: string): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error(`${label} must have two coordinates`);
  }
  return [finite(value[0], label), finite(value[1], label)];
}

function integerPair(value: unknown, label: string): [number, number] {
  const pair = coordinatePair(value, label);
  if (!pair.every((item) => Number.isInteger(item) && item >= 0)) {
    throw new Error(`${label} indices must be non-negative integers`);
  }
  return pair;
}

function optionalVec3(value: unknown, label: string): Vec3 | null {
  if (value === undefined || value === null) {
    return null;
  }
  return finiteVec3(value, label);
}

function finiteVec3(value: unknown, label: string): Vec3 {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${label} must be a 3-vector`);
  }
  return value.map((entry) => finite(entry, label)) as Vec3;
}

function sha256(value: unknown, label: string): string {
  const digest = text(value, label);
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`${label} is invalid`);
  }
  return digest;
}

function unit(vector: Vec3): Vec3 {
  const norm = length(vector);
  if (norm <= 1e-12) {
    throw new Error("Direction vector cannot be zero");
  }
  return vector.map((value) => value / norm) as Vec3;
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function length(vector: Vec3): number {
  return Math.hypot(...vector);
}

function dot(left: Vec3, right: Vec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function radiansToDegrees(value: number): number {
  return (value * 180) / Math.PI;
}
