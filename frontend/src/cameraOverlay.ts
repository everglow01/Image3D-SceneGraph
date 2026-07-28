export type Vec3 = [number, number, number];
export type Mat4 = [
  number[],
  number[],
  number[],
  number[]
];

export type CameraFrame = {
  center: Vec3;
  corners: [Vec3, Vec3, Vec3, Vec3];
};

type AxisSigns = { x: 1 | -1; y: 1 | -1; z: 1 | -1 };

export function parseCameraFrames(payload: unknown, frustumDepth: number): CameraFrame[] {
  if (!Number.isFinite(frustumDepth) || frustumDepth <= 0) {
    throw new Error("Camera frustum depth must be positive");
  }
  const value = asRecord(payload, "camera payload");
  if (value.coordinate_system === "colmap_world") {
    return parseColmapFrames(value, frustumDepth);
  }
  if (value.coordinate_system === "opencv_camera_from_world") {
    return parseVggtFrames(value, frustumDepth);
  }
  throw new Error("Unsupported camera coordinate system");
}

export function parseAlignmentTransform(payload: unknown): Mat4 {
  const value = asRecord(payload, "alignment payload");
  const matrix = finiteMatrix(value.transform, 4, 4, "alignment transform") as Mat4;
  if (!matrix[3].every((item, index) => Math.abs(item - [0, 0, 0, 1][index]) <= 1e-12)) {
    throw new Error("Alignment transform must be affine");
  }
  return matrix;
}

export function transformCameraFrames(
  frames: CameraFrame[],
  worldTransform: Mat4 | null,
  cloudCenter: Vec3
): CameraFrame[] {
  return frames.map((frame) => ({
    center: centerPoint(transformPoint(frame.center, worldTransform), cloudCenter),
    corners: frame.corners.map((corner) =>
      centerPoint(transformPoint(corner, worldTransform), cloudCenter)
    ) as CameraFrame["corners"]
  }));
}

export function applyAxisSigns(point: Vec3, signs: AxisSigns): Vec3 {
  return [point[0] * signs.x, point[1] * signs.y, point[2] * signs.z];
}

export function cameraLinePositions(frames: CameraFrame[]): Float32Array {
  const positions: number[] = [];
  for (const frame of frames) {
    for (const corner of frame.corners) {
      addSegment(positions, frame.center, corner);
    }
    for (let index = 0; index < frame.corners.length; index += 1) {
      addSegment(positions, frame.corners[index], frame.corners[(index + 1) % 4]);
    }
  }
  for (let index = 1; index < frames.length; index += 1) {
    addSegment(positions, frames[index - 1].center, frames[index].center);
  }
  return new Float32Array(positions);
}

function parseColmapFrames(payload: Record<string, unknown>, depth: number): CameraFrame[] {
  const cameras = arrayValue(payload.cameras, "COLMAP cameras");
  const images = arrayValue(payload.images, "COLMAP images");
  const cameraById = new Map<number, Record<string, unknown>>();
  for (const item of cameras) {
    const camera = asRecord(item, "COLMAP camera");
    cameraById.set(finiteNumber(camera.camera_id, "camera_id"), camera);
  }
  return images.map((item) => {
    const image = asRecord(item, "COLMAP image");
    const camera = cameraById.get(finiteNumber(image.camera_id, "camera_id"));
    if (!camera) {
      throw new Error("COLMAP image references a missing camera");
    }
    const qvec = finiteVector(image.qvec, 4, "COLMAP qvec");
    const tvec = finiteVector(image.tvec, 3, "COLMAP tvec") as Vec3;
    const worldFromCamera = transpose3(qvecToRotation(qvec));
    const center = scale3(multiply3(worldFromCamera, tvec), -1);
    const cameraCorners = imagePlaneCorners(colmapIntrinsics(camera), depth);
    return {
      center,
      corners: cameraCorners.map((corner) =>
        add3(center, multiply3(worldFromCamera, corner))
      ) as CameraFrame["corners"]
    };
  });
}

function parseVggtFrames(payload: Record<string, unknown>, depth: number): CameraFrame[] {
  const cameras = arrayValue(payload.cameras, "VGGT cameras");
  const imageShape = asRecord(payload.image_shape, "VGGT image shape");
  const width = finiteNumber(imageShape.width, "image width");
  const height = finiteNumber(imageShape.height, "image height");
  return cameras.map((item) => {
    const camera = asRecord(item, "VGGT camera");
    const cameraToWorld = finiteMatrix(camera.camera_to_world, 3, 4, "camera_to_world");
    const intrinsic = finiteMatrix(camera.intrinsic, 3, 3, "intrinsic");
    const center: Vec3 = [cameraToWorld[0][3], cameraToWorld[1][3], cameraToWorld[2][3]];
    const rotation = cameraToWorld.slice(0, 3).map((row) => row.slice(0, 3));
    const cameraCorners = imagePlaneCorners(
      {
        width,
        height,
        fx: intrinsic[0][0],
        fy: intrinsic[1][1],
        cx: intrinsic[0][2],
        cy: intrinsic[1][2]
      },
      depth
    );
    return {
      center,
      corners: cameraCorners.map((corner) =>
        add3(center, multiply3(rotation, corner))
      ) as CameraFrame["corners"]
    };
  });
}

function colmapIntrinsics(camera: Record<string, unknown>) {
  const model = String(camera.model);
  if (!["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"].includes(model)) {
    throw new Error(`Unsupported COLMAP camera model: ${model}`);
  }
  const params = finiteVector(camera.params, model === "PINHOLE" ? 4 : 3, "camera params");
  const width = finiteNumber(camera.width, "camera width");
  const height = finiteNumber(camera.height, "camera height");
  if (model === "PINHOLE") {
    return { width, height, fx: params[0], fy: params[1], cx: params[2], cy: params[3] };
  }
  if (["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"].includes(model)) {
    return { width, height, fx: params[0], fy: params[0], cx: params[1], cy: params[2] };
  }
  throw new Error(`Unsupported COLMAP camera model: ${model}`);
}

function imagePlaneCorners(
  camera: { width: number; height: number; fx: number; fy: number; cx: number; cy: number },
  depth: number
): CameraFrame["corners"] {
  if (camera.fx <= 0 || camera.fy <= 0 || camera.width <= 0 || camera.height <= 0) {
    throw new Error("Camera dimensions and focal lengths must be positive");
  }
  return [
    pixelToCamera(0, 0, camera, depth),
    pixelToCamera(camera.width, 0, camera, depth),
    pixelToCamera(camera.width, camera.height, camera, depth),
    pixelToCamera(0, camera.height, camera, depth)
  ];
}

function pixelToCamera(
  u: number,
  v: number,
  camera: { fx: number; fy: number; cx: number; cy: number },
  depth: number
): Vec3 {
  return [((u - camera.cx) / camera.fx) * depth, ((v - camera.cy) / camera.fy) * depth, depth];
}

function qvecToRotation(qvec: number[]): number[][] {
  const norm = Math.hypot(...qvec);
  if (norm <= 1e-12) {
    throw new Error("COLMAP qvec cannot be zero");
  }
  const [w, x, y, z] = qvec.map((value) => value / norm);
  return [
    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
  ];
}

function transformPoint(point: Vec3, matrix: Mat4 | null): Vec3 {
  if (!matrix) {
    return [...point];
  }
  return [
    matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
    matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
    matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3]
  ];
}

function centerPoint(point: Vec3, center: Vec3): Vec3 {
  return [point[0] - center[0], point[1] - center[1], point[2] - center[2]];
}

function asRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function arrayValue(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
}

function finiteVector(value: unknown, length: number, label: string): number[] {
  if (!Array.isArray(value) || value.length < length) {
    throw new Error(`${label} has the wrong length`);
  }
  return value.slice(0, length).map((item) => finiteNumber(item, label));
}

function finiteMatrix(value: unknown, rows: number, columns: number, label: string): number[][] {
  if (!Array.isArray(value) || value.length !== rows) {
    throw new Error(`${label} has the wrong shape`);
  }
  return value.map((row) => {
    if (!Array.isArray(row) || row.length !== columns) {
      throw new Error(`${label} has the wrong shape`);
    }
    return row.map((item) => finiteNumber(item, label));
  });
}

function multiply3(matrix: number[][], vector: Vec3): Vec3 {
  return matrix.map((row) => row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2]) as Vec3;
}

function transpose3(matrix: number[][]): number[][] {
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

function add3(left: Vec3, right: Vec3): Vec3 {
  return [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
}

function scale3(vector: Vec3, scale: number): Vec3 {
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale];
}

function addSegment(positions: number[], start: Vec3, end: Vec3) {
  positions.push(...start, ...end);
}
