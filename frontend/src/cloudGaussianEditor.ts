import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import type { GaussianCameraPath, Mat4 } from "./gaussianViewerMetadata";

export type CameraView = { key: string; position: number[]; target: number[]; up: number[]; fov: number };

export function captureView(key: string, camera: PerspectiveCamera, target: Vector3): CameraView {
  return { key, position: camera.position.toArray(), target: target.toArray(), up: camera.up.toArray(), fov: camera.fov };
}

export function restoreView(camera: PerspectiveCamera, view: CameraView): Vector3 {
  camera.position.fromArray(view.position); camera.up.fromArray(view.up); camera.fov = view.fov;
  const target = new Vector3().fromArray(view.target);
  camera.lookAt(target); camera.updateProjectionMatrix();
  return target;
}

export function referenceView(camera: PerspectiveCamera, path: GaussianCameraPath | null, worldFromNormalized: Mat4 | null, upright: Matrix4, center: Vector3, radius: number): Vector3 {
  const key = worldFromNormalized ? path?.keyframes[0] : null;
  if (key && worldFromNormalized) {
    // Published camera rotation is in world space; its center is in normalized space.
    const normalizedFromWorld = new Matrix4().set(...worldFromNormalized.flat() as Parameters<Matrix4["set"]>).invert();
    const w = key.world_from_camera;
    const forward = new Vector3(w[0][2], w[1][2], w[2][2])
      .transformDirection(normalizedFromWorld).transformDirection(upright);
    camera.position.set(...key.center_normalized).applyMatrix4(upright);
    const target = camera.position.clone().addScaledVector(forward, Math.max(radius * 0.2, 0.1));
    camera.lookAt(target); return target;
  }
  camera.position.copy(center).addScaledVector(new Vector3(0.75, 1, 0.45).normalize(), radius * 2.4);
  camera.lookAt(center); return center.clone();
}


export type CloudCamera = {
  camera_from_normalized: number[][];
  intrinsic: number[][];
  width: number;
  height: number;
};
export type Pixel = [number, number];

export function renderSize(width: number, height: number): [number, number] {
  if (!Number.isFinite(width + height) || width <= 0 || height <= 0) throw new Error("视口尺寸无效");
  const scale = Math.min(1920 / width, 1920 / height, Math.sqrt(1920 * 1080 / (width * height)));
  return [Math.max(16, Math.floor(width * scale)), Math.max(16, Math.floor(height * scale))];
}

export function nativeCamera(camera: PerspectiveCamera, upright: Matrix4, width: number, height: number): CloudCamera {
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  const pose = new Matrix4().makeScale(1, -1, -1).multiply(camera.matrixWorldInverse).multiply(upright);
  const f = height / (2 * Math.tan(camera.fov * Math.PI / 360));
  return {
    camera_from_normalized: [0, 1, 2, 3].map(row => [0, 1, 2, 3].map(column => pose.elements[column * 4 + row])),
    intrinsic: [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], width, height
  };
}

export function imagePoint(x: number, y: number, box: {left: number; top: number; width: number; height: number}, width: number, height: number, zoom = 1, shift: Pixel = [0, 0]): Pixel | null {
  x = (x - box.left - box.width / 2 - shift[0]) / zoom + box.width / 2 + box.left;
  y = (y - box.top - box.height / 2 - shift[1]) / zoom + box.height / 2 + box.top;
  const scale = Math.min(box.width / width, box.height / height);
  if (!Number.isFinite(scale) || scale <= 0) return null;
  const px = (x - box.left - (box.width - width * scale) / 2) / scale;
  const py = (y - box.top - (box.height - height * scale) / 2) / scale;
  return px < 0 || py < 0 || px > width || py > height ? null : [px, py];
}

export function rectangle(a: Pixel, b: Pixel): Pixel[] {
  return [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
}

export function acceptsFrame(generation: number, incomingGeneration: number, revision: number, incomingRevision: number): boolean {
  return generation === incomingGeneration && revision === incomingRevision;
}

export function operationId(): string {
  // crypto.randomUUID needs HTTPS; getRandomValues also works on the existing LAN HTTP origin.
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), v => v.toString(16).padStart(2, "0")).join("");
}
