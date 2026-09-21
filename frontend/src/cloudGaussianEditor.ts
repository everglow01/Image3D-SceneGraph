import { Matrix4, PerspectiveCamera } from "three";

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

export function imagePoint(x: number, y: number, box: {left: number; top: number; width: number; height: number}, width: number, height: number): Pixel | null {
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
