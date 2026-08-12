declare module "@mkkellogg/gaussian-splats-3d" {
  import type * as THREE from "three";

  export const RenderMode: {
    readonly Always: 0;
    readonly OnChange: 1;
    readonly Never: 2;
  };

  export type ViewerControls = {
    target: THREE.Vector3;
    rotateSpeed: number;
    zoomSpeed: number;
    panSpeed: number;
    enableDamping: boolean;
    dampingFactor: number;
    screenSpacePanning: boolean;
    zoomToCursor: boolean;
    minDistance: number;
    maxDistance: number;
    minPolarAngle: number;
    maxPolarAngle: number;
    enabled: boolean;
    update(): void;
    saveState(): void;
  };

  export type SplatMesh = {
    computeBoundingBox(applySceneTransforms?: boolean, sceneIndex?: number): THREE.Box3;
  };

  export class Viewer {
    camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    controls: ViewerControls | null;
    renderer?: THREE.WebGLRenderer;

    constructor(options?: {
      rootElement?: HTMLElement;
      cameraUp?: [number, number, number];
      initialCameraPosition?: [number, number, number];
      initialCameraLookAt?: [number, number, number];
      sharedMemoryForWorkers?: boolean;
      sphericalHarmonicsDegree?: number;
      ignoreDevicePixelRatio?: boolean;
      integerBasedSort?: boolean;
      renderMode?: (typeof RenderMode)[keyof typeof RenderMode];
      showInfo?: boolean;
      threeScene?: THREE.Scene;
    });

    addSplatScene(
      path: string,
      options?: {
        format?: number;
        showLoadingUI?: boolean;
        progressiveLoad?: boolean;
        splatAlphaRemovalThreshold?: number;
        position?: [number, number, number];
        rotation?: [number, number, number, number];
        scale?: [number, number, number];
      }
    ): Promise<void>;

    start(): void;
    stop(): void;
    forceRenderNextFrame?(): void;
    getSplatMesh(): SplatMesh;
    dispose(): Promise<void>;
  }
}
