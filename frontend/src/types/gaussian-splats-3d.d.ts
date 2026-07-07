declare module "@mkkellogg/gaussian-splats-3d" {
  export class Viewer {
    constructor(options?: {
      rootElement?: HTMLElement;
      cameraUp?: [number, number, number];
      initialCameraPosition?: [number, number, number];
      initialCameraLookAt?: [number, number, number];
      sharedMemoryForWorkers?: boolean;
      sphericalHarmonicsDegree?: number;
      ignoreDevicePixelRatio?: boolean;
      integerBasedSort?: boolean;
      showInfo?: boolean;
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
    dispose(): Promise<void>;
  }
}
