import * as THREE from "three";
import { dyno, type GsplatModifier, type SplatMesh } from "@sparkjsdev/spark";
import { p0FullMask, requireP0Mesh, validateP0Mask } from "./localGaussianP0Source.ts";

const WIDTH = 1024;

export class P0AlphaMask {
  readonly texture: THREE.DataTexture;
  readonly modifier: GsplatModifier;
  private readonly mesh: SplatMesh;
  private readonly count: number;
  private disposed = false;

  constructor(mesh: SplatMesh) {
    requireP0Mesh(mesh);
    this.mesh = mesh;
    this.count = mesh.numSplats;
    const initial = p0FullMask(this.count);
    const bytes = new Uint8Array(WIDTH * Math.ceil(initial.length / WIDTH));
    bytes.set(initial);
    this.texture = new THREE.DataTexture(bytes, WIDTH, bytes.length / WIDTH, THREE.RedIntegerFormat, THREE.UnsignedByteType);
    this.texture.internalFormat = "R8UI";
    this.texture.unpackAlignment = 1;
    this.texture.minFilter = this.texture.magFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
    this.texture.needsUpdate = true;
    const mask = dyno.dynoUsampler2D(this.texture);
    this.modifier = dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
      return new dyno.Dyno({
        inTypes: { gsplat: dyno.Gsplat, mask: "usampler2D" }, outTypes: { gsplat: dyno.Gsplat },
        inputs: { gsplat, mask },
        statements: ({ inputs, outputs }) => [
          `${outputs.gsplat} = ${inputs.gsplat};`,
          `int sourceId = ${inputs.gsplat}.index;`,
          `if (sourceId < 0 || sourceId >= ${this.count}) { ${outputs.gsplat}.rgba.a = 0.0; }`,
          "else {",
          "  int maskByte = sourceId >> 3;",
          `  uint bits = texelFetch(${inputs.mask}, ivec2(maskByte % ${WIDTH}, maskByte / ${WIDTH}), 0).r;`,
          `  if ((bits & (1u << uint(sourceId & 7))) == 0u) ${outputs.gsplat}.rgba.a = 0.0;`,
          "}"
        ]
      }).outputs;
    });
    mesh.objectModifiers = [this.modifier];
    mesh.updateGenerator();
  }

  update(visible: Uint8Array) {
    if (this.disposed || this.mesh.numSplats !== this.count || this.mesh.objectModifiers?.[0] !== this.modifier) {
      throw new Error("P0 掩码已释放或模型管线已变化");
    }
    validateP0Mask(visible, this.count);
    const data = this.texture.image.data as Uint8Array;
    data.fill(0);
    data.set(visible);
    this.texture.needsUpdate = true;
    this.mesh.updateGenerator();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.mesh.objectModifiers?.length === 1 && this.mesh.objectModifiers[0] === this.modifier) {
      this.mesh.objectModifiers = undefined;
      this.mesh.updateGenerator();
    }
    this.texture.dispose();
  }
}
