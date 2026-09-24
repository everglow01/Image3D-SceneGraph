import * as THREE from "three";
import { dyno, type GsplatModifier, type SplatMesh } from "@sparkjsdev/spark";
import { P0AlphaMask } from "./localGaussianP0Mask.ts";
import { validateP0Mask } from "./localGaussianP0Source.ts";

export class LocalGaussianDisplay {
  private readonly alpha: P0AlphaMask;
  readonly texture: THREE.DataTexture;
  readonly modifier: GsplatModifier;
  private disposed = false;
  private readonly mesh: SplatMesh;

  constructor(mesh: SplatMesh) {
    this.mesh = mesh;
    this.alpha = new P0AlphaMask(mesh);
    const width = 1024, length = Math.ceil(mesh.numSplats / 8);
    const data = new Uint8Array(width * Math.ceil(length / width) * 2);
    this.texture = new THREE.DataTexture(data, width, data.length / (2 * width), THREE.RGIntegerFormat, THREE.UnsignedByteType);
    this.texture.internalFormat = "RG8UI";
    this.texture.unpackAlignment = 1;
    this.texture.minFilter = this.texture.magFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
    this.texture.needsUpdate = true;
    try {
      const mask = dyno.dynoUsampler2D(this.texture);
      this.modifier = dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => new dyno.Dyno({
      inTypes: { gsplat: dyno.Gsplat, mask: "usampler2D" }, outTypes: { gsplat: dyno.Gsplat }, inputs: { gsplat, mask },
      statements: ({ inputs, outputs }) => [
        `${outputs.gsplat} = ${inputs.gsplat};`,
        `int id = ${inputs.gsplat}.index;`,
        `if (id >= 0 && id < ${mesh.numSplats}) {`,
        `  uvec2 bits = texelFetch(${inputs.mask}, ivec2((id >> 3) % ${width}, (id >> 3) / ${width}), 0).rg;`,
        "  uint bit = 1u << uint(id & 7);",
        `  if ((bits.g & bit) != 0u) ${outputs.gsplat}.rgba.rgb = mix(${inputs.gsplat}.rgba.rgb, vec3(0.1, 0.8, 1.0), 0.55);`,
        `  if ((bits.r & bit) != 0u) ${outputs.gsplat}.rgba.rgb = mix(${outputs.gsplat}.rgba.rgb, vec3(1.0, 0.6, 0.05), 0.65);`,
        "}"
      ]
    }).outputs);
      mesh.worldModifiers = [this.modifier];
      mesh.updateGenerator();
    } catch (error) { mesh.worldModifiers = undefined; this.alpha.dispose(); this.texture.dispose(); throw error; }
  }

  update(visible: Uint8Array, selected: Uint8Array, protectedMask: Uint8Array, highlight: boolean) {
    if (this.disposed || this.mesh.worldModifiers?.length !== 1 || this.mesh.worldModifiers[0] !== this.modifier) throw new Error("本地显示管线已变化");
    for (const mask of [visible, selected, protectedMask]) validateP0Mask(mask, this.mesh.numSplats);
    this.alpha.update(visible);
    const data = this.texture.image.data as Uint8Array;
    data.fill(0);
    if (highlight) for (let i = 0; i < selected.length; i++) {
      data[i * 2] = selected[i] & visible[i]; data[i * 2 + 1] = protectedMask[i] & visible[i];
    }
    this.texture.needsUpdate = true;
    this.mesh.updateGenerator();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.mesh.worldModifiers?.length === 1 && this.mesh.worldModifiers[0] === this.modifier) this.mesh.worldModifiers = undefined;
    this.alpha.dispose();
    this.texture.dispose();
    this.mesh.updateGenerator();
  }
}
