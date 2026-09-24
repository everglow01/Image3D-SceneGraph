import assert from "node:assert/strict";
import test from "node:test";
import { dyno, type SplatMesh } from "@sparkjsdev/spark";
import { P0AlphaMask } from "../src/localGaussianP0Mask.ts";
import { p0FullMask } from "../src/localGaussianP0Source.ts";

function meshStub() {
  let updates = 0;
  const mesh = { isInitialized: true, numSplats: 17, maxSh: 3,
    extSplats: { numSplats: 17, getNumSh: () => 3, maxSh: 3, extra: {} },
    updateGenerator() { updates++; } } as unknown as SplatMesh;
  return { mesh, updates: () => updates };
}

test("P0 uses the real Spark shader builder and modifies alpha only", () => {
  const { mesh, updates } = meshStub();
  const mask = new P0AlphaMask(mesh);
  const program = new dyno.DynoProgram({ graph: mask.modifier, inputs: { gsplat: "inputSplat" }, outputs: { gsplat: "outputSplat" },
    template: new dyno.DynoProgramTemplate("{{ GLOBALS }}\nvoid main() {\n{{ STATEMENTS }}\n}") });
  assert.match(program.shader, /inputSplat/);
  assert.match(program.shader, /texelFetch/);
  assert.match(program.shader, /\.rgba\.a = 0\.0/);
  assert.doesNotMatch(program.shader, /\.rgba(?:\.rgb|\.[rgb])?\s*=/);
  assert.ok(Object.values(program.uniforms).some(value => value.value === mask.texture));
  const visible = p0FullMask(17); visible[0] &= ~8;
  mask.update(visible);
  assert.equal((mask.texture.image.data as Uint8Array)[0], 247);
  visible[0] = 0;
  assert.equal((mask.texture.image.data as Uint8Array)[0], 247);
  mask.update(p0FullMask(17));
  assert.equal((mask.texture.image.data as Uint8Array)[0], 255);
  assert.equal(updates(), 3);
  assert.equal(mesh.maxSh, 3);
  assert.equal(mesh.splatRgba, undefined);
  mask.dispose(); mask.dispose();
  assert.equal(mesh.objectModifiers, undefined);
  assert.equal(updates(), 4);
  assert.throws(() => mask.update(p0FullMask(17)), /释放/);
});

test("P0 rejects unsupported pipelines rather than overwriting modifiers", () => {
  const { mesh } = meshStub();
  mesh.enableLod = true;
  assert.throws(() => new P0AlphaMask(mesh), /非 LOD/);
  mesh.enableLod = false;
  const mask = new P0AlphaMask(mesh);
  assert.throws(() => new P0AlphaMask(mesh), /未经修改/);
  assert.throws(() => mask.update(new Uint8Array([255, 255, 255])), /补齐位/);
  mask.dispose();
});
