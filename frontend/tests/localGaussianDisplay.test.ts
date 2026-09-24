import assert from "node:assert/strict";
import test from "node:test";
import { dyno, type SplatMesh } from "@sparkjsdev/spark";
import { LocalGaussianDisplay } from "../src/localGaussianDisplay.ts";

test("显示高亮与仅alpha掩码分离，关闭高亮清空纹理而不改源属性", () => {
  const mesh = { isInitialized: true, numSplats: 6, maxSh: 3, opacity: 1,
    extSplats: { numSplats: 6, getNumSh: () => 3, maxSh: 3, extra: {} }, updateGenerator() {} } as unknown as SplatMesh;
  const ext = mesh.extSplats, display = new LocalGaussianDisplay(mesh);
  const program = new dyno.DynoProgram({ graph: display.modifier, inputs: { gsplat: "inputSplat" }, outputs: { gsplat: "outputSplat" },
    template: new dyno.DynoProgramTemplate("{{ GLOBALS }}\nvoid main() {\n{{ STATEMENTS }}\n}") });
  assert.match(program.shader, /\.rgba\.rgb = mix/);
  assert.doesNotMatch(program.shader, /\.rgba\.a\s*=/);
  assert.ok(Object.values(program.uniforms).some(value => value.value === display.texture));
  display.update(new Uint8Array([61]), new Uint8Array([3]), new Uint8Array([4]), true);
  assert.deepEqual(Array.from(display.texture.image.data as Uint8Array).slice(0, 2), [1, 4]);
  display.update(new Uint8Array([63]), new Uint8Array([3]), new Uint8Array([4]), false);
  assert.ok(Array.from(display.texture.image.data as Uint8Array).every(v => v === 0));
  assert.equal(mesh.extSplats, ext); assert.equal(mesh.maxSh, 3); assert.equal(mesh.splatRgba, undefined);
  assert.throws(() => display.update(new Uint8Array([255]), new Uint8Array([3]), new Uint8Array([0]), true), /补齐/);
  display.dispose(); display.dispose();
  assert.equal(mesh.worldModifiers, undefined); assert.equal(mesh.objectModifiers, undefined);
});
