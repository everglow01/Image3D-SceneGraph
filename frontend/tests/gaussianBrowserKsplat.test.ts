import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { convertBrowserKsplat } from "../../scripts/convert_gaussian_browser_ksplat.mjs";

const fields = [
  "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
  ...Array.from({ length: 45 }, (_, index) => `f_rest_${index}`),
  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
];

function fixture(sh3 = 0): Buffer {
  const count = 64;
  const header = Buffer.from([
    "ply", "format binary_little_endian 1.0", `element vertex ${count}`,
    ...fields.map((field) => `property float ${field}`), "end_header", ""
  ].join("\n"));
  const ply = Buffer.alloc(header.length + count * fields.length * 4);
  header.copy(ply);
  for (let row = 0; row < count; row += 1) {
    for (let column = 0; column < fields.length; column += 1) {
      const field = fields[column];
      const rest = field.startsWith("f_rest_") ? Number(field.slice(7)) : -1;
      const thirdOrder = rest >= 0 && rest % 15 >= 8;
      const value = field === "x" ? row * 0.01
        : field === "opacity" ? 4
        : field.startsWith("scale_") ? Math.log(0.05)
        : field === "rot_0" ? 1
        : rest >= 0 ? thirdOrder ? sh3 : 0.02 * (rest % 8 + 1) : 0;
      ply.writeFloatLE(value, header.length + (row * fields.length + column) * 4);
    }
  }
  return ply;
}

test("SH3 PLY converts to readable compressed SH2 without changing count", async () => {
  const input = fixture();
  for (const level of [1, 2]) {
    const output = await convertBrowserKsplat(input, level);
    assert.ok(output.byteLength < input.byteLength);
  }
});

test("browser SH2 derivative excludes third-order coefficients", async () => {
  const baseline = await convertBrowserKsplat(fixture(0), 2);
  const altered = await convertBrowserKsplat(fixture(0.8), 2);
  assert.deepEqual(altered, baseline);
});

test("converter rejects noncanonical PLY without writing output", async () => {
  const malformed = fixture();
  malformed.write("property double x", malformed.indexOf("property float x"), "ascii");
  await assert.rejects(convertBrowserKsplat(malformed, 1), { name: "AssertionError" });
});

test("offline CLI writes only a new KSPLAT and refuses overwrite", async () => {
  const directory = await mkdtemp(join(tmpdir(), "image3d-ksplat-test-"));
  try {
    const input = join(directory, "scene.ply");
    const output = join(directory, "browser.ksplat");
    const source = fixture();
    await writeFile(input, source);
    const script = new URL("../../scripts/convert_gaussian_browser_ksplat.mjs", import.meta.url);
    const args = [script.pathname, input, output, "2"];
    assert.equal(spawnSync(process.execPath, args, { encoding: "utf8" }).status, 0);
    assert.deepEqual(await readFile(input), source);
    const converted = await readFile(output);
    assert.ok(converted.byteLength < source.byteLength);
    assert.notEqual(spawnSync(process.execPath, args, { encoding: "utf8" }).status, 0);
    assert.deepEqual(await readFile(output), converted);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
