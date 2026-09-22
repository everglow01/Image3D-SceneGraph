import assert from "node:assert/strict";
import { readFile, open, stat } from "node:fs/promises";
import { pathToFileURL } from "node:url";

// The installed viewer expects a browser timer even when converting in Node.
globalThis.window ??= { setTimeout };
const { PlyLoader, KSplatLoader } = await import(
  "../frontend/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js"
);

const fields = [
  "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
  ...Array.from({ length: 45 }, (_, index) => `f_rest_${index}`),
  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
];

export async function convertBrowserKsplat(ply, compressionLevel) {
  assert.ok(compressionLevel === 1 || compressionLevel === 2, "compression must be 1 or 2");
  const headerEnd = ply.indexOf("end_header\n");
  assert.ok(headerEnd > 0 && headerEnd < 8192, "missing Gaussian PLY header");
  const headerSize = headerEnd + "end_header\n".length;
  const lines = ply.subarray(0, headerSize).toString("ascii").trimEnd().split("\n");
  assert.deepEqual(lines.slice(0, 2), ["ply", "format binary_little_endian 1.0"]);
  const match = lines.find((line) => line.startsWith("element vertex "))?.match(/^element vertex (\d+)$/);
  const count = match ? Number(match[1]) : NaN;
  assert.ok(Number.isSafeInteger(count) && count > 0 && count <= 3_000_000, "invalid Gaussian count");
  assert.deepEqual(lines.filter((line) => line.startsWith("property ")), fields.map((field) => `property float ${field}`));
  assert.equal(ply.byteLength, headerSize + count * fields.length * 4, "Gaussian PLY size mismatch");

  const data = ply.byteOffset === 0 && ply.byteLength === ply.buffer.byteLength
    ? ply.buffer : ply.buffer.slice(ply.byteOffset, ply.byteOffset + ply.byteLength);
  const splats = await PlyLoader.loadFromFileData(data, 0, compressionLevel, true, 2);
  assert.equal(splats.getSplatCount(), count, "conversion changed Gaussian count");
  assert.equal(splats.getMinSphericalHarmonicsDegree(), 2, "conversion lost SH2");
  const restored = await KSplatLoader.loadFromFileData(splats.bufferData);
  assert.equal(restored.getSplatCount(), count, "KSPLAT readback changed Gaussian count");
  assert.equal(restored.getMinSphericalHarmonicsDegree(), 2, "KSPLAT readback lost SH2");
  return Buffer.from(splats.bufferData);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [input, output, level] = process.argv.slice(2);
  if (!input || !output || !["1", "2"].includes(level) || !output.endsWith(".ksplat")) {
    throw new Error("usage: node scripts/convert_gaussian_browser_ksplat.mjs source.ply NEW-output.ksplat 1|2");
  }
  const source = await stat(input);
  assert.ok(source.isFile() && source.size <= 1_073_741_824, "input must be a regular PLY up to 1 GiB");
  const ply = await readFile(input);
  const converted = await convertBrowserKsplat(ply, Number(level));
  const handle = await open(output, "wx");
  try {
    await handle.writeFile(converted);
    await handle.sync();
  } finally {
    await handle.close();
  }
  console.log(JSON.stringify({ inputBytes: ply.byteLength, outputBytes: converted.byteLength, compressionLevel: Number(level) }));
}
