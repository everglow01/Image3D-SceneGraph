import assert from "node:assert/strict";
import test from "node:test";
import type { ExtSplats } from "@sparkjsdev/spark";
import { verifyP0EncodedRow, p0FullMask, p0IdentityFixture, p0Visible, validateP0Mask, verifyP0FixtureRows } from "../src/localGaussianP0Source.ts";

test("P0 exact row references include every geometry and SH word, including coincident centers", () => {
  const make = (count: number) => ({ numSplats: count, extArrays: [new Uint32Array(count * 4), new Uint32Array(count * 4)],
    extra: Object.fromEntries(["sh1", "sh2", "sh3a", "sh3b"].map(k => [k, new Uint32Array(count * 4)])) }) as unknown as ExtSplats;
  const actual = make(2), reference = make(1);
  verifyP0EncodedRow(actual, 0, reference);
  verifyP0EncodedRow(actual, 1, reference);
  const arrays = [...actual.extArrays, ...["sh1", "sh2", "sh3a", "sh3b"].map(k => actual.extra[k] as Uint32Array)];
  for (const array of arrays) for (let word = 0; word < 4; word++) {
    array[4 + word] = 1;
    assert.throws(() => verifyP0EncodedRow(actual, 1, reference), /源行 1 编码/);
    verifyP0EncodedRow(actual, 0, reference);
    array[4 + word] = 0;
  }
  assert.throws(() => verifyP0EncodedRow(actual, -1, reference), /不合法/);
  assert.throws(() => verifyP0EncodedRow(actual, 2, reference), /不合法/);
  assert.throws(() => verifyP0EncodedRow(actual, 0, actual), /不合法/);
  delete reference.extra.sh3b;
  assert.throws(() => verifyP0EncodedRow(actual, 0, reference), /完整/);
});

const sha256 = "a".repeat(64);

test("P0 identity probe distinguishes every source row and rejects shuffled decoder output", () => {
  const fixture = p0IdentityFixture();
  const source = { sha256, count: fixture.count, geometry: fixture.expected.slice() };
  verifyP0FixtureRows(source, fixture.expected);
  source.geometry.copyWithin(0, 11, 22);
  assert.throws(() => verifyP0FixtureRows(source, fixture.expected), /源行 0/);
  assert.throws(() => verifyP0FixtureRows({ ...source, count: fixture.count - 1 }, fixture.expected), /不合法/);
  source.geometry.set(fixture.expected); source.geometry[0] = NaN;
  assert.throws(() => verifyP0FixtureRows(source, fixture.expected), /源行 0/);
});

test("P0 fixture is the strict 62-float SH3 layout with a weak first row", () => {
  const fixture = p0IdentityFixture();
  const text = new TextDecoder().decode(fixture.bytes.slice(0, 2048));
  const end = text.indexOf("end_header\n") + "end_header\n".length;
  assert.ok(end > 0);
  assert.equal(fixture.bytes.length - end, fixture.count * 62 * 4);
  assert.match(text.slice(0, end), /property float f_rest_44/);
  assert.ok(fixture.expected[10] < 1 / 255);
  const centers = Array.from({ length: fixture.count }, (_, i) => Array.from(fixture.expected.slice(i * 11, i * 11 + 3)).join(","));
  assert.equal(new Set(centers).size, fixture.count);
});

test("P0 packed masks use original row IDs and reject nonzero padding", () => {
  const mask = p0FullMask(17);
  validateP0Mask(mask, 17);
  assert.deepEqual([...mask], [255, 255, 1]);
  mask[0] &= ~(1 << 3);
  assert.equal(p0Visible(mask, 3), false);
  assert.equal(p0Visible(mask, 4), true);
  assert.equal(p0Visible(mask, 16), true);
  assert.throws(() => validateP0Mask(new Uint8Array([255, 255, 3]), 17), /补齐位/);
  assert.throws(() => validateP0Mask(new Uint8Array(2), 17), /长度/);
  assert.throws(() => p0FullMask(3_000_001), /数量/);
});
