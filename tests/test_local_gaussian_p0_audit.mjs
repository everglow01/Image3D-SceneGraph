import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readP0String } from '../scripts/gaussian_local_p0_cdp.mjs';

test('P0 transports multi-megabyte evidence in bounded CDP messages', async () => {
  const text = 'x'.repeat(4 * 1024 * 1024 + 17); let messages = 0;
  const result = await readP0String(async (method, { expression }) => {
    assert.equal(method, 'Runtime.evaluate'); messages++;
    if (expression.endsWith('.length')) return { result: { value: text.length } };
    const [, begin, end] = expression.match(/\.slice\((\d+),(\d+)\)$/);
    assert.ok(Number(end) - Number(begin) <= 65536);
    return { result: { value: text.slice(Number(begin), Number(end)) } };
  }, 'window.evidence');
  assert.equal(result, text); assert.equal(messages, 66);
});

test('P0 rejects oversized, truncated and exceptional CDP evidence', async () => {
  await assert.rejects(readP0String(async () => ({ result: { value: 16 * 1024 * 1024 + 1 } }), 'window.evidence'), /长度超限/);
  await assert.rejects(readP0String(async (_, { expression }) => ({ result: { value: expression.endsWith('.length') ? 100 : '' } }), 'window.evidence'), /截断/);
  await assert.rejects(readP0String(async () => ({ exceptionDetails: { text: 'not available' } }), 'window.evidence'), /not available/);
});
