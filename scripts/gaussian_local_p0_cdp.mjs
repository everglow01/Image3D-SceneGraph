export async function readP0String(call, expression) {
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', { expression, returnByValue: true });
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const length = await evaluate(`(${expression}).length`);
  if (!Number.isSafeInteger(length) || length < 0 || length > 16 * 1024 * 1024) throw Error('审计字符串长度超限');
  const chunks = [];
  // Large screenshot replies can close the local CDP connection; bound each message, not just the total.
  for (let offset = 0; offset < length; offset += 65536) {
    const chunk = await evaluate(`(${expression}).slice(${offset},${offset + 65536})`);
    if (typeof chunk !== 'string' || chunk.length !== Math.min(65536, length - offset)) throw Error('审计字符串截断');
    chunks.push(chunk);
  }
  return chunks.join('');
}
