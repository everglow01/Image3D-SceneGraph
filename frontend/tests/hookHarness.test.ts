import assert from "node:assert/strict";
import test from "node:test";
import { createHookHarness, findNode } from "./hookHarness.ts";

test("shared hook harness waits for an explicit state and cleans effects", async () => {
  const h = createHookHarness();
  let value = -1, cleaned = 0;
  const render = () => {
    const [count, setCount] = h.react.useState(0);
    value = count;
    h.react.useEffect(() => {
      void Promise.resolve().then(() => setCount(1));
      return () => { cleaned++; };
    }, []);
  };
  await h.flush(render, () => value === 1);
  assert.equal(value, 1);
  await assert.rejects(h.flush(render, () => value === 2), /expected state/);
  h.unmount();
  assert.equal(cleaned, 1);
  const child = { type: "button", props: {} };
  assert.equal(findNode({ props: { children: [false, [child]] } }, n => n.type === "button"), child);
});
