import assert from "node:assert/strict";
import test from "node:test";

import { formatResultKind } from "../src/resultKind.ts";

test("recovered derivative has an explicit read-only identity", () => {
  assert.equal(formatResultKind("salvaged_derivative"), "恢复衍生结果");
  assert.equal(formatResultKind(undefined), null);
});
