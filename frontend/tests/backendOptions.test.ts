import test from "node:test";
import assert from "node:assert/strict";
import {
  isBackendAvailable,
  isOutputSupported
} from "../src/backendOptions.ts";

test("unknown backend availability does not lock the selectors", () => {
  assert.equal(isBackendAvailable("vggt", null), true);
  assert.equal(isOutputSupported("mesh", undefined), true);
});
