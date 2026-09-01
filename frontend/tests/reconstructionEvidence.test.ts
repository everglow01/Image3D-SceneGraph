import assert from "node:assert/strict";
import test from "node:test";

import { buildEvidenceStages } from "../src/reconstructionEvidence.ts";

test("evidence stages distinguish old jobs from complete diagnostics", () => {
  const old = buildEvidenceStages({
    hasDiagnostics: false,
    hasSparseGeometry: true,
    hasGaussian: true,
    sparsePointCount: 549_025,
    gaussianCount: 13_100_000
  });
  assert.equal(old[0].available, false);
  assert.equal(old[0].value, "诊断不可用");
  assert.equal(old[2].value, "549,025 点");
  assert.equal(old[3].available, true);
  assert.equal(
    buildEvidenceStages({ hasDiagnostics: false, hasSparseGeometry: true, hasGaussian: true })[2].value,
    "可查看"
  );

  const complete = buildEvidenceStages({
    hasDiagnostics: true,
    hasSparseGeometry: true,
    hasGaussian: true,
    imageCount: 1_931,
    registeredImageCount: 1_742,
    pairCount: 12_345,
    sparsePointCount: 549_025,
    gaussianCount: 13_100_000
  });
  assert.equal(complete[0].value, "1,742 / 1,931 注册");
  assert.equal(complete[1].value, "12,345 图对");
  assert.ok(complete.every((stage) => stage.available));
});
