import assert from "node:assert/strict";
import test from "node:test";

import {
  parseContentLength,
  parseGaussianExportMetadata,
  viewerAlphaThreshold
} from "../src/gaussianViewerMetadata.ts";

test("missing Content-Length remains unknown", () => {
  assert.equal(parseContentLength(null), null);
  assert.equal(parseContentLength(""), null);
  assert.equal(parseContentLength("invalid"), null);
  assert.equal(parseContentLength("247352"), 247352);
});

test("export metadata controls SH and alpha threshold", () => {
  const metadata = parseGaussianExportMetadata({
    sh_degree: 3,
    viewer_minimum_opacity: 0.005,
    scene_center: [0.1, -0.2, 0.3],
    scene_radius_p95: 1.25
  });

  assert.deepEqual(metadata, {
    sh_degree: 3,
    viewer_minimum_opacity: 0.005,
    scene_center: [0.1, -0.2, 0.3],
    scene_radius_p95: 1.25
  });
  assert.equal(viewerAlphaThreshold(metadata.viewer_minimum_opacity), 1);
});

test("legacy export metadata falls back to bounding-box framing", () => {
  assert.deepEqual(
    parseGaussianExportMetadata({ sh_degree: 2, viewer_minimum_opacity: 0.01 }),
    {
      sh_degree: 2,
      viewer_minimum_opacity: 0.01,
      scene_center: null,
      scene_radius_p95: null
    }
  );
});
