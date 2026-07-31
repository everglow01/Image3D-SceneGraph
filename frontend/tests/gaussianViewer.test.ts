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
    viewer_minimum_opacity: 0.005
  });

  assert.deepEqual(metadata, { sh_degree: 3, viewer_minimum_opacity: 0.005 });
  assert.equal(viewerAlphaThreshold(metadata.viewer_minimum_opacity), 1);
});
