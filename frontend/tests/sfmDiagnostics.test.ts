import assert from "node:assert/strict";
import test from "node:test";

import {
  assetUrl,
  pairKey,
  parsePairIndex,
  parseSfmDiagnostics,
  rankSfmImages,
  type SfmImage
} from "../src/sfmDiagnostics.ts";

const hash = "a".repeat(64);

function image(
  id: number,
  center: [number, number, number] | null,
  forward: [number, number, number] | null
) {
  return {
    frame_uid: hash,
    colmap_image_id: id,
    name: `frame-${id}.jpg`,
    path: `frames/selected/frame-${id}.jpg`,
    sha256: hash,
    width: 100,
    height: 80,
    registered: center !== null,
    split: center === null ? null : "train",
    feature_count: 20,
    ...(center === null
      ? {}
      : {
          center_normalized: center,
          forward_normalized: forward,
          up_normalized: [0, -1, 0],
          horizontal_fov_degrees: 60,
          vertical_fov_degrees: 45
        })
  };
}

function payload(images: unknown[]) {
  return {
    schema_version: 1,
    profile: "sfm_frontend_diagnostics_v1",
    coordinate_frame: "normalized",
    camera_convention: "opencv",
    world_units: "arbitrary",
    default_run_id: "run-1",
    runs: [
      {
        run_id: "run-1",
        detector: { name: "sift", implementation: "colmap", version: "4.0" },
        matcher: { name: "sequential", implementation: "colmap", version: "4.0" },
        feature_index_path: "diagnostics/sfm/features.json.gz",
        pair_index_path: "diagnostics/sfm/pairs.json.gz"
      }
    ],
    images
  };
}

test("exact input camera pose ranks itself first", () => {
  const diagnostics = parseSfmDiagnostics(
    payload([
      image(1, [0, 0, 0], [0, 0, 1]),
      image(2, [0.5, 0, 0], [0, 0, 1]),
      image(3, null, null)
    ])
  );

  const ranked = rankSfmImages(diagnostics, [0, 0, 0], [0, 0, 1], "view");

  assert.equal(ranked[0].image.colmap_image_id, 1);
  assert.equal(ranked[0].distance, 0);
  assert.equal(ranked[0].angleDegrees, 0);
  assert.equal(ranked[0].confidence, "good");
  assert.equal(ranked.some((entry) => entry.image.colmap_image_id === 3), false);
});

test("view-similar and position-nearest modes intentionally differ", () => {
  const diagnostics = parseSfmDiagnostics(
    payload([
      image(1, [0.05, 0, 0], [0, 0, -1]),
      image(2, [0.4, 0, 0], [0, 0, 1])
    ])
  );

  assert.equal(
    rankSfmImages(diagnostics, [0, 0, 0], [0, 0, 1], "position")[0].image.colmap_image_id,
    1
  );
  assert.equal(
    rankSfmImages(diagnostics, [0, 0, 0], [0, 0, 1], "view")[0].image.colmap_image_id,
    2
  );
});

test("unsafe paths and non-finite camera data are rejected", () => {
  assert.throws(
    () => parseSfmDiagnostics(payload([{ ...image(1, [0, 0, 0], [0, 0, 1]), path: "../secret.jpg" }])),
    /unsafe/
  );
  assert.throws(
    () => parseSfmDiagnostics(payload([image(1, [Number.NaN, 0, 0], [0, 0, 1])])),
    /finite/
  );
  assert.throws(() => assetUrl("job", "/absolute.json"), /unsafe/);
});

test("pair index distinguishes untested from tested zero-match pairs", () => {
  const index = parsePairIndex({
    schema_version: 1,
    pairs: [
      {
        pair_key: "1-2",
        image_ids: [1, 2],
        candidate_match_count: 0,
        inlier_count: 0,
        geometric_config: 0,
        detail_shard: "diagnostics/sfm/pair.json.gz"
      }
    ]
  });

  assert.equal(index.find((entry) => entry.pair_key === pairKey(1, 2))?.candidate_match_count, 0);
  assert.equal(index.find((entry) => entry.pair_key === pairKey(1, 3)), undefined);
  assert.equal(
    assetUrl("job-1", "frames/selected/a b.jpg"),
    "/api/jobs/job-1/assets/frames/selected/a%20b.jpg"
  );
});
