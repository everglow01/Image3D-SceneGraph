import assert from "node:assert/strict";
import test from "node:test";

import {
  assetUrl,
  filterSfmImages,
  pairKey,
  pairNeighbors,
  parsePairIndex,
  parseSfmDiagnostics,
  rankSfmImages,
  sampleDeterministic,
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

function v2Payload(images: unknown[]) {
  return {
    ...payload(images),
    schema_version: 2,
    profile: "sfm_frontend_diagnostics_v2",
    runs: [
      {
        run_id: "run-1",
        feature: {
          profile: "aliked_n16rot_v1",
          extractor: "ALIKED_N16ROT",
          descriptor: "ALIKED",
          max_features: 8192,
          extractor_model_sha256: hash,
          implementation: "colmap",
          version: "4.0"
        },
        local_matcher: {
          name: "ALIKED_BRUTEFORCE",
          implementation: "colmap",
          version: "4.0",
          model_sha256: hash
        },
        pairing: { name: "exhaustive", implementation: "colmap", version: "4.0" },
        mapper: { name: "incremental", implementation: "colmap", version: "4.0" },
        feature_index_path: "diagnostics/sfm/features.json.gz",
        pair_index_path: "diagnostics/sfm/pairs.json.gz"
      }
    ]
  };
}


test("schema 1 provenance maps to explicit SIFT stages", () => {
  const run = parseSfmDiagnostics(payload([image(1, [0, 0, 0], [0, 0, 1])])).runs[0];

  assert.equal(run.feature.profile, "sift_v1");
  assert.equal(run.feature.extractor, "SIFT");
  assert.equal(run.local_matcher.name, "SIFT_BRUTEFORCE");
  assert.equal(run.pairing.name, "sequential");
  assert.equal(run.mapper.name, "incremental");
});


test("schema 2 preserves learned feature provenance", () => {
  const run = parseSfmDiagnostics(v2Payload([image(1, [0, 0, 0], [0, 0, 1])])).runs[0];

  assert.equal(run.feature.profile, "aliked_n16rot_v1");
  assert.equal(run.feature.extractor_model_sha256, hash);
  assert.equal(run.local_matcher.name, "ALIKED_BRUTEFORCE");
  assert.equal(run.pairing.name, "exhaustive");
});


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

test("frame filtering and pair adjacency expose every tested neighbor", () => {
  const diagnostics = parseSfmDiagnostics(
    payload([
      { ...image(1, [0, 0, 0], [0, 0, 1]), source_time_seconds: 2 },
      { ...image(2, [0, 0, 0], [0, 0, 1]), source_time_seconds: 1, split: "validation" },
      { ...image(3, null, null), source_time_seconds: 1.5 }
    ])
  );
  assert.deepEqual(
    filterSfmImages(diagnostics.images, "", "all", "all").map((item) => item.colmap_image_id),
    [2, 3, 1]
  );
  assert.deepEqual(
    filterSfmImages(diagnostics.images, "frame-3", "unregistered", "unassigned").map(
      (item) => item.colmap_image_id
    ),
    [3]
  );

  const index = parsePairIndex({
    schema_version: 1,
    pairs: [
      { pair_key: "1-2", image_ids: [1, 2], candidate_match_count: 10, inlier_count: 5, geometric_config: 3, detail_shard: "a.json.gz" },
      { pair_key: "1-3", image_ids: [1, 3], candidate_match_count: 20, inlier_count: 12, geometric_config: 3, detail_shard: "b.json.gz" },
      { pair_key: "2-3", image_ids: [2, 3], candidate_match_count: 50, inlier_count: 1, geometric_config: 3, detail_shard: "c.json.gz" }
    ]
  });
  const neighbors = pairNeighbors(index, 1);
  assert.deepEqual(neighbors.map((item) => item.neighbor_image_id), [3, 2]);
  assert.equal(neighbors[0].inlier_rate, 0.6);
  assert.deepEqual(sampleDeterministic([0, 1, 2, 3, 4], 3), [0, 1, 3]);
});
