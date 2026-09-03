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
  summarizeSfmViewGraph,
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
          profile: "bruteforce",
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


function v3Payload(images: unknown[]) {
  const value = v2Payload(images);
  value.schema_version = 3;
  value.profile = "sfm_frontend_diagnostics_v3";
  Object.assign(value.runs[0], {
    geometric_verification: {
      profile: "guided_v1",
      guided_matching: true,
      skip_geometric_verification: false,
      raw_parameter_policy: "colmap_build_defaults",
      implementation: "colmap",
      version: "4.0"
    },
    view_graph: {
      schema_version: 1,
      profile: "sfm_verified_view_graph_v1",
      edge_definition: "nonempty_two_view_geometry",
      node_count: images.length,
      registered_node_count: images.length,
      tested_pair_count: 1,
      candidate_pair_count: 1,
      verified_edge_count: 1,
      verified_edge_ratio: 1,
      match_totals: {
        candidate: 4,
        candidate_inliers: 3,
        guided_inliers: 1,
        verified: 4,
        outliers: 1
      },
      geometric_config_counts: { "3": 1 },
      connected_component_count: 1,
      largest_component_node_count: images.length,
      largest_component_ratio: 1,
      largest_component_registered_node_count: images.length,
      largest_component_unregistered_node_count: 0,
      isolated_node_count: 0,
      degree_one_node_count: images.length,
      degree_distribution: { count: images.length, min: 1, p50: 1, p90: 1, max: 1 },
      component_size_distribution: { count: 1, min: images.length, p50: images.length, p90: images.length, max: images.length },
      candidate_match_distribution: { count: 1, min: 4, p50: 4, p90: 4, max: 4 },
      verified_inlier_distribution: { count: 1, min: 4, p50: 4, p90: 4, max: 4 },
      candidate_survival_ratio_distribution: { count: 1, min: 0.75, p50: 0.75, p90: 0.75, max: 0.75 },
      video: null
    }
  });
  return value;
}


function v4Payload(images: unknown[]) {
  const value = v3Payload(
    images.map((entry) => ({
      ...(entry as Record<string, unknown>),
      camera_id: 1
    }))
  );
  value.schema_version = 4;
  value.profile = "sfm_frontend_diagnostics_v4";
  Object.assign(value.runs[0], {
    camera_calibration: {
      profile: "shared_opencv_v1",
      camera_model: "OPENCV",
      sharing_policy: "single_camera",
      grouping_key_policy: "all_images",
      initial_focal_policy: "colmap_exif_or_default",
      planned_camera_count: 1,
      initial_camera_count: 1,
      final_camera_count: 1,
      prior_focal_camera_count: 1,
      warning_count: 0,
      implementation: "colmap",
      version: "4.0",
      diagnostics_path: "diagnostics/sfm_camera_calibration.json"
    }
  });
  return value;
}


test("schema 1 provenance maps to explicit SIFT stages", () => {
  const run = parseSfmDiagnostics(payload([image(1, [0, 0, 0], [0, 0, 1])])).runs[0];

  assert.equal(run.feature.profile, "sift_v1");
  assert.equal(run.feature.extractor, "SIFT");
  assert.equal(run.local_matcher.profile, "bruteforce");
  assert.equal(run.local_matcher.name, "SIFT_BRUTEFORCE");
  assert.equal(run.pairing.name, "sequential");
  assert.equal(run.camera_calibration.profile, "shared_opencv_v1");
  assert.equal(run.camera_calibration.final_camera_count, null);
  assert.equal(run.mapper.name, "incremental");
});


test("schema 3 preserves geometric verification and view graph provenance", () => {
  const value = v3Payload([
    image(1, [0, 0, 0], [0, 0, 1]),
    image(2, [1, 0, 0], [0, 0, 1])
  ]);

  const run = parseSfmDiagnostics(value).runs[0];

  assert.equal(run.geometric_verification.profile, "guided_v1");
  assert.equal(run.geometric_verification.guided_matching, true);
  assert.equal(run.view_graph?.verified_edge_count, 1);
  assert.equal(run.view_graph?.match_totals.guided_inliers, 1);
});


test("schema 3 rejects inconsistent geometric verification", () => {
  const value = v3Payload([
    image(1, [0, 0, 0], [0, 0, 1]),
    image(2, [1, 0, 0], [0, 0, 1])
  ]);
  value.runs[0].geometric_verification.guided_matching = false;

  assert.throws(() => parseSfmDiagnostics(value), /inconsistent/);
});


test("schema 4 preserves camera calibration and image grouping", () => {
  const value = v4Payload([
    image(1, [0, 0, 0], [0, 0, 1]),
    image(2, [1, 0, 0], [0, 0, 1])
  ]);

  const diagnostics = parseSfmDiagnostics(value);

  assert.equal(
    diagnostics.runs[0].camera_calibration.profile,
    "shared_opencv_v1"
  );
  assert.equal(diagnostics.runs[0].camera_calibration.camera_model, "OPENCV");
  assert.equal(diagnostics.runs[0].camera_calibration.final_camera_count, 1);
  assert.deepEqual(
    diagnostics.images.map((entry) => entry.camera_id),
    [1, 1]
  );
});


test("schema 4 rejects inconsistent camera calibration", () => {
  const value = v4Payload([
    image(1, [0, 0, 0], [0, 0, 1]),
    image(2, [1, 0, 0], [0, 0, 1])
  ]);
  value.runs[0].camera_calibration.camera_model = "SIMPLE_RADIAL";

  assert.throws(() => parseSfmDiagnostics(value), /inconsistent/);

  value.runs[0].camera_calibration.camera_model = "OPENCV";
  value.images[1].camera_id = 2;
  assert.throws(() => parseSfmDiagnostics(value), /image mapping is inconsistent/);
});


test("schema 2 preserves learned feature provenance", () => {
  const run = parseSfmDiagnostics(v2Payload([image(1, [0, 0, 0], [0, 0, 1])])).runs[0];

  assert.equal(run.feature.profile, "aliked_n16rot_v1");
  assert.equal(run.feature.extractor_model_sha256, hash);
  assert.equal(run.local_matcher.profile, "bruteforce");
  assert.equal(run.local_matcher.name, "ALIKED_BRUTEFORCE");
  assert.equal(run.pairing.name, "exhaustive");
});


test("schema 2 validates vocabulary-tree pairing provenance", () => {
  const value = v2Payload([image(1, [0, 0, 0], [0, 0, 1])]);
  value.runs[0].pairing = {
    name: "vocab_tree",
    implementation: "colmap",
    version: "4.0",
    vocab_tree_sha256: hash
  };

  const run = parseSfmDiagnostics(value).runs[0];

  assert.equal(run.pairing.name, "vocab_tree");
  assert.equal(run.pairing.vocab_tree_sha256, hash);

  delete value.runs[0].pairing.vocab_tree_sha256;
  assert.throws(() => parseSfmDiagnostics(value), /provenance is missing/);
});


test("schema 2 accepts explicit LightGlue provenance", () => {
  const value = v2Payload([image(1, [0, 0, 0], [0, 0, 1])]);
  value.runs[0].local_matcher = {
    profile: "lightglue",
    name: "ALIKED_LIGHTGLUE",
    implementation: "colmap",
    version: "4.0",
    model_sha256: hash
  };

  const run = parseSfmDiagnostics(value).runs[0];

  assert.equal(run.local_matcher.profile, "lightglue");
  assert.equal(run.local_matcher.name, "ALIKED_LIGHTGLUE");
});


test("schema 2 rejects inconsistent matcher profile and name", () => {
  const value = v2Payload([image(1, [0, 0, 0], [0, 0, 1])]);
  value.runs[0].local_matcher.profile = "lightglue";

  assert.throws(() => parseSfmDiagnostics(value), /inconsistent/);
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

test("pair index schema 2 separates guided matches from candidate survival", () => {
  const index = parsePairIndex({
    schema_version: 2,
    pairs: [
      {
        pair_key: "1-2",
        image_ids: [1, 2],
        candidate_match_count: 4,
        candidate_inlier_count: 3,
        guided_inlier_count: 2,
        inlier_count: 5,
        outlier_count: 1,
        geometric_config: 3,
        detail_shard: "diagnostics/sfm/pair.json.gz"
      }
    ]
  });

  assert.equal(index[0].candidate_inlier_count, 3);
  assert.equal(index[0].guided_inlier_count, 2);
  assert.equal(index[0].inlier_count, 5);
  assert.equal(index[0].outlier_count, 1);
});


test("historical pair indexes derive the same compact view graph summary", () => {
  const images = [
    image(1, [0, 0, 0], [0, 0, 1]),
    image(2, [1, 0, 0], [0, 0, 1]),
    image(3, null, null)
  ].map((entry) => parseSfmDiagnostics(payload([entry])).images[0]);
  const pairs = parsePairIndex({
    schema_version: 1,
    pairs: [
      { pair_key: "1-2", image_ids: [1, 2], candidate_match_count: 4, inlier_count: 3, geometric_config: 3, detail_shard: "a.json.gz" }
    ]
  });

  const summary = summarizeSfmViewGraph(images, pairs);

  assert.equal(summary.connected_component_count, 2);
  assert.equal(summary.largest_component_node_count, 2);
  assert.equal(summary.isolated_node_count, 1);
  assert.equal(summary.match_totals.guided_inliers, 0);
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
