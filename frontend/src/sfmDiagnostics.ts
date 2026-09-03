export type Vec3 = [number, number, number];
export type SfmInspectionTab = "nearest" | "keypoints" | "matches";

export type SfmImage = {
  frame_uid: string;
  colmap_image_id: number;
  camera_id: number | null;
  name: string;
  path: string;
  sha256: string;
  width: number;
  height: number;
  registered: boolean;
  split: "train" | "validation" | "test" | null;
  source_time_seconds: number | null;
  feature_count: number;
  center_normalized: Vec3 | null;
  forward_normalized: Vec3 | null;
  up_normalized: Vec3 | null;
  horizontal_fov_degrees: number | null;
  vertical_fov_degrees: number | null;
};

export type SfmAlgorithm = {
  name: string;
  implementation: string;
  version: string;
};

export type SfmFeature = {
  profile: string;
  extractor: string;
  descriptor: string;
  max_features: number;
  extractor_model_sha256: string | null;
  implementation: string;
  version: string;
};

export type SfmGeometricVerification = {
  profile: "default_v1" | "guided_v1";
  guided_matching: boolean;
  skip_geometric_verification: false;
  raw_parameter_policy: "colmap_build_defaults";
  implementation: string;
  version: string;
};

export type SfmCameraCalibration = {
  profile:
    | "shared_opencv_v1"
    | "shared_simple_radial_v1"
    | "auto_grouped_simple_radial_v1";
  camera_model: "OPENCV" | "SIMPLE_RADIAL";
  sharing_policy: "single_camera" | "focal_aware_groups";
  planned_camera_count: number | null;
  initial_camera_count: number | null;
  final_camera_count: number | null;
  prior_focal_camera_count: number | null;
  warning_count: number | null;
  implementation: string;
  version: string;
  diagnostics_path: string | null;
};

export type SfmDistributionMiddle = {
  p50: number;
  p90: number;
};

export type SfmViewGraphSummary = {
  node_count: number;
  tested_pair_count: number;
  verified_edge_count: number;
  verified_edge_ratio: number;
  connected_component_count: number;
  largest_component_node_count: number;
  largest_component_ratio: number;
  isolated_node_count: number;
  degree_one_node_count: number;
  degree_distribution: SfmDistributionMiddle;
  match_totals: { guided_inliers: number };
  video: {
    registered_gap_count: number;
    directly_bridged_registered_gap_count: number;
  } | null;
};

export type SfmRun = {
  run_id: string;
  feature: SfmFeature;
  local_matcher: SfmAlgorithm & {
    profile: "bruteforce" | "lightglue";
    model_sha256: string | null;
  };
  pairing: SfmAlgorithm & {
    vocab_tree_sha256: string | null;
  };
  geometric_verification: SfmGeometricVerification;
  camera_calibration: SfmCameraCalibration;
  view_graph: SfmViewGraphSummary | null;
  mapper: SfmAlgorithm;
  feature_index_path: string;
  pair_index_path: string;
};

export type SfmDiagnostics = {
  default_run_id: string;
  images: SfmImage[];
  runs: SfmRun[];
};

export type FeatureIndexEntry = {
  image_id: number;
  feature_count: number;
  detail_shard: string;
};

export type PairIndexEntry = {
  pair_key: string;
  image_ids: [number, number];
  candidate_match_count: number;
  candidate_inlier_count: number;
  guided_inlier_count: number;
  inlier_count: number;
  outlier_count: number;
  geometric_config: number;
  detail_shard: string;
};

export type SfmPairDetail = {
  inliers: Array<[number, number]>;
  outliers: Array<[number, number]>;
};

export type RankedSfmImage = {
  image: SfmImage;
  distance: number;
  angleDegrees: number;
  score: number;
  frustumRelation: "aligned" | "edge" | "outside";
  confidence: "good" | "limited" | "poor";
};

export type RankingMode = "view" | "position";
export type RegistrationFilter = "all" | "registered" | "unregistered";
export type SplitFilter = "all" | "train" | "validation" | "test" | "unassigned";

export type PairNeighbor = {
  entry: PairIndexEntry;
  neighbor_image_id: number;
  inlier_rate: number;
};

export function parseSfmDiagnostics(value: unknown): SfmDiagnostics {
  const record = object(value, "SfM diagnostics");
  const schemaVersion = record.schema_version;
  if (
    (schemaVersion !== 1 &&
      schemaVersion !== 2 &&
      schemaVersion !== 3 &&
      schemaVersion !== 4) ||
    record.profile !== `sfm_frontend_diagnostics_v${schemaVersion}` ||
    record.coordinate_frame !== "normalized" ||
    record.camera_convention !== "opencv" ||
    record.world_units !== "arbitrary"
  ) {
    throw new Error("SfM diagnostics schema is unsupported");
  }
  const runs = array(record.runs, "SfM runs").map((value) =>
    parseRun(value, schemaVersion)
  );
  const defaultRunId = text(record.default_run_id, "default run ID");
  if (!runs.some((run) => run.run_id === defaultRunId)) {
    throw new Error("SfM default run is missing");
  }
  const images = array(record.images, "SfM images").map((value) =>
    parseImage(value, schemaVersion)
  );
  const ids = new Set(images.map((image) => image.colmap_image_id));
  if (ids.size !== images.length) {
    throw new Error("SfM image IDs are not unique");
  }
  if (schemaVersion === 4) {
    const cameraIds = new Set(images.map((image) => image.camera_id));
    const registeredCameraIds = new Set(
      images
        .filter((image) => image.registered)
        .map((image) => image.camera_id)
    );
    if (
      cameraIds.has(null) ||
      runs.some(
        (run) =>
          run.camera_calibration.initial_camera_count !== cameraIds.size ||
          run.camera_calibration.final_camera_count !==
            registeredCameraIds.size
      )
    ) {
      throw new Error("SfM camera calibration image mapping is inconsistent");
    }
  }
  if (
    runs.some(
      (run) =>
        run.view_graph !== null && run.view_graph.node_count !== images.length
    )
  ) {
    throw new Error("SfM view graph node count is inconsistent");
  }
  return { default_run_id: defaultRunId, runs, images };
}

export function parseFeatureIndex(value: unknown): FeatureIndexEntry[] {
  const record = object(value, "feature index");
  if (record.schema_version !== 1) {
    throw new Error("Feature index schema is unsupported");
  }
  return array(record.images, "feature index images").map((value) => {
    const entry = object(value, "feature index entry");
    return {
      image_id: positiveInteger(entry.image_id, "feature image ID"),
      feature_count: nonNegativeInteger(entry.feature_count, "feature count"),
      detail_shard: assetPath(entry.detail_shard)
    };
  });
}

export function parsePairIndex(value: unknown): PairIndexEntry[] {
  const record = object(value, "pair index");
  const schemaVersion = record.schema_version;
  if (schemaVersion !== 1 && schemaVersion !== 2) {
    throw new Error("Pair index schema is unsupported");
  }
  return array(record.pairs, "pair index pairs").map((value) => {
    const entry = object(value, "pair index entry");
    const imageIds = array(entry.image_ids, "pair image IDs");
    if (imageIds.length !== 2) {
      throw new Error("Pair image IDs have the wrong length");
    }
    const left = positiveInteger(imageIds[0], "left image ID");
    const right = positiveInteger(imageIds[1], "right image ID");
    if (left >= right) {
      throw new Error("Pair image IDs are not ordered");
    }
    const candidateMatchCount = nonNegativeInteger(
      entry.candidate_match_count,
      "candidate matches"
    );
    const inlierCount = nonNegativeInteger(entry.inlier_count, "inliers");
    const candidateInlierCount =
      schemaVersion === 1
        ? inlierCount
        : nonNegativeInteger(
            entry.candidate_inlier_count,
            "candidate inliers"
          );
    const guidedInlierCount =
      schemaVersion === 1
        ? 0
        : nonNegativeInteger(entry.guided_inlier_count, "guided inliers");
    const outlierCount =
      schemaVersion === 1
        ? candidateMatchCount - candidateInlierCount
        : nonNegativeInteger(entry.outlier_count, "outliers");
    if (
      candidateInlierCount > candidateMatchCount ||
      candidateInlierCount + guidedInlierCount !== inlierCount ||
      candidateInlierCount + outlierCount !== candidateMatchCount
    ) {
      throw new Error("Pair match counts are inconsistent");
    }
    return {
      pair_key: text(entry.pair_key, "pair key"),
      image_ids: [left, right],
      candidate_match_count: candidateMatchCount,
      candidate_inlier_count: candidateInlierCount,
      guided_inlier_count: guidedInlierCount,
      inlier_count: inlierCount,
      outlier_count: outlierCount,
      geometric_config: nonNegativeInteger(
        entry.geometric_config,
        "geometric config"
      ),
      detail_shard: assetPath(entry.detail_shard)
    };
  });
}

export function parseFeatureShard(value: unknown, imageId: number): Array<[number, number]> {
  const record = object(value, "feature shard");
  if (record.schema_version !== 1) {
    throw new Error("Feature shard schema is unsupported");
  }
  const images = object(record.images, "feature shard images");
  const image = object(images[String(imageId)], "feature shard image");
  return array(image.points, "feature points").map((point) => coordinatePair(point, "feature point"));
}

export function parsePairShard(value: unknown, key: string): SfmPairDetail {
  const record = object(value, "pair shard");
  if (record.schema_version !== 1 && record.schema_version !== 2) {
    throw new Error("Pair shard schema is unsupported");
  }
  const pairs = object(record.pairs, "pair shard pairs");
  const pair = object(pairs[key], "pair shard detail");
  return {
    inliers: array(pair.inliers, "pair inliers").map((match) => integerPair(match, "inlier")),
    outliers: array(pair.outliers, "pair outliers").map((match) => integerPair(match, "outlier"))
  };
}

export function rankSfmImages(
  diagnostics: SfmDiagnostics,
  queryCenter: Vec3,
  queryForward: Vec3,
  mode: RankingMode,
  limit = 3
): RankedSfmImage[] {
  const center = finiteVec3(queryCenter, "query center");
  const forward = unit(finiteVec3(queryForward, "query forward"));
  const ranked = diagnostics.images
    .filter(
      (image): image is SfmImage & {
        center_normalized: Vec3;
        forward_normalized: Vec3;
        horizontal_fov_degrees: number;
        vertical_fov_degrees: number;
      } =>
        image.registered &&
        image.center_normalized !== null &&
        image.forward_normalized !== null &&
        image.horizontal_fov_degrees !== null &&
        image.vertical_fov_degrees !== null
    )
    .map((image) => {
      const distance = length(subtract(image.center_normalized, center));
      const cameraForward = unit(image.forward_normalized);
      const angleDegrees = radiansToDegrees(
        Math.acos(clamp(dot(cameraForward, forward), -1, 1))
      );
      const score =
        mode === "position"
          ? distance
          : 0.5 * clamp(distance / 2, 0, 1) + 0.5 * (angleDegrees / 180);
      const halfNarrowFov = Math.min(
        image.horizontal_fov_degrees,
        image.vertical_fov_degrees
      ) / 2;
      const halfDiagonalFov =
        Math.hypot(image.horizontal_fov_degrees, image.vertical_fov_degrees) / 2;
      const frustumRelation =
        angleDegrees <= halfNarrowFov
          ? "aligned"
          : angleDegrees <= halfDiagonalFov
            ? "edge"
            : "outside";
      const confidence =
        distance <= 0.35 && angleDegrees <= 30
          ? "good"
          : distance <= 0.8 && angleDegrees <= 60
            ? "limited"
            : "poor";
      return { image, distance, angleDegrees, score, frustumRelation, confidence } satisfies RankedSfmImage;
    });
  return ranked
    .sort((left, right) => left.score - right.score || left.image.colmap_image_id - right.image.colmap_image_id)
    .slice(0, Math.max(0, Math.floor(limit)));
}

export function pairKey(left: number, right: number): string {
  const first = Math.min(left, right);
  const second = Math.max(left, right);
  return `${first}-${second}`;
}

export function filterSfmImages(
  images: SfmImage[],
  query: string,
  registration: RegistrationFilter,
  split: SplitFilter
): SfmImage[] {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  return images
    .filter((image) => {
      if (registration === "registered" && !image.registered) return false;
      if (registration === "unregistered" && image.registered) return false;
      if (split === "unassigned" && image.split !== null) return false;
      if (split !== "all" && split !== "unassigned" && image.split !== split) return false;
      if (!normalizedQuery) return true;
      return (
        image.name.toLocaleLowerCase().includes(normalizedQuery) ||
        String(image.colmap_image_id) === normalizedQuery ||
        (image.source_time_seconds !== null && image.source_time_seconds.toFixed(2).includes(normalizedQuery))
      );
    })
    .sort((left, right) => {
      const leftTime = left.source_time_seconds ?? Number.POSITIVE_INFINITY;
      const rightTime = right.source_time_seconds ?? Number.POSITIVE_INFINITY;
      return leftTime - rightTime || left.name.localeCompare(right.name) || left.colmap_image_id - right.colmap_image_id;
    });
}

export function pairNeighbors(index: PairIndexEntry[], imageId: number): PairNeighbor[] {
  return index
    .filter((entry) => entry.image_ids.includes(imageId))
    .map((entry) => ({
      entry,
      neighbor_image_id: entry.image_ids[0] === imageId ? entry.image_ids[1] : entry.image_ids[0],
      inlier_rate:
        entry.candidate_match_count === 0
          ? 0
          : entry.candidate_inlier_count / entry.candidate_match_count
    }))
    .sort(
      (left, right) =>
        right.entry.inlier_count - left.entry.inlier_count ||
        right.inlier_rate - left.inlier_rate ||
        right.entry.candidate_match_count - left.entry.candidate_match_count ||
        left.entry.pair_key.localeCompare(right.entry.pair_key)
    );
}

export function summarizeSfmViewGraph(
  images: SfmImage[],
  pairs: PairIndexEntry[]
): SfmViewGraphSummary {
  if (images.length === 0) throw new Error("View graph has no images");
  const adjacency = new Map(
    images.map((image) => [image.colmap_image_id, new Set<number>()])
  );
  const verifiedEdges = pairs
    .filter((pair) => pair.inlier_count > 0)
    .map((pair) => pair.image_ids);
  for (const [left, right] of verifiedEdges) {
    const leftNeighbors = adjacency.get(left);
    const rightNeighbors = adjacency.get(right);
    if (!leftNeighbors || !rightNeighbors) {
      throw new Error("View graph pair references a missing image");
    }
    leftNeighbors.add(right);
    rightNeighbors.add(left);
  }
  const componentSizes = connectedComponentSizes(adjacency);
  const degrees = [...adjacency.values()].map((neighbors) => neighbors.size);
  return {
    node_count: images.length,
    tested_pair_count: pairs.length,
    verified_edge_count: verifiedEdges.length,
    verified_edge_ratio:
      pairs.length === 0 ? 0 : verifiedEdges.length / pairs.length,
    connected_component_count: componentSizes.length,
    largest_component_node_count: componentSizes[0],
    largest_component_ratio: componentSizes[0] / images.length,
    isolated_node_count: degrees.filter((degree) => degree === 0).length,
    degree_one_node_count: degrees.filter((degree) => degree === 1).length,
    degree_distribution: distributionMiddle(degrees),
    match_totals: {
      guided_inliers: pairs.reduce(
        (total, pair) => total + pair.guided_inlier_count,
        0
      )
    },
    video: summarizeViewGraphVideo(images, verifiedEdges)
  };
}

function connectedComponentSizes(adjacency: Map<number, Set<number>>): number[] {
  const remaining = new Set(adjacency.keys());
  const sizes: number[] = [];
  while (remaining.size > 0) {
    const stack = [remaining.values().next().value!];
    let size = 0;
    while (stack.length > 0) {
      const imageId = stack.pop()!;
      if (!remaining.delete(imageId)) continue;
      size += 1;
      stack.push(...(adjacency.get(imageId) ?? []));
    }
    sizes.push(size);
  }
  return sizes.sort((left, right) => right - left);
}

function summarizeViewGraphVideo(
  images: SfmImage[],
  verifiedEdges: Array<[number, number]>
): SfmViewGraphSummary["video"] {
  const timestamps = new Map(
    images.flatMap((image) =>
      image.source_time_seconds === null
        ? []
        : [[image.colmap_image_id, image.source_time_seconds] as const]
    )
  );
  if (timestamps.size === 0) return null;
  const registeredTimes = images
    .flatMap((image) =>
      image.registered && image.source_time_seconds !== null
        ? [image.source_time_seconds]
        : []
    )
    .sort((left, right) => left - right);
  const gaps = registeredTimes.slice(1).flatMap((end, index) =>
    end - registeredTimes[index] > 2
      ? [[registeredTimes[index], end] as const]
      : []
  );
  const timedEdges = verifiedEdges.flatMap(([left, right]) => {
    const leftTime = timestamps.get(left);
    const rightTime = timestamps.get(right);
    return leftTime === undefined || rightTime === undefined
      ? []
      : [[Math.min(leftTime, rightTime), Math.max(leftTime, rightTime)] as const];
  });
  const bridged = gaps.filter(([gapStart, gapEnd]) =>
    timedEdges.some(
      ([edgeStart, edgeEnd]) => edgeStart <= gapStart && edgeEnd >= gapEnd
    )
  ).length;
  return {
    registered_gap_count: gaps.length,
    directly_bridged_registered_gap_count: bridged
  };
}

function distributionMiddle(values: number[]) {
  const ordered = [...values].sort((left, right) => left - right);
  const percentile = (fraction: number) => {
    const position = (ordered.length - 1) * fraction;
    const lower = Math.floor(position);
    const weight = position - lower;
    return ordered[lower] * (1 - weight) + ordered[Math.ceil(position)] * weight;
  };
  return { p50: percentile(0.5), p90: percentile(0.9) };
}

export function sampleDeterministic<T>(items: T[], maximum: number): T[] {
  const limit = Math.max(0, Math.floor(maximum));
  if (items.length <= limit) return items;
  if (limit === 0) return [];
  return Array.from({ length: limit }, (_, index) => items[Math.floor((index * items.length) / limit)]);
}

export function assetUrl(jobId: string, path: string): string {
  if (!jobId || jobId.includes("/") || jobId.includes("\\")) {
    throw new Error("Job ID is invalid");
  }
  return `/api/jobs/${encodeURIComponent(jobId)}/assets/${assetPath(path)
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;
}

function parseRun(value: unknown, schemaVersion: 1 | 2 | 3 | 4): SfmRun {
  const run = object(value, "SfM run");
  const featureIndexPath = assetPath(run.feature_index_path);
  const pairIndexPath = assetPath(run.pair_index_path);
  if (schemaVersion === 1) {
    const detector = parseAlgorithm(run.detector, "detector");
    const legacyMatcher = parseAlgorithm(run.matcher, "matcher");
    return {
      run_id: text(run.run_id, "run ID"),
      feature: {
        profile: "sift_v1",
        extractor: "SIFT",
        descriptor: "SIFT",
        max_features: 8192,
        extractor_model_sha256: null,
        implementation: detector.implementation,
        version: detector.version
      },
      local_matcher: {
        profile: "bruteforce",
        name: "SIFT_BRUTEFORCE",
        implementation: "colmap",
        version: detector.version,
        model_sha256: null
      },
      pairing: { ...legacyMatcher, vocab_tree_sha256: null },
      geometric_verification: defaultGeometricVerification(
        detector.version
      ),
      camera_calibration: defaultCameraCalibration(detector.version),
      view_graph: null,
      mapper: {
        name: "incremental",
        implementation: "colmap",
        version: detector.version
      },
      feature_index_path: featureIndexPath,
      pair_index_path: pairIndexPath
    };
  }
  const feature = object(run.feature, "feature");
  const localMatcher = object(run.local_matcher, "local matcher");
  const parsedLocalMatcher = parseAlgorithm(localMatcher, "local matcher");
  const pairing = object(run.pairing, "pairing");
  const parsedPairing = parseAlgorithm(pairing, "pairing");
  const pairingNames = new Set([
    "exhaustive",
    "sequential",
    "sequential_loop",
    "vocab_tree"
  ]);
  if (!pairingNames.has(parsedPairing.name)) {
    throw new Error("pairing profile is invalid");
  }
  const pairingVocabTreeSha256 = optionalSha256(
    pairing.vocab_tree_sha256,
    "vocabulary tree SHA-256"
  );
  if (
    (parsedPairing.name === "sequential_loop" ||
      parsedPairing.name === "vocab_tree") &&
    pairingVocabTreeSha256 === null
  ) {
    throw new Error("pairing vocabulary-tree provenance is missing");
  }
  if (
    parsedPairing.name === "exhaustive" &&
    pairingVocabTreeSha256 !== null
  ) {
    throw new Error("exhaustive pairing has a vocabulary tree");
  }
  const geometricVerification =
    schemaVersion >= 3
      ? parseGeometricVerification(run.geometric_verification)
      : defaultGeometricVerification(parsedLocalMatcher.version);
  const cameraCalibration =
    schemaVersion === 4
      ? parseCameraCalibration(run.camera_calibration)
      : defaultCameraCalibration(parsedLocalMatcher.version);
  const viewGraph =
    schemaVersion >= 3 ? parseViewGraphSummary(run.view_graph) : null;
  return {
    run_id: text(run.run_id, "run ID"),
    feature: {
      profile: text(feature.profile, "feature profile"),
      extractor: text(feature.extractor, "feature extractor"),
      descriptor: text(feature.descriptor, "feature descriptor"),
      max_features: positiveInteger(feature.max_features, "maximum features"),
      extractor_model_sha256: optionalSha256(
        feature.extractor_model_sha256,
        "extractor model SHA-256"
      ),
      implementation: text(feature.implementation, "feature implementation"),
      version: text(feature.version, "feature version")
    },
    local_matcher: {
      ...parsedLocalMatcher,
      profile: localMatcherProfile(localMatcher.profile, parsedLocalMatcher.name),
      model_sha256: optionalSha256(
        localMatcher.model_sha256,
        "matcher model SHA-256"
      )
    },
    pairing: {
      ...parsedPairing,
      vocab_tree_sha256: pairingVocabTreeSha256
    },
    geometric_verification: geometricVerification,
    camera_calibration: cameraCalibration,
    view_graph: viewGraph,
    mapper: parseAlgorithm(run.mapper, "mapper"),
    feature_index_path: featureIndexPath,
    pair_index_path: pairIndexPath
  };
}

function parseGeometricVerification(value: unknown): SfmGeometricVerification {
  const record = object(value, "geometric verification");
  const implementation = text(
    record.implementation,
    "geometric verification implementation"
  );
  const version = text(record.version, "geometric verification version");
  const profile = record.profile;
  if (profile !== "default_v1" && profile !== "guided_v1") {
    throw new Error("geometric verification profile is invalid");
  }
  const guidedMatching = boolean(
    record.guided_matching,
    "guided matching"
  );
  if (guidedMatching !== (profile === "guided_v1")) {
    throw new Error("geometric verification profile is inconsistent");
  }
  if (
    record.skip_geometric_verification !== false ||
    record.raw_parameter_policy !== "colmap_build_defaults"
  ) {
    throw new Error("geometric verification policy is invalid");
  }
  return {
    implementation,
    version,
    profile,
    guided_matching: guidedMatching,
    skip_geometric_verification: false,
    raw_parameter_policy: "colmap_build_defaults"
  };
}

function defaultGeometricVerification(
  version: string
): SfmGeometricVerification {
  return {
    profile: "default_v1",
    guided_matching: false,
    skip_geometric_verification: false,
    raw_parameter_policy: "colmap_build_defaults",
    implementation: "colmap",
    version
  };
}

function parseCameraCalibration(value: unknown): SfmCameraCalibration {
  const record = object(value, "camera calibration");
  const profile = record.profile;
  const expected = {
    shared_opencv_v1: ["OPENCV", "single_camera", "all_images"],
    shared_simple_radial_v1: [
      "SIMPLE_RADIAL",
      "single_camera",
      "all_images"
    ],
    auto_grouped_simple_radial_v1: [
      "SIMPLE_RADIAL",
      "focal_aware_groups",
      "exif_device_lens_focal_size_orientation_v1"
    ]
  } as const;
  if (typeof profile !== "string" || !(profile in expected)) {
    throw new Error("camera calibration profile is invalid");
  }
  const typedProfile = profile as keyof typeof expected;
  const [cameraModel, sharingPolicy, groupingPolicy] = expected[typedProfile];
  if (
    record.camera_model !== cameraModel ||
    record.sharing_policy !== sharingPolicy ||
    record.grouping_key_policy !== groupingPolicy ||
    record.initial_focal_policy !== "colmap_exif_or_default"
  ) {
    throw new Error("camera calibration profile is inconsistent");
  }
  const planned = positiveInteger(
    record.planned_camera_count,
    "planned camera count"
  );
  const initial = positiveInteger(
    record.initial_camera_count,
    "initial camera count"
  );
  const final = positiveInteger(record.final_camera_count, "final camera count");
  const prior = nonNegativeInteger(
    record.prior_focal_camera_count,
    "prior focal camera count"
  );
  const warnings = nonNegativeInteger(
    record.warning_count,
    "camera warning count"
  );
  if (
    planned !== initial ||
    final > initial ||
    prior > initial ||
    (sharingPolicy === "single_camera" && (planned !== 1 || final !== 1))
  ) {
    throw new Error("camera calibration counts are inconsistent");
  }
  return {
    profile: typedProfile,
    camera_model: cameraModel,
    sharing_policy: sharingPolicy,
    planned_camera_count: planned,
    initial_camera_count: initial,
    final_camera_count: final,
    prior_focal_camera_count: prior,
    warning_count: warnings,
    implementation: text(
      record.implementation,
      "camera calibration implementation"
    ),
    version: text(record.version, "camera calibration version"),
    diagnostics_path: assetPath(record.diagnostics_path)
  };
}

function defaultCameraCalibration(version: string): SfmCameraCalibration {
  return {
    profile: "shared_opencv_v1",
    camera_model: "OPENCV",
    sharing_policy: "single_camera",
    planned_camera_count: null,
    initial_camera_count: null,
    final_camera_count: null,
    prior_focal_camera_count: null,
    warning_count: null,
    implementation: "colmap",
    version,
    diagnostics_path: null
  };
}

function parseViewGraphSummary(value: unknown): SfmViewGraphSummary {
  const record = object(value, "view graph");
  if (
    record.schema_version !== 1 ||
    record.profile !== "sfm_verified_view_graph_v1" ||
    record.edge_definition !== "nonempty_two_view_geometry"
  ) {
    throw new Error("view graph schema is unsupported");
  }
  const nodeCount = positiveInteger(record.node_count, "view graph nodes");
  const testedPairCount = nonNegativeInteger(
    record.tested_pair_count,
    "tested pairs"
  );
  const verifiedEdgeCount = nonNegativeInteger(
    record.verified_edge_count,
    "verified edges"
  );
  const componentCount = positiveInteger(
    record.connected_component_count,
    "connected components"
  );
  const largestComponent = positiveInteger(
    record.largest_component_node_count,
    "largest component nodes"
  );
  const isolated = nonNegativeInteger(record.isolated_node_count, "isolated nodes");
  const degreeOne = nonNegativeInteger(record.degree_one_node_count, "degree-one nodes");
  if (
    verifiedEdgeCount > testedPairCount ||
    componentCount > nodeCount ||
    largestComponent > nodeCount ||
    isolated > nodeCount ||
    degreeOne > nodeCount
  ) {
    throw new Error("view graph counts are inconsistent");
  }
  const totals = object(record.match_totals, "view graph match totals");
  const candidate = nonNegativeInteger(totals.candidate, "candidate total");
  const candidateInliers = nonNegativeInteger(
    totals.candidate_inliers,
    "candidate inlier total"
  );
  const guidedInliers = nonNegativeInteger(
    totals.guided_inliers,
    "guided inlier total"
  );
  const verified = nonNegativeInteger(totals.verified, "verified total");
  const outliers = nonNegativeInteger(totals.outliers, "outlier total");
  if (
    candidateInliers + guidedInliers !== verified ||
    candidateInliers + outliers !== candidate
  ) {
    throw new Error("view graph match totals are inconsistent");
  }
  const degree = object(record.degree_distribution, "degree distribution");
  const videoRecord = record.video;
  const video =
    videoRecord === null
      ? null
      : (() => {
          const item = object(videoRecord, "view graph video summary");
          const gaps = nonNegativeInteger(item.registered_gap_count, "registered gaps");
          const bridged = nonNegativeInteger(
            item.directly_bridged_registered_gap_count,
            "directly bridged gaps"
          );
          const unbridged = nonNegativeInteger(
            item.unbridged_registered_gap_count,
            "unbridged gaps"
          );
          if (bridged + unbridged !== gaps) {
            throw new Error("view graph gap counts are inconsistent");
          }
          return {
            registered_gap_count: gaps,
            directly_bridged_registered_gap_count: bridged
          };
        })();
  return {
    node_count: nodeCount,
    tested_pair_count: testedPairCount,
    verified_edge_count: verifiedEdgeCount,
    verified_edge_ratio: ratio(record.verified_edge_ratio, "verified edge ratio"),
    connected_component_count: componentCount,
    largest_component_node_count: largestComponent,
    largest_component_ratio: ratio(
      record.largest_component_ratio,
      "largest component ratio"
    ),
    isolated_node_count: isolated,
    degree_one_node_count: degreeOne,
    degree_distribution: {
      p50: nonNegativeNumber(degree.p50, "degree p50"),
      p90: nonNegativeNumber(degree.p90, "degree p90")
    },
    match_totals: { guided_inliers: guidedInliers },
    video
  };
}

function localMatcherProfile(
  value: unknown,
  matcherName: string
): "bruteforce" | "lightglue" {
  const inferred = matcherName.endsWith("_BRUTEFORCE")
    ? "bruteforce"
    : matcherName.endsWith("_LIGHTGLUE")
      ? "lightglue"
      : null;
  if (inferred === null) {
    throw new Error("local matcher name is invalid");
  }
  if (value === undefined) {
    return inferred;
  }
  if (value !== "bruteforce" && value !== "lightglue") {
    throw new Error("local matcher profile is invalid");
  }
  if (value !== inferred) {
    throw new Error("local matcher profile and name are inconsistent");
  }
  return value;
}

function parseAlgorithm(value: unknown, label: string) {
  const record = object(value, label);
  return {
    name: text(record.name, `${label} name`),
    implementation: text(record.implementation, `${label} implementation`),
    version: text(record.version, `${label} version`)
  };
}

function parseImage(
  value: unknown,
  schemaVersion: 1 | 2 | 3 | 4
): SfmImage {
  const image = object(value, "SfM image");
  const registered = boolean(image.registered, "registered");
  const split = image.split;
  if (split !== null && split !== "train" && split !== "validation" && split !== "test") {
    throw new Error("SfM image split is invalid");
  }
  const center = optionalVec3(image.center_normalized, "camera center");
  const forward = optionalVec3(image.forward_normalized, "camera forward");
  const up = optionalVec3(image.up_normalized, "camera up");
  const horizontalFov = optionalPositiveNumber(image.horizontal_fov_degrees, "horizontal FOV");
  const verticalFov = optionalPositiveNumber(image.vertical_fov_degrees, "vertical FOV");
  if (registered !== (center !== null && forward !== null && up !== null && horizontalFov !== null && verticalFov !== null)) {
    throw new Error("SfM image registration and pose are inconsistent");
  }
  return {
    frame_uid: sha256(image.frame_uid, "frame UID"),
    colmap_image_id: positiveInteger(image.colmap_image_id, "COLMAP image ID"),
    camera_id:
      schemaVersion === 4
        ? positiveInteger(image.camera_id, "COLMAP camera ID")
        : null,
    name: text(image.name, "image name"),
    path: assetPath(image.path),
    sha256: sha256(image.sha256, "image SHA-256"),
    width: positiveInteger(image.width, "image width"),
    height: positiveInteger(image.height, "image height"),
    registered,
    split,
    source_time_seconds: optionalNonNegativeNumber(image.source_time_seconds, "source time"),
    feature_count: nonNegativeInteger(image.feature_count, "feature count"),
    center_normalized: center,
    forward_normalized: forward,
    up_normalized: up,
    horizontal_fov_degrees: horizontalFov,
    vertical_fov_degrees: verticalFov
  };
}

function assetPath(value: unknown): string {
  const path = text(value, "asset path");
  const segments = path.replaceAll("\\", "/").split("/");
  if (
    path.startsWith("/") ||
    segments.some((segment) => !segment || segment === "." || segment === "..") ||
    path.includes("?") ||
    path.includes("#")
  ) {
    throw new Error("Asset path is unsafe");
  }
  return segments.join("/");
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || !value) {
    throw new Error(`${label} must be non-empty text`);
  }
  return value;
}

function boolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be boolean`);
  }
  return value;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
}

function positiveInteger(value: unknown, label: string): number {
  const number = finite(value, label);
  if (!Number.isInteger(number) || number <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return number;
}

function nonNegativeNumber(value: unknown, label: string): number {
  const number = finite(value, label);
  if (number < 0) {
    throw new Error(`${label} must not be negative`);
  }
  return number;
}

function ratio(value: unknown, label: string): number {
  const number = finite(value, label);
  if (number < 0 || number > 1) {
    throw new Error(`${label} must be between zero and one`);
  }
  return number;
}

function nonNegativeInteger(value: unknown, label: string): number {
  const number = finite(value, label);
  if (!Number.isInteger(number) || number < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return number;
}

function optionalNonNegativeNumber(value: unknown, label: string): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const number = finite(value, label);
  if (number < 0) {
    throw new Error(`${label} must not be negative`);
  }
  return number;
}

function optionalPositiveNumber(value: unknown, label: string): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const number = finite(value, label);
  if (number <= 0 || number >= 180) {
    throw new Error(`${label} is outside its valid range`);
  }
  return number;
}

function coordinatePair(value: unknown, label: string): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error(`${label} must have two coordinates`);
  }
  return [finite(value[0], label), finite(value[1], label)];
}

function integerPair(value: unknown, label: string): [number, number] {
  const pair = coordinatePair(value, label);
  if (!pair.every((item) => Number.isInteger(item) && item >= 0)) {
    throw new Error(`${label} indices must be non-negative integers`);
  }
  return pair;
}

function optionalVec3(value: unknown, label: string): Vec3 | null {
  if (value === undefined || value === null) {
    return null;
  }
  return finiteVec3(value, label);
}

function finiteVec3(value: unknown, label: string): Vec3 {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${label} must be a 3-vector`);
  }
  return value.map((entry) => finite(entry, label)) as Vec3;
}

function optionalSha256(value: unknown, label: string): string | null {
  if (value === undefined || value === null) {
    return null;
  }
  return sha256(value, label);
}

function sha256(value: unknown, label: string): string {
  const digest = text(value, label);
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`${label} is invalid`);
  }
  return digest;
}

function unit(vector: Vec3): Vec3 {
  const norm = length(vector);
  if (norm <= 1e-12) {
    throw new Error("Direction vector cannot be zero");
  }
  return vector.map((value) => value / norm) as Vec3;
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function length(vector: Vec3): number {
  return Math.hypot(...vector);
}

function dot(left: Vec3, right: Vec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function radiansToDegrees(value: number): number {
  return (value * 180) / Math.PI;
}
