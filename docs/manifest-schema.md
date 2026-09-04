# Manifest schema

`outputs/jobs/{job_id}/manifest.json` is the stable interface between the backend and frontend. Paths in the manifest are relative to the job directory; consumers must not depend on model-internal directories that are not listed here.

## Top-level fields

Newly completed jobs contain:

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Local job identifier. |
| `status` | string | Current terminal value is `done`. |
| `stage` | string | Last completed pipeline stage. |
| `progress` | number | Completion fraction. |
| `mode` | string | Requested input mode. |
| `input_type` | string | Normalized input type; panoramas use `equirectangular_panorama`. |
| `geometry_backend` | string | Effective geometry backend. |
| `output_type` | string | Requested geometry asset type. |
| `created_at` | string | UTC creation timestamp. |
| `inputs` | array | Stored input filename, relative path, content type, byte count, and SHA-256. |
| `assets` | object | Available output assets keyed by stable role. |
| `metrics` | object | Backend and postprocess diagnostics suitable for display and audit. |
| `sfm_feature_profile` | string | Optional requested COLMAP feature profile for `colmap`, `colmap_vggt`, and `project_3dgs`: stable `sift_v1` or experimental `aliked_n16rot_v1`. Historical manifests may omit it. |
| `sfm_feature_effective_profile` | string or null | Feature profile actually used by a completed COLMAP feature stage; queued jobs use `null`. Missing learned assets never trigger a silent SIFT fallback. |
| `sfm_local_matcher` | string | Optional requested local descriptor matcher: stable `bruteforce` or experimental `lightglue`. Pairing remains a separate policy. Historical manifests that omit this field used brute-force. |
| `sfm_local_matcher_effective` | string or null | Local matcher profile actually used by the completed COLMAP matching stage; queued jobs use `null`. Missing LightGlue assets never trigger a silent brute-force fallback. |
| `sfm_pairing` | string | Optional requested image-pair policy: stable `exhaustive`, video-only experimental `sequential_loop`, or multi-image experimental `vocab_tree`. Historical manifests may omit it. |
| `sfm_pairing_effective` | string or null | Pairing profile actually used by the completed COLMAP matching stage; queued jobs use `null`. Missing or incompatible vocabulary trees never trigger an exhaustive fallback. |
| `sfm_geometric_verification` | string | Optional requested two-view geometry profile: stable `default_v1` or experimental `guided_v1`. Historical manifests that omit it used the default COLMAP verification path. |
| `sfm_geometric_verification_effective` | string or null | Geometric-verification profile actually used by completed matching, including standard-v2 recovery matching; queued jobs use `null`. Unsupported Guided never falls back silently. |
| `sfm_camera_calibration` | string | Optional requested camera model/sharing profile: `shared_opencv_v1`, `shared_simple_radial_v1`, or multi-image-only experimental `auto_grouped_simple_radial_v1`. Omitted requests preserve backend history: Project Gaussian used shared OPENCV; direct COLMAP and COLMAP+VGGT used shared SIMPLE_RADIAL. |
| `sfm_camera_calibration_effective` | string or null | Profile validated from the completed raw sparse model's camera diagnostics; queued jobs use `null`. Unsupported combinations fail rather than silently changing camera model or sharing. |
| `gaussian_config` | object | Optional resolved 3DGS configuration provenance for jobs that have explicitly entered the project-owned Gaussian lifecycle. |
| `gaussian_trainer` | object | Optional selected trainer identity, label, pinned revision, and license. Current IDs are `graphdeco`, `project`, and experimental native `mcmc`; historical manifests may omit it. |
| `gaussian_geometry_source` | string | Optional requested Gaussian geometry source: stable `colmap` or experimental video-only `vggt_ba`. Historical Gaussian jobs may omit it. |
| `gaussian_geometry_effective_source` | string or null | Geometry actually used by a completed Gaussian job: `colmap` or `vggt_ba`; queued jobs use `null`. Historical jobs may omit it rather than receiving a synthetic value. |
| `gaussian_geometry_fallback_applied` | boolean | Whether a requested `vggt_ba` job completed using the explicit ordinary-COLMAP fallback. |
| `gaussian_geometry_fallback_reason` | string or null | One allow-listed geometry-quality reason when fallback occurred; otherwise `null`. |
| `gaussian_postprocess` | string | Optional requested Gaussian derivative: `none` or experimental `vggt_visibility_v1`. |
| `gaussian_postprocess_status` | string | Optional derivative lifecycle: `pending`, `not_requested`, `available`, or `unavailable`. Failure is fail-soft and does not invalidate Original Gaussian assets. |
| `gaussian_postprocess_reason` | string or null | Optional diagnostic reason when the requested derivative is unavailable. |
| `gaussian_sor_filter` | string | Optional SOR floater-cleanup request for project Gaussian jobs: `on` (default) or `off`. Absent on historical manifests, which never ran the filter. |
| `gaussian_sor_filter_status` | string | Optional SOR lifecycle: `pending`, `disabled`, `available`, or `unavailable`. Failure is fail-soft and the unfiltered model remains a successful result. |
| `gaussian_sor_filter_reason` | string or null | Optional diagnostic reason when the requested SOR cleanup is unavailable. |
| `gaussian_recovery_prune` | string | Optional experimental recovery-gated prune request for project Gaussian jobs: `on` or `off` (default). When `on`, the effective Gaussian config enables `opacity_reset.recovery_prune`. Absent on historical manifests. |
| `navigation_status` | string | Optional Gaussian-navigation lifecycle: `pending`, `not_generated`, `queued`, `generating`, `available`, or `unavailable`. Navigation failure never changes a completed Gaussian job from `done`. |
| `navigation_reason` | string or null | Stable machine-readable reason when navigation is unavailable. |
| `navigation_details` | object or null | Hashes, budgets, coordinate semantics, and Train-only publication evidence for an available navigation asset set. |
| `mesh_variants` | array | Optional generated mesh alternatives. |

Historical manifests can omit fields added after that job was created. Readers must tolerate absent optional fields and must not manufacture an effective policy that the old job did not record.

## Local lifecycle schema v1

R2.6 submissions add `lifecycle_schema_version: 1` and are atomically persisted before execution. Their status transitions are:

```text
queued -> running -> done
                  -> failed
       -> cancelled
failed/cancelled -> queued (new bounded retry attempt)
```

Lifecycle manifests additionally record `updated_at`, `queued_at`, nullable `started_at`/`completed_at`/`cancel_requested_at`, `active_attempt_id`, an `attempts` history, and nullable structured `error: {code, message}`. Attempt history records immutable IDs, `fresh` or `retry` kind, parent attempt, timestamps, status, and error. A retry starts clean, never loads checkpoint state, and is capped at three total attempts.

`POST /api/jobs` returns HTTP 202 with the durable `queued` manifest. One local filesystem worker processes jobs FIFO and holds a filesystem lease, so only one GPU-heavy job runs for an output root. Queued work remains queued across API restart. A stale `running`/`exporting` job discovered at worker startup becomes explicit `failed` with `worker_interrupted` (or `cancelled` when a cancellation request was already durable); it is never inferred to be successful.

A running attempt writes under `lifecycle/attempts/{attempt_id}/workspace`. Complete outputs are moved to stable job paths before the terminal `done` manifest is published. Failed/cancelled partial output is retained under the attempt for diagnostics but is not listed in `assets`. Inputs and completed diagnostics are not deleted by cancellation. R2.6 attempt history is lifecycle metadata; R2.5 checkpoint descriptors remain separate and are created only when a trainer has real dataset/config/code/environment provenance.

Legacy terminal-only manifests remain valid and readable. They do not gain synthetic attempt history and cannot use the R2.6 retry operation.

## Gaussian effective configuration

`gaussian_config` is present only when a trusted project-owned 3DGS caller supplies a resolved configuration. It contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Gaussian configuration schema version; new Project/MCMC jobs use version `10`, while historical versions remain immutable evidence. |
| `requested_profile` | string | Versioned profile selected before resolution; `standard_v1` is public and `mcmc_v1` is selected only through the trusted `gaussian_trainer=mcmc` method identity. |
| `effective_config` | object | Complete validated training configuration after trusted internal overrides. It is authoritative over request fields or environment variables. |
| `effective_config_hash` | string | SHA-256 of `effective_config` serialized as sorted, compact JSON. The requested profile is provenance and is not part of this hash. |

The same schema version, requested profile, effective configuration, and hash are written to `logs/run.log`. Public job submission does not expose raw Gaussian hyperparameters; a future Gaussian-specific public route may select only an allow-listed quality profile. Internal research callers may supply validated overrides, and a recorded ablation must differ from its baseline in exactly one effective leaf field.

Training configuration contains Validation cadence but no Test cadence. Schema v10 adds an explicit strategy, method-specific initialization/loss/LR fields, and an optional global Gaussian cap while preserving all `standard_v1/default_v1` numerical behavior. `mcmc_v1` is a frozen experimental method package: gsplat `MCMCStrategy`, 30,000 iterations, initial opacity `0.5`, frozen 3NN scale multiplied by `0.1`, opacity/scale regularization `0.01`, position-LR delay `0.01`, opacity LR `0.05`, relocation/growth from iteration 500 through 25,000 every 100 iterations, and a 3,000,000-Gaussian cap across all ranks. It disables Default pruning, opacity reset, and recovery-prune; requesting `gaussian_recovery_prune=on` with `mcmc` is rejected. Raw MCMC tuning leaves are not public form controls. New Project and MCMC frontend jobs export the Validation-selected model without loading Test; `gaussian_test_evaluation` and `gaussian_test_decision` remain absent unless a separately authorized frozen-candidate evaluation produces them. MCMC is runnable but remains experimental pending remote quality/resource evidence; Graphdeco remains the default.

Jobs without `gaussian_config`, including all historical geometry jobs and imported-splat fixtures, remain valid. Readers must not add a default profile or infer effective parameters for them.

## Assets

Common asset roles include:

- `point_cloud`, `point_cloud_aligned`, and `cameras`
- `sfm_sparse_point_cloud` and `sfm_diagnostics` for Project Gaussian jobs that publish the final accepted COLMAP sparse PLY/cameras and frontend feature/match diagnostics
- `sfm_pose_health` for every new Project Gaussian geometry result, plus `sfm_pose_recovery` for ordinary-COLMAP results; these preserve the pre-Gaussian pose gate and any bounded recovery provenance
- `sfm_camera_calibration_diagnostics` for every current COLMAP-backed reconstruction; it points to `diagnostics/sfm_camera_calibration.json`, derived from the raw sparse model before Gaussian image undistortion
- `alignment_diagnostics`, `fusion_diagnostics`, `visibility_graph`, `scale_disagreement_diagnostics`, and `consistency_diagnostics`
- `mesh` and `mesh_diagnostics`
- `scene_splat`, `scene_graph`, and `log`
- `gaussian_raw_model`, `gaussian_model`, `gaussian_training_result`, `gaussian_progress`, and `gaussian_dataset` for complete project-owned training jobs; the raw role is the immutable Validation-selected trainer output before SOR, while `gaussian_model` is the model passed to common evaluation/export
- `gaussian_replay_dataset` and `gaussian_replay_record` for a self-contained frozen trainer input rooted at `gaussian/replay/`; it preserves dataset/image/camera/initialization hashes and includes the registered undistorted images without publishing the COLMAP database or matches
- `gaussian_evaluation`, `gaussian_export_metadata`, `gaussian_canonical`, `gaussian_camera_path`, and `gaussian_bundle` for complete Stage 2D delivery; optional `gaussian_test_evaluation` and `gaussian_test_decision` appear only after an authorized frozen-candidate Test evaluation
- `collision_mesh`, `navigation`, and `navigation_diagnostics` for a complete Train-only first-person navigation set
- `video_probe`, `video_frame_selection`, `video_keyframe_timing`, `video_registration_diagnostics`, and `video_keyframe_contact_sheet` for a completed bounded video attempt; ordinary-COLMAP `standard_v2` attempts additionally publish `video_initial_registration_expansion`, `video_registration_recovery`, and `colmap_timing`
- `vggt_ba_diagnostics` and `vggt_ba_window_graph` for a completed experimental VGGT-BA attempt, including one that used the explicit COLMAP fallback; `vggt_ba_initialization_diagnostics` appears only when VGGT-BA remained the effective source
- `gaussian_vggt_filtered_model`, `gaussian_vggt_filter_diagnostics`, `gaussian_vggt_filter_mask`, `gaussian_vggt_filtered_evaluation`, `gaussian_vggt_filtered_export_metadata`, `gaussian_vggt_filtered_canonical`, `scene_splat_vggt_filtered`, and `gaussian_vggt_filtered_bundle` for a complete experimental filtered derivative

`collision_mesh` is a low-poly invisible local-physics GLB, not the customer-visible `mesh` or `scene_splat`. `navigation` is the versioned normalized-coordinate boundary/spawn/player contract; `navigation_diagnostics` is its Train-only quality record. The three roles are published together only after source path containment, source/model/config/export hashes, split isolation, schema, topology, triangle/size/time budgets, and GLB integrity pass. Validation/Test IDs must be empty and selected render IDs must be a subset of Train.

New Gaussian jobs attempt navigation after export. A failed attempt records `navigation_status: unavailable` and a stable reason while the Gaussian job remains `status: done` with `scene_splat`. `POST /api/jobs/{job_id}/navigation-assets` queues generation for an old successful Gaussian job, is idempotent while queued/generating/available, never retrains, and runs through the same single-worker filesystem lease. Generation writes under `lifecycle/navigation/attempt-NNN/workspace`; one directory rename publishes the complete set, while cancellation/interruption preserves partial output outside stable asset paths. A retry increments `navigation_attempt` and does not overwrite a published set.

Navigation assets and their navigation lifecycle diagnostics are excluded from the existing job download ZIP. The ZIP contains a sanitized manifest without navigation roles/status so it cannot reference omitted files. This preserves the existing delivery contract until explicitly revised; direct manifest asset URLs remain the product path.

The desktop Gaussian viewer enables Walk only when `scene_splat`, `collision_mesh`, and `navigation` are all present and the navigation set passes client-side schema, Train-only, normalized/arbitrary-unit, range, byte-count, SHA-256, triangle-count, and spawn checks. It uses the collision GLB only to construct a Three.js `Octree`; the mesh is hidden by default and the Gaussian splat remains the customer-visible scene. Missing, generating, unavailable, or invalid navigation leaves Orbit mode usable. Completed old Gaussian jobs expose the idempotent generation action in the frontend, while Walk provides Pointer Lock, WASD/arrows, hard-boundary enforcement, safe-position preservation, and spawn reset. Player dimensions and speed are relative to estimated eye height `H`, not metres.

`scene_splat` is the versioned browser derivative; `gaussian_canonical` is the project-owned deterministic PLY contract. The Gaussian dataset/evaluation/export roles are per-attempt hash-bound records. Incomplete or failed training/evaluation/export files must not be added to `assets`.

Only roles present in `assets` are available. Generic postprocessing can add existing alignment or mesh assets when an older manifest is loaded; this does not rerun reconstruction.

## SfM frontend diagnostics

New `project_3dgs + gaussian_splat` jobs publish the final accepted sparse model as `sfm_sparse_point_cloud` (`geometry/points.ply`) and its camera source as `cameras`. The dedicated role keeps raw `colmap_world` arbitrary-unit geometry independently addressable, while generic postprocessing also aligns that same sparse cloud into `point_cloud_aligned` with `alignment_diagnostics`; the frontend defaults to aligned geometry but preserves Raw/Aligned and display-axis controls. Because a fitted plane normal has an ambiguous sign, viewers use transformed camera-up evidence to choose the initial ±Z display orientation. Nearest-input-view records are separately transformed into the Gaussian normalized frame and never claim metric scale.

After the final Mapper/registration/recovery/BA result and dataset split are frozen, the adapter attempts a fail-soft frontend export from that same workspace's COLMAP database. Success publishes `sfm_diagnostics=diagnostics/sfm/manifest.json` and `sfm_diagnostics_status=available`; failure omits the role, records `sfm_diagnostics_status=unavailable` plus `sfm_diagnostics_reason`, and does not fail Gaussian training. Data from failed/replaced `lifecycle/partial` attempts is never combined with the accepted model. Historical manifests may omit all these roles and metrics.

`sfm_diagnostics` schema 4 (`profile=sfm_frontend_diagnostics_v4`) records:

- normalized OpenCV camera axes and arbitrary-unit semantics;
- dataset hash, default run ID, one or more run records, and aggregate image/keypoint/pair/tentative/candidate-inlier/Guided-inlier/final-inlier/outlier counts;
- every feature-extracted source image's stable frame UID, job-relative path/hash/dimensions, COLMAP image and camera IDs, optional source video time, final registration/split state, and—only when registered—normalized center/forward/up and horizontal/vertical FOV;
- per-run `feature` (profile/extractor/descriptor/keypoint budget/extractor-model hash), `local_matcher` (stable `bruteforce|lightglue` profile, descriptor-compatible COLMAP enum and optional model hash), `pairing` (stable profile and optional descriptor-compatible vocabulary-tree SHA), `geometric_verification` (`default_v1|guided_v1`, Guided flag, verification-not-skipped, raw-parameter policy), `camera_calibration` (validated profile/model/sharing policy, planned/initial/final/prior-focal/warning counts, implementation/build, and diagnostics path), Mapper provenance, plus job-relative feature-index and pair-index paths;
- per-run `view_graph` schema 1 (`profile=sfm_verified_view_graph_v1`): nodes are every database image and a verified edge is exactly a non-empty COLMAP `two_view_geometries` correspondence set. It reports tested/candidate/verified pair totals, match totals, geometric-config counts, degree/component/distribution summaries, largest-component registered membership, isolated/degree-one nodes, and optional video edge-span plus direct soft-gap bridge evidence. These are diagnostics, not reconstruction gates.

Schemas 1, 2, and 3 remain readable. Schema 1's historical `detector=sift` plus `matcher=exhaustive|sequential` is interpreted as SIFT + `SIFT_BRUTEFORCE`, with the old matcher field representing pairing. Missing geometric-verification provenance in schema 1/2 means historical `default_v1`; missing camera provenance in schema 1/2/3 means the historical Project Gaussian shared-OPENCV path. Their View Graph summary can be derived read-only from the retained pair index. Feature shards remain schema 1.

Schema 4 pair indexes/details continue to use shard schema 2 because camera grouping does not change keypoint/correspondence semantics and Guided Matching may add verified correspondences that are absent from the original tentative `matches` table. Pair indexes retain `candidate_match_count` and final `inlier_count`, and add `candidate_inlier_count`, `guided_inlier_count`, and `outlier_count`. Pair details store every final verified correspondence in `inliers`; `outliers` contains only rejected tentative candidates. Therefore candidate survival is `candidate_inlier_count / candidate_match_count`; final verified count must not be divided by tentative count when Guided additions exist. Historical pair shard schema 1 remains readable and implies `candidate_inlier_count=inlier_count`, `guided_inlier_count=0`, and `outlier_count=candidate_match_count-inlier_count`.

The separate `sfm_camera_calibration_diagnostics` asset uses schema 1 / `sfm_camera_calibration_diagnostics_v1`. It is generated from the final raw sparse cameras before `image_undistorter`, so a later Gaussian PINHOLE dataset camera never replaces the original OPENCV/SIMPLE_RADIAL evidence. It records requested/effective profile, grouping policy and privacy-bounded grouping evidence, initial database cameras/images and prior-focal flags, final registered camera groups/rates, named focal/principal-point/distortion values, initial-to-final focal change, sparse observation/track/reprojection summaries, and COLMAP build. Non-finite or non-positive focal values, profile/model mismatch, shared-profile splitting, planned/database partition drift, or final image-to-camera reassignment are contract errors. Focal-ratio, extra-distortion, and principal-point warnings are explicitly soft (`warnings_are_job_gates=false`).

`sfm_pose_health` uses schema 1 / `sfm_pose_health_v1` and is evaluated on the final raw COLMAP text model before undistortion, split construction, Gaussian normalization, or CUDA. It records scale-invariant camera-center distributions and isolated/branch ratios, timestamp-normalized translation and rotation-step evidence, registration timeline, observation/depth support, bounded covisibility components, outlier cameras, and healthy-core↔outlier bridge pairs with final-track and optional database match/verified-inlier counts. Hard reasons are limited to catastrophic pose geometry (`isolated_camera_pose_outlier`, `multiscale_camera_pose_branch`, or `degenerate_camera_extent`); coordinates and depths remain arbitrary world units. A Project Gaussian geometry result is accepted only when the final report is `passed`. Test images/RGB are not loaded or used.

Ordinary-COLMAP jobs also publish schema 1 / `sfm_pose_recovery_v1`. Every incremental Mapper model is checked before selection. If none is healthy, the runner copies the SQLite database, runs `view_graph_calibrator` and calibrated Global Mapper with the same feature/match/pairing/calibration identity, and accepts it only after the same pose and 12-frame/70%-registration/80%-temporal-coverage gates. Only when Global recovery fails may a timestamped video remove one complete catastrophic branch, capped at 10% of registered cameras, then run point filtering and bundle adjustment before all gates are reapplied. `effective_mapper` is `incremental`, `global_recovery_v1`, or `incremental_core_repair_v1`; the record contains job-relative candidate/model/database paths, sparse-model file hashes, source/effective database hashes, removed image IDs, and failure-safe provenance. Recovery never silently switches to SIFT, Brute-force, another pairing policy, or another camera profile. A Global/core result is explicitly recovered evidence, not a clean incremental or ALIKED/LightGlue pass.

`scripts/run_colmap_sparse.py` writes `diagnostics/sfm_frontend_contract.json` (`sfm_frontend_contract_v1`) before Mapper execution so a pose-failed standalone arm still retains its frozen COLMAP build, feature, local matcher, pairing, geometric verification, camera profile, video-selection hash, COLMAP default random seed `0`, Mapper policy, and v2 seed contract. `scripts/evaluate_sfm_frontend_factorial.py` consumes four existing geometry-only arms (SIFT/ALIKED × Brute-force/LightGlue), rejects mismatched contracts, and reports only bounded extractor/matcher/interaction/common-pipeline or solver-sensitivity evidence. It never runs geometry or training.

Large run children remain deterministic gzip-compressed JSON shards. Their filenames end in `.json.gz`; the asset endpoint returns them as `application/json` with `Content-Encoding: gzip`, so browser consumers use ordinary JSON decoding. Feature records contain only rounded `(x,y)` coordinates in original upright feature-input pixels. A pair missing from the index means it was not represented in the retained COLMAP match/geometry tables, while an indexed pair with zero counts has no retained correspondence. COLMAP descriptors, the mutable database, and failed-attempt internals are not published. Pixel mosaic stitching and final SfM 3D-track observations remain outside this contract.

Metrics on success include the existing diagnostics/image/feature/local-matcher/pairing/Mapper fields plus `sfm_geometric_verification_profile`, `sfm_camera_calibration_profile`, `sfm_camera_model`, planned/initial/final/prior-focal camera counts, camera warning count, median focal-length ratio, optional median reprojection error/track length, `sfm_view_graph_verified_edge_count`, `sfm_view_graph_component_count`, `sfm_view_graph_largest_component_ratio`, `sfm_view_graph_isolated_node_count`, and `sfm_view_graph_guided_inlier_count`. New Project Gaussian results also expose `sfm_pose_health_status`; ordinary-COLMAP results add `sfm_effective_mapper`, `sfm_pose_recovery_status`, `sfm_pose_recovery_applied`, and `sfm_pose_recovery_removed_camera_count`. Full pose, camera, and graph distributions stay in diagnostics assets rather than being duplicated into top-level metrics.

## Bounded video contract

`mode=video` is implemented only with `project_3dgs + gaussian_splat`. The API accepts `video_keyframe_profile=standard_v1|standard_v2` and defaults new requests to v2; the frontend also submits v2. Both profiles accept one MP4/MOV/M4V/WebM, 10–600 seconds (606 seconds is the technical container-tolerance limit), at most 2 GiB, and analyze no more than 3,636 candidates at 6 fps. Default v2 selection schema 2 records an immutable 4 fps uniform base, deterministic sparse Lucas–Kanade motion/descriptor fallback, at most one adaptive frame per second up to 5 fps, stable candidate-index/source-PTS filenames, and base/adaptive selection reasons and counts. Historical v1 remains explicitly selectable and selects at most 1,000 keyframes. Upload staging uses bounded chunks; the original video remains under `input/` and retry deterministically regenerates keyframes from it.

`sfm_pairing` defaults to `exhaustive`. The experimental video-only `sequential_loop` profile applies to both ordinary COLMAP and `vggt_ba` geometry, uses COLMAP sequential matching with loop detection, and resolves the vocabulary tree for the selected descriptor: the pinned 256K-word SIFT tree or 64K-word ALIKED N16Rot tree. The multi-image-only `vocab_tree` profile uses the same descriptor-compatible assets with `vocab_tree_matcher`. Missing, corrupt, mode-incompatible, or descriptor-incompatible trees fail before geometry instead of falling back to exhaustive or reusing the SIFT tree for ALIKED. Legacy `colmap_matcher=exhaustive|sequential` remains readable and maps to `exhaustive|sequential_loop`; conflicting old/new fields fail. For v2, sequential overlap is `clamp(ceil(effective_fps × 4), 16, 24)` so initial matching covers approximately four seconds. To bound large-sequence Mapper cost, v2 raises intermediate global-BA growth triggers to 1.5, raises frame/point frequencies to 1,000/1,000,000, and limits each global refinement cycle to one; final BA remains enabled. VGGT-BA seeded registration and any classified ordinary-COLMAP fallback reuse the same feature database, pairing policy, and v2 Mapper policy.

Video Project jobs historically default to `shared_opencv_v1` and may explicitly select `shared_simple_radial_v1`; both keep one shared camera. `auto_grouped_simple_radial_v1` is rejected for video and VGGT-BA. Ordinary standard-v2 recovery uses `ImageReader.existing_camera_id` for newly materialized frames, so it extends the original shared group rather than rerunning EXIF grouping or creating a new camera.

`video_probe` records the normalized ffprobe result, source hash, source/display dimensions, and applied quarter-turn without exposing location values. `video_frame_selection` is the authoritative final source-PTS, quality, rejection, output hash, dimensions, minimal generated-EXIF, and selection-reason record; accepted v2 recovery frames are atomically added to this same sidecar before final undistortion and dataset construction. `video_keyframe_timing` records the complete probe/analyze/materialize elapsed time and profile; `video_keyframe_contact_sheet` is display-only. For ordinary-COLMAP v2, Mapper starts from at most 1,000 uniformly spaced `base` frames while feature extraction and matching still cover the complete selected set. `video_initial_registration_expansion` records up to two subsequent selected-frame `image_registrator` + non-clearing `point_triangulator` passes, including registered-camera and sparse-point retention. `colmap_timing` records ordinary-COLMAP feature extraction, matching, seed mapping, initial registration expansion, gap recovery, undistortion, conversions, seed count, frozen v2 Mapper options, requested/effective Mapper, effective database path/hash, pose-evidence paths, frontend-contract path, and total elapsed time. `video_registration_diagnostics` maps the final selected source timestamps to final COLMAP registrations and records registration rate, temporal coverage, largest gap, and `gap_violations`—adjacent registered-frame intervals exceeding `maximum_registered_gap_threshold_seconds` (2 seconds). Gap violations are a soft warning: the job continues, and when present the manifest publishes their count/total/excess and `run.log` records the intervals.

For v2, `video_registration_recovery` is schema 1 and records the frozen incremental-COLMAP policy, the effective SfM feature/local-matcher/geometric-verification identities, the initial stable `sfm_pairing`, and its own `bounded_temporal_pair_list` recovery-pairing identity. Recovery does not pretend to rerun exhaustive/sequential-loop/vocab retrieval: it constructs only bounded local pairs for newly materialized gap frames. The record also includes initial/final selected and registered counts, registered-camera retention, sparse-point counts, pre/post gap count/total/excess, up to two registration rounds, candidates/materialized frames/local pair counts, per-substage elapsed time, acceptance/rejection reasons, one final-BA record, and fail-soft status. Recovery feature extraction and `matches_importer` receive the same extractor, descriptor-compatible local-matcher, and `default_v1|guided_v1` verification options as the initial database; they never fall back to COLMAP's SIFT/brute-force/default-verification behavior. Each round adds at most 25% of the initial selection and all rounds add at most 50%; only candidates in a gap plus two-second bridge margins are eligible. The first round extracts/matches only new local evidence. Every round runs `image_registrator`; after an accepted triangulation, the second round may run in `propagation` mode with zero new candidates so the new 3D points can register frames deeper inside the same gap. A round proceeds to non-clearing triangulation only when registration strictly improves gap count, maximum gap, or total gap excess; registering only out-of-gap cameras is not sufficient. Accepted rounds lose no existing registered camera, retain at least 90% of sparse points, and continue to pass the 12/70%/80% product gates. Initial registration expansion and accepted recovery rounds share one final bundle adjustment, preferring CUDA and retrying once on CPU when CUDA BA is unavailable. Recovery command/model failure preserves the previously accepted model and selection and continues to final soft-gap diagnostics.

Completed v2 metrics include `video_keyframe_elapsed_seconds`, `colmap_geometry_elapsed_seconds`, the serialized COLMAP stage timings, initial-registration expansion status/pass count/registered gain, `video_initial_selected_count`, `video_base_selected_count`, `video_adaptive_selected_count`, `video_recovery_selected_count`, recovery status/reason/attempted and accepted round counts, pair/time totals, final-BA timing/fallback state, registered gain, and pre/post timeline and sparse-point summaries. Existing `video_selected_count` and `video_registration_*` fields always describe the final geometry input and final model. Historical jobs and v1 jobs may omit v2-only fields.

`scripts/evaluate_video_v2_promotion.py` retains the previously frozen strict geometry evaluator. It requires same-source v1 baseline timing, two independent v2 selection records, v2 initial-expansion/timing/recovery diagnostics, and exits nonzero unless selector evidence is identical, registration is at least 95%, every registered gap is at most two seconds, no Mapper-registered or expansion-registered camera is lost, at least 90% of the Mapper-input sparse points survive the complete expansion/recovery pipeline, recovery stays within two rounds/50%, and extraction plus COLMAP time is at most twice the v1 baseline. These remain available as stricter diagnostics, but the user-directed 2026-09-03 default switch no longer makes them product default blockers; it does not retroactively mark earlier evidence as passing.

Generated keyframe JPEGs are physically upright and use EXIF Orientation `1` plus a Software tag. Container creation time/device tags are not fabricated as per-frame photographic EXIF; focal length, ISO, shutter, GPS, and original-photo timestamps are not invented. Estimated camera parameters remain in `geometry/cameras.json`. COLMAP must register at least 12 selected frames, at least 70% of selected frames, and at least 80% of their selected temporal span before Gaussian training starts.

Registered video frames are assigned to deterministic two-second temporal groups. A group belongs wholly to Train, Validation, or Test; the trainer still loads only Train and Validation. Held-out groups are never chosen from the groups adjacent to a registration gap above the 2-second threshold, so holdouts do not compound registration holes; if that avoidance leaves too few eligible groups, dataset contract construction fails. Video extraction, registration gates, navigation, and model selection cannot load Test, and video ingestion does not create a `*.test-consumed.json` record.

This contract is bounded offline reconstruction, not realtime SLAM, guaranteed loop closure, metric scale, or a drift-free long-horizon claim. Missing FFmpeg/ffprobe marks only Project video ingestion unavailable; image jobs remain available.

## Experimental Gaussian geometry and filtered derivative

New `project_3dgs + gaussian_splat` requests may independently select:

```text
gaussian_geometry_source = colmap | vggt_ba
gaussian_postprocess = none | vggt_visibility_v1
```

Both request fields default to the historical behavior (`colmap` and `none`). `vggt_ba` is video-only and research-only. Its local windows classify a frame as strong at 32 or more reliable observations; weak frames do not enter local BA or Sim(3) edge evidence. Each adjacent disconnect receives at most one deterministic recovery window of no more than eight frames, with at least three reliable frames from each side. Existing bounded DINOv2 nonlocal bridge windows remain independent.

After recovery, only a connected component with at least 12 reliable cameras, at least 70% reliable-camera coverage, at least 80% index-span temporal coverage, finite Sim(3), and non-worsening pose-graph optimization may seed geometry. The seeded path is:

```text
partial VGGT camera model
-> COLMAP point_triangulator
-> COLMAP image_registrator
-> COLMAP bundle_adjuster
-> final registration gates
```

The selected feature profile, local matcher, and exhaustive/sequential-loop pairing run once for all selected keyframes. Ordinary COLMAP Mapper may reuse that database and those matches only for these three classified quality states:

```text
vggt_graph_unusable_after_recovery
vggt_seed_geometry_insufficient
vggt_registration_gate_failed
```

There is no broad exception fallback. OOM/CUDA, dependency or checkpoint, non-finite, corrupted-input, I/O, cancellation, unexpected-code, and unclassified subprocess failures remain failed Jobs. A fallback model must pass the same 12-camera/70%/80% gates or the Job fails. Missing verified nonlocal edges remains `open_trajectory_unverified`; it does not trigger fallback and is not loop-closure or bounded-drift evidence.

Completed manifests preserve `gaussian_geometry_source=vggt_ba` as the request and publish `gaussian_geometry_effective_source`, `gaussian_geometry_fallback_applied`, and the nullable allow-listed reason. Fallback Jobs remain viewable and retain VGGT diagnostics, but they are not successful VGGT-BA A/B evidence. Metrics also include profile, supported cameras/points, elapsed time, trajectory status, and verified nonlocal-edge count.

When VGGT-BA remains the effective source, the global camera/point solution may use all registered images, matching the existing camera-estimation contract. The Gaussian sparse initializer receives only points with at least two Train observations, recolored from Train observations. A COLMAP fallback follows ordinary-COLMAP initialization semantics instead. `vggt_ba_initialization_diagnostics` records accepted, heldout-only rejected, insufficient-Train-support, mixed-track, and recoloring counts only for effective VGGT-BA initialization. This does not authorize Validation/Test RGB in training.

`vggt_visibility_v1` is a fail-soft derivative after the immutable Original model has completed Validation and export. It recomputes VGGT depth from at most 64 Train images, aligns each usable depth map to final sparse geometry, and conservatively removes only multi-view free-space contradictions with no surface support or unsupported oversized Gaussians outside every Train capture envelope. The mask records row-aligned keep/reason/support arrays. It does not use Validation/Test to derive deletion decisions, fill holes, create walls, alter Original, retrain, or change navigation geometry.

When filtering succeeds, `gaussian_postprocess_status` is `available` and all filtered roles are published together. Original and filtered models receive separate Validation/export records and hashes; filtered bundles include `postprocess/diagnostics.json` and `postprocess/filter-mask.npz`. `scene_splat` remains Original for backward compatibility, while `scene_splat_vggt_filtered` is the optional Viewer A/B derivative. If filtering, filtered Validation, or filtered export fails, status is `unavailable`, a reason is recorded, no partial filtered roles enter `assets`, and Original remains a successful result.

Both capabilities remain experimental/research-only pending real CUDA jobs, cross-scene Validation, resource measurement, and license review. They retain normalized arbitrary units and make no metric-scale, complete-room, guaranteed-loop, or drift-free claim.

## Gaussian SOR floater cleanup (default pipeline step)

New `project_3dgs + gaussian_splat` jobs run a post-hoc statistical outlier removal (SOR) cleanup on the selected model snapshot before Validation evaluation and export. The request field defaults to the historical behavior plus cleanup:

```text
gaussian_sor_filter = on | off          (default on)
```

An operator may disable it per job through the API/frontend, or server-wide with `IMAGE3D_GAUSSIAN_SOR_FILTER=off` when the request does not override it. Only the Stage-1 render-lossless conservative band preset is used (`nb_neighbors=30`, `std_ratio=2.0`, `band_opacity=0.05`, 50% removal safety gate); no other tuning is exposed.

On success `gaussian_sor_filter_status` is `available`, the filtered snapshot replaces the training snapshot for Validation evaluation and export (so `scene_splat` is the cleaned model), and the export bundle records `postprocess/filter-record.json` and `postprocess/filter-mask.npz`. Asset roles `gaussian_sor_filter_record` and `gaussian_sor_filter_mask` point to the per-attempt record and mask. Metrics include `gaussian_sor_filter_input_count`, `gaussian_sor_filter_kept_count`, and `gaussian_sor_filter_removed_count`.

Failure is fail-soft: the status is `unavailable` with a reason, the unfiltered snapshot continues through Validation/export unchanged, and the job remains a successful result. Setting the field to `off` records status `disabled` and skips the filter entirely.

This step is independent of `gaussian_postprocess`. When both are requested, the SOR cleanup runs first on the selected snapshot, and any `vggt_visibility_v1` derivative is then derived from the SOR-filtered model; each lifecycle status is reported separately.

## Gaussian recovery-gated prune (experimental option)

```text
gaussian_recovery_prune = on | off      (default off)
```

When requested `on` (API form field, or server-wide `IMAGE3D_GAUSSIAN_RECOVERY_PRUNE=on` when the request does not override it), the resolved Gaussian config enables the schema-9 `opacity_reset.recovery_prune` leaf, and the project trainer fires a one-shot prune of gaussians still below the recovery threshold 500 iterations after each opacity reset (progress events carry `recovery_prune` with before/after counts). The request is echoed as the `gaussian_recovery_prune` manifest field, and the effective leaf is visible in the `gaussian_config` record. This is a research-only option: the room A/B passed its gates, but promotion to a default requires floater-rich video-job confirmation. No frontend switch exists yet.

## COLMAP+VGGT effective-policy metrics

A completed `colmap_vggt` job records effective runtime settings parsed from `logs/run.log`. Relevant fields include:

| Metric | Stable G1.26 fallback | Notes |
| --- | --- | --- |
| `fusion_mode` | `points` | `tsdf` remains experimental. |
| `vggt_batch_size` | `4` | Hardware-sensitive. |
| `vggt_grouping` | `sequential` | `covisibility` remains experimental. |
| `vggt_overlap_size` | `2` | Requested overlap. |
| `overlap_size` | `0` for sequential grouping | Effective consecutive overlap; may differ from the request. |
| `conf_percentile` | `50.0` | Percentile, not a metric confidence guarantee. |
| `confidence_threshold_scope` | `global` | `per_frame` remains experimental. |
| `consistency_support_policy` | `any_support` | `adaptive_two` remains experimental. |
| `max_points` | `2000000` | Final point-cloud cap only. |
| `point_budget_policy` | `random` | `spatial_balanced` remains experimental. |
| `point_budget_input_points` | run-dependent | Accepted points before the cap. |
| `point_budget_output_points` | run-dependent | Points written after the cap. |
| `point_budget_applied` | run-dependent | Whether point-budget policy actually ran. |

API form fields are optional. When omitted, the adapter resolves each setting from its environment variable and then its stable fallback. Therefore manifest metrics and runner diagnostics—not a request form alone—are authoritative for effective behavior. Historical jobs may lack some or all policy metrics and remain valid.

## Coordinates and scale

Raw geometry uses the reconstruction backend's source coordinate system. An aligned point cloud is a separate asset with its transform recorded in alignment diagnostics. Viewer axis flips are display-only.

COLMAP+VGGT world units are arbitrary unless an independently evaluated scale source is present. Gaussian navigation is likewise `coordinate_frame: normalized` and `world_units: arbitrary`; eye height `H`, capsule dimensions, movement speed, steps, and boundary coordinates are scene-relative rather than metres. The manifest and ETH3D's evaluation-time camera Sim(3) alignment do not establish true metres or metric accuracy.
