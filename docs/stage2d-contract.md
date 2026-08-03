# Stage 2D evaluation, export, and delivery contract

Stage 2D turns the project-owned Stage 2C model into evaluated and downloadable browser assets. It does not change the arbitrary-scale geometry claim and does not introduce an external trainer.

## Evaluation schema v1

`image3d_scenegraph.gaussian.evaluation` is shared by cadence validation and standalone evaluation. The trainer still calls `load_training_views`, which accepts only `train` and `validation`. The standalone evaluator accepts `validation` or `test`.

A held-out `test` run requires a frozen candidate record binding:

- candidate ID;
- dataset hash;
- effective-config hash;
- immutable model SHA-256;
- `selection_split: validation` and `test_metrics_seen: false`.

The evaluator atomically creates a sibling `*.test-consumed.json` with exclusive creation before loading test views. A candidate cannot be evaluated twice, and changed dataset/config/model hashes are rejected. Test output is terminal evidence only; no code path feeds it to training, checkpoint selection, or profile resolution.

Each result records per-view and mean/min/p10/p50/p90/max PSNR, SSIM, render milliseconds, aggregate FPS, failures, Gaussian count, opacity/scale distributions, official-strategy topology count changes/reset counts, normalized screen-radius and maximum-axis-scale health distributions, and peak CUDA allocation/reservation. Training images remain CPU-resident and only the active view is transferred to CUDA. The 30,000-step baseline publishes the Validation-selected model, while the final checkpoint remains lifecycle/resume evidence. Fixed rendered PNGs and JSONL rows are emitted. Rendering metrics are explicitly separate from geometry metrics.

LPIPS remains `not_run: pretrained_weight_license_and_hash_not_audited`. `lpips` is not installed. Upstream package code is BSD-2-Clause, but default AlexNet uses a torchvision ImageNet checkpoint downloaded at runtime and no approved local weight hash/license record exists. Stage 2D does not download it or substitute another metric.

## Canonical Gaussian export v1

`canonical.ply` is deterministic binary little-endian PLY in normalized coordinates and arbitrary units. Its vertex layout is:

1. `x y z`;
2. zero `nx ny nz` compatibility fields;
3. `f_dc_0..2`;
4. `f_rest_0..44`, ordered channel-major over SH bands 1–15;
5. opacity logit;
6. three natural-log scales;
7. normalized `wxyz` quaternion.

`scene.ply` is the versioned INRIA-v1 browser derivative. Today its bytes equal the canonical PLY; consumers must still use the separate role because future browser optimization cannot redefine the canonical format.

`export.json` records schema/format, coordinate frame, arbitrary units, camera axes, `world_from_normalized`, Gaussian/SH layout, checkpoint/model/dataset/config/evaluation hashes, browser opacity threshold/health summary, and media state. The browser reads its SH degree and alpha threshold from this metadata; missing `Content-Length` remains unknown rather than displaying zero. Export rejects a model with no effective-opacity rows. `camera_path.json` contains validation-camera keyframes in normalized coordinates. Every center must remain within radius 2 of the normalized origin. Stage 2 baseline retains this descriptor; video is `not_generated` rather than silently invoking an unpinned encoder.

`result.zip` uses sorted names, fixed timestamps, fixed permissions, and project-relative paths. It includes canonical/browser PLY, export/camera metadata, dataset/effective config, and validation evaluation. `bundle.json` records the archive SHA-256 externally to avoid a self-referential archive hash. No absolute local paths or optimizer internals enter the bundle.

## Worker and browser delivery

A successful `project_3dgs` attempt progresses through geometry, Gaussian training, validation, export, and atomic workspace publication. Stable roles are:

- `gaussian_model`, `gaussian_training_result`, `gaussian_progress`;
- `gaussian_dataset` at the real per-attempt preparation path;
- `gaussian_evaluation`, `gaussian_test_evaluation`, and `gaussian_test_decision`;
- `gaussian_export_metadata`, `gaussian_canonical`, `gaussian_camera_path`, `gaussian_bundle`;
- `scene_splat` for browser loading.

Failed/cancelled partial evaluation/export stays in lifecycle attempt diagnostics and is not advertised. The frontend progressively loads `scene_splat`, aborts its metadata requests and disposes the viewer on switch/unmount, reports load/error/size state, labels the view as normalized arbitrary units, and retains point/mesh fallback when Gaussian is absent. Canonical Gaussian coordinates remain unchanged for display: when `camera_path.json` is available, the viewer follows Nerfstudio's COLMAP defaults by averaging normalized camera centers for the initial orbit pivot and averaging OpenCV image-up axes for its natural up direction. CAD presets and OrbitControls use that same display frame; missing, malformed, or degenerate camera-path orientation falls back to the export scene center/bounds and normalized +Z without blocking older assets.

## Official 3DGS baseline profile

Public `standard_v1` and internal `rtx4060_8gb_development_v1` now resolve to the same schema-v5 baseline with effective-config hash `e10433a11846f014801924e7d6780beaa50152377c7d4b7762253489a6304a09`. This supersedes rather than layers on top of the schema-v4 floater cleanup profile:

- Project 3DGS COLMAP extraction uses a shared `OPENCV` camera, exhaustive matching, mapper global-BA tolerance `1e-6`, the reconstruction with the most registered images/points, then `image_undistorter --output_type COLMAP`; training consumes the resulting pinhole images, cameras, and sparse points.
- Training runs for 30,000 iterations at up to 1280 pixels on the longest edge, with L1/SSIM weights 0.8/0.2 and SH degree promoted from 0 to 3 every 1,000 iterations.
- Position LR decays from `1.6e-4` to `1.6e-6`; feature, opacity, scale, and rotation LRs remain at `0.0025`, `0.05`, `0.005`, and `0.001`, while higher SH bands use feature LR divided by 20. Graphdeco passes a `0.01` delay multiplier but no delay steps, so its scheduler applies no warm-up delay; schema v5 records the effective behavior rather than the inert argument.
- The pinned Apache-2.0 `gsplat==1.5.3` `DefaultStrategy` owns signed-gradient duplication/splitting, stochastic local splitting, opacity pruning/reset, and optimizer-state remapping. Refinement runs from 500 to 15,000 every 100 iterations with gradient threshold `0.0002`, scale threshold `0.01`, prune opacity `0.005`, maximum normalized scale `0.1`, and opacity reset every 3,000 iterations.
- There is no hard Gaussian-count truncation, custom split quota, screen-size cleanup, late cleanup, or custom recovery pause. CUDA OOM is an explicit failed attempt rather than a silent algorithm change.
- Training cameras are shuffled without replacement per epoch. Matching the official baseline, Validation runs at iterations 7,000 and 30,000; only Validation can select the published candidate. A successful run retains one final lifecycle checkpoint and one published model.

Graphdeco Gaussian Splatting commit `54c035f7834b564019656c3e3fcc3646292f727d` is used only as the algorithm/COLMAP CLI reference because its custom license limits reuse to non-commercial research/evaluation. No Graphdeco trainer source is copied. Nerfstudio commit `50e0e3c70c775e89333256213363badbf074f29d` is an Apache-2.0 cross-check, not a runtime dependency. The actual topology implementation is the already pinned Apache-2.0 gsplat release.

The prior measured schema-v4 evidence remains historical: its retained 225-image profile reduced visible oversized splats but still produced ceiling holes and circular blur, so it is not layered into this baseline. Its Validation and already-consumed held-out Test artifacts remain immutable and are not rerun or used to tune schema v5.

The first completed schema-v5 public `room` Validation-only baseline used the frozen 32-view contract (26 Train / 3 Validation / 3 untouched Test) and all 112,627 accepted COLMAP sparse points. It completed 30,000 iterations in 2,255.04 seconds, peaked at 1,664,833,024 allocated and 2,116,026,368 reserved CUDA bytes, and selected iteration 7,000 over iteration 30,000 by Validation PSNR. The selected 1,008,582-Gaussian model reached mean/P10 PSNR 25.9971/21.9621 dB and mean SSIM 0.8396; iteration 30,000 reached 25.2327/0.8056 and was not published. The selected model has 197 Validation-visible screen-radius violations above 0.15 (165 with opacity above 0.1), zero scale violations above 0.1, and 10,180 effective rows outside twice its robust radius. Canonical/browser PLY export completed with 1,008,582 rows and a 250,129,924-byte scene asset. The three fixed Validation renders are detailed and mostly free of circular blur, but one wide view still shows peripheral stretched/transparent artifacts and this 32-view benchmark does not directly observe the ceiling; therefore it establishes a functioning official baseline, not proof that the private 225-image ceiling defect is solved. No Test view was loaded or evaluated.

Training writes no periodic full checkpoint and a successful job retains only its final checkpoint. Validation candidates use one overwrite-in-place model-only snapshot. Evaluations and exports expose screen-radius, maximum-scale, and robust-distance health diagnostics without hiding rows in the browser. Export records a robust median scene center and 95th-percentile radius as framing fallback; the browser prefers the exported camera-path pose center and image-up orientation for stable room-centered CAD-style orbit, bounded zoom, coherent presets, and click-to-focus while remaining backward-compatible with older export metadata.

## Integrated evidence

Ignored final job `outputs/experiments/r2_15/integrated-room-v2/20260730_092826_4c1497f3` completed on its first attempt with the final validation→frozen-test→export worker route. It used 32 uploaded room images, COLMAP sparse reconstruction, 100-iteration project training, validation, one consumed held-out test evaluation, export, API retrieval, deterministic bundle, and viewer PLY parsing. It produced 3,608 Gaussians and an 896,369-byte browser PLY; the audited viewer loader decoded all 3,608 rows. Every advertised asset returned HTTP 200.

The earlier diagnostic job `outputs/experiments/r2_15/integrated-room-v1/20260730_091731_0ac5a833` retained two failed attempts that exposed and fixed workspace input handling:

1. relative input symlink resolved from the workspace to the wrong target;
2. absolute symlink fixed discovery but violated dataset-root containment.

The root-cause fix copies immutable uploaded inputs into the disposable attempt workspace. Original job inputs remain preserved, retries stay isolated, and the completed workspace is removed after atomic publication.
