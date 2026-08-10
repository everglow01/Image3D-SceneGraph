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

Failed/cancelled partial evaluation/export stays in lifecycle attempt diagnostics and is not advertised. The frontend progressively loads `scene_splat`, aborts its metadata requests and disposes the viewer on switch/unmount, reports load/error/size state, labels the view as normalized arbitrary units, and retains point/mesh fallback when Gaussian is absent. Canonical Gaussian coordinates remain unchanged for display: when `camera_path.json` is available, the viewer averages normalized camera centers for the initial orbit pivot and OpenCV image-up axes for its natural up direction. CAD presets and OrbitControls use that same display frame; missing, malformed, or degenerate camera-path orientation falls back to the export scene center/bounds and normalized +Z without blocking older assets.

## Official 3DGS baseline profile

Public `standard_v1` and internal `rtx4060_8gb_development_v1` now resolve to the same schema-v6 baseline with effective-config hash `684890c30587d15f92170a4958d19db305f38ef57ac2ad0bf9e7c784ef53e082`. Schema v6 directly applies the first Graphdeco-alignment stage to the frontend-selectable Project trainer while preserving the 30,000-iteration budget and Validation-only selection:

- Project 3DGS COLMAP extraction uses a shared `OPENCV` camera, exhaustive matching, mapper global-BA tolerance `1e-6`, the reconstruction with the most registered images/points, then `image_undistorter --output_type COLMAP`; training consumes the resulting pinhole images, cameras, and sparse points.
- Training runs for 30,000 iterations at up to 1280 pixels on the longest edge, with L1/SSIM weights 0.8/0.2, a Graphdeco-equivalent `[0,1]` render clamp before loss, and SH degree promoted from 0 to 3 every 1,000 iterations.
- Sparse and dense seeds initialize isotropic scale from the RMS distance to the three nearest neighbors without global clipping, and initialize every quaternion to identity. These replace the schema-v5 one-neighbor clipped scale and random quaternion initialization.
- Position LR decays from `1.6e-4` to `1.6e-6` after multiplying by the Graphdeco Train-camera extent (maximum distance from the Train-camera mean times 1.1); feature, opacity, scale, and rotation LRs remain at `0.0025`, `0.025`, `0.005`, and `0.001`, while higher SH bands use feature LR divided by 20.
- The pinned Apache-2.0 `gsplat==1.5.3` `DefaultStrategy` owns signed-gradient duplication/splitting, stochastic local splitting, opacity pruning/reset, and optimizer-state remapping. Refinement runs from 500 to 15,000 every 100 iterations with gradient threshold `0.0002`, Train-extent-scaled world thresholds `0.01`/`0.1`, prune opacity `0.005`, Graphdeco-equivalent 20-pixel screen-radius pruning after iteration 3,000, and opacity reset every 3,000 iterations. Screen radius is used only for pruning, not additional splitting.
- There is no hard Gaussian-count truncation, custom split quota, late cleanup, or custom recovery pause. CUDA OOM is an explicit failed attempt rather than a silent algorithm change.
- Training cameras are shuffled without replacement per epoch. Stage 2.1 observes Validation at iterations 3,000, 5,000, 7,000, 10,000, 15,000, 20,000, 25,000, and 30,000; each observation records the nominal 30,000-iteration budget, actual optimizer updates, current-attempt elapsed time, render metrics, Gaussian count, and topology/health diagnostics. Only Validation PSNR can select the overwrite-in-place published candidate. This adds no early stopping and no intermediate full checkpoint: a successful run still retains one final lifecycle checkpoint and one published best-Validation model.

Graphdeco Gaussian Splatting commit `54c035f7834b564019656c3e3fcc3646292f727d` remains the algorithm/COLMAP reference and can be selected as an isolated research/evaluation trainer. Its custom license still prohibits unrestricted commercial runtime use: the checkout, submodules, patch hook, environment, and outputs remain ignored under `external/`/`outputs/`, and setup requires explicit research-license acceptance. The external environment never enters the project uv environment; the Project arm continues to use pinned Apache-2.0 gsplat 1.5.3.

Trainer selection is one product option under `project_3dgs + gaussian_splat`: `graphdeco` (current default) or `project`, available through `run_gaussian_training.py --trainer`, `gaussian_trainer` on `POST /api/jobs`, and the frontend Trainer selector. The choice/revision/license/command hash are persisted rather than inferred. Both arms derive native inputs from the same frozen contract: identical image hashes and Train/Validation/Test IDs, matching pinhole intrinsics/poses, and identical accepted sparse seeds. Graphdeco uses a minimal runtime wrapper to honor `sparse/0/test.txt` as the exact Validation list. Its INRIA PLY is validated and converted to the project normalized snapshot, after which the existing common Validation evaluator/exporter/manifest/Viewer path is reused. Test remains outside trainer runtime and is not consumed during integration or smoke.

The prior measured schema-v4 and schema-v5 evidence remains historical. Existing Validation and consumed held-out Test artifacts remain immutable and are not rerun or used to tune schema v6.

The first schema-v6 Stage 2.2 `room` Validation-only trajectory completed on the same frozen 32-view contract and all 112,627 sparse points. Its observations were 3k `10.6753/0.2481`, 5k `12.2432/0.4848`, 7k `11.5762/0.4318`, 10k `11.1324/0.3601`, 15k `12.5068/0.4507`, 20k `20.7823/0.6837`, 25k `20.7715/0.6896`, and 30k `20.7430/0.6913` mean PSNR/SSIM. Validation PSNR selected 20k with 662,453 Gaussians; the full run took 1,233.26 seconds and peaked at 1,405,091,840 reserved CUDA bytes. This is materially below schema-v5 best Validation and Graphdeco final quality, so neither 7k nor 20k is promoted as an early-stop default. Gaussian count fell from 676,675 at 3k to 509,144 at 3.1k and 469,714 at 3.3k, then quality recovered mainly after topology refinement stopped. Test remained untouched. The frozen ignored protocol, complete trajectory, hashes, and audit are under `outputs/experiments/schema-v6-stage2-2-room-validation-trajectory-20260810/`.

A subsequent frozen Validation-only 2×2 factorial isolated initial scale (`3NN RMS` versus legacy clipped `1NN`) and screen-radius pruning (`20 px` versus disabled) while preserving the same data, points/colors, seed, 30,000-step budget, and eight observations. The four best Validation cells were: 3NN+20px `20.7823/0.6837` at 20k (662,453 Gaussians), 3NN+off `26.7929/0.8456` at 10k (948,899), legacy+20px `21.9523/0.7233` at 30k (672,011), and legacy+off `26.4699/0.8423` at 10k (975,322). At 3.1k, disabling screen pruning retained 83,792 additional Gaussians under 3NN and 82,531 under legacy; by 3.3k the differences were 154,763 and 145,183. Disabling the 20px rule improved final PSNR by 5.5888 dB under 3NN and 4.0938 dB under legacy. With that rule disabled, 3NN exceeded legacy by 0.3231 dB PSNR and 0.0033 SSIM at their shared 10k observation. Same-view previews agreed with the metrics: both screen-off arms were detailed, while both screen-on arms showed blur, stretching, and opaque smearing. The bounded causal conclusion is that 20-pixel screen-radius pruning—not 3NN initialization—was the dominant schema-v6 regression cause on this scene; 3NN was slightly preferable when screen pruning was absent. This single-scene result does not authorize early stopping or a broader quality claim. Test IDs `117/207/40` were never loaded or evaluated, and no Test artifact or consumption record exists. The complete protocol, three new arms, hashes, previews, runner, and final audit remain ignored under `outputs/experiments/schema-v6-screen-init-ablation-20260810/`.

The completed schema-v5 public `room` Validation-only baseline remains the before-change comparison point: It completed 30,000 iterations in 2,255.04 seconds, peaked at 1,664,833,024 allocated and 2,116,026,368 reserved CUDA bytes, and selected iteration 7,000 over iteration 30,000 by Validation PSNR. The selected 1,008,582-Gaussian model reached mean/P10 PSNR 25.9971/21.9621 dB and mean SSIM 0.8396; iteration 30,000 reached 25.2327/0.8056 and was not published. The selected model has 197 Validation-visible screen-radius violations above 0.15 (165 with opacity above 0.1), zero scale violations above 0.1, and 10,180 effective rows outside twice its robust radius. Canonical/browser PLY export completed with 1,008,582 rows and a 250,129,924-byte scene asset. The three fixed Validation renders are detailed and mostly free of circular blur, but one wide view still shows peripheral stretched/transparent artifacts and this 32-view benchmark does not directly observe the ceiling; therefore it establishes a functioning official baseline, not proof that the private 225-image ceiling defect is solved. No Test view was loaded or evaluated.

Training writes no periodic full checkpoint and a successful job retains only its final checkpoint. Validation candidates use one overwrite-in-place model-only snapshot. Evaluations and exports expose screen-radius, maximum-scale, and robust-distance health diagnostics without hiding rows in the browser. Export records a robust median scene center and 95th-percentile radius as framing fallback; the browser prefers the exported camera-path pose center and image-up orientation for stable room-centered CAD-style orbit, bounded zoom, coherent presets, and click-to-focus while remaining backward-compatible with older export metadata.

## Integrated evidence

Ignored final job `outputs/experiments/r2_15/integrated-room-v2/20260730_092826_4c1497f3` completed on its first attempt with the final validation→frozen-test→export worker route. It used 32 uploaded room images, COLMAP sparse reconstruction, 100-iteration project training, validation, one consumed held-out test evaluation, export, API retrieval, deterministic bundle, and viewer PLY parsing. It produced 3,608 Gaussians and an 896,369-byte browser PLY; the audited viewer loader decoded all 3,608 rows. Every advertised asset returned HTTP 200.

The earlier diagnostic job `outputs/experiments/r2_15/integrated-room-v1/20260730_091731_0ac5a833` retained two failed attempts that exposed and fixed workspace input handling:

1. relative input symlink resolved from the workspace to the wrong target;
2. absolute symlink fixed discovery but violated dataset-root containment.

The root-cause fix copies immutable uploaded inputs into the disposable attempt workspace. Original job inputs remain preserved, retries stay isolated, and the completed workspace is removed after atomic publication.
