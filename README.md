# Image3D-SceneGraph

Image3D-SceneGraph is a calibration-free image/video/panorama to semantic 3D scene reconstruction project.

The user uploads one image, multiple images, a video, or a 360 panorama. The system estimates scene geometry internally, reconstructs a 3D representation, attaches semantic objects, infers spatial relations, and exposes the result through a web interface.

## Current Scope

This repository is being built as an algorithm-focused computer vision demo, not a generic 3D reconstruction platform.

The first MVP targets:

1. A local job pipeline for image / multi-image / video / panorama inputs.
2. A backend API for upload, reconstruction jobs, result assets, and scene graph JSON.
3. A frontend for upload, progress, 3D viewing, object inspection, and export.
4. A geometry baseline using a modern image-to-3D reconstruction model.
5. Algorithmic additions around scale recovery, semantic 3D fusion, physical consistency, and scene graph reasoning.

## Important Constraints

- Users do not provide camera intrinsics, extrinsics, depth, or pose.
- The system may estimate camera parameters internally.
- Do not claim metric accuracy until scale recovery is evaluated.
- Keep model checkpoints, datasets, generated outputs, and debug artifacts out of git.

## Repository Layout

```text
Image3D-SceneGraph/
  codex.md                  # project plan and maintenance guide
  README.md
  pyproject.toml            # Python project metadata, managed with uv
  frontend/                 # web UI, to be initialized later
  backend/                  # API service, to be initialized later
  src/image3d_scenegraph/   # core algorithm and pipeline code
  scripts/                  # runnable local tools
  docs/                     # design notes and experiment records
  examples/                 # lightweight example placeholders
```

See [codex.md](codex.md) for the working plan, milestones, and development rules.

## Environment

Python dependencies are managed with `uv`.

```bash
uv sync
```

Add dependencies only when the related backend, algorithm, or tooling code is introduced.

## Backend MVP

Start the local API:

```bash
uv run uvicorn backend.main:app --reload
```

Run the backend checks:

```bash
uv run ruff check backend src scripts tests
uv run pytest
uv run python scripts/smoke_backend.py
```

Current mock API:

- `GET /api/health`
- `GET /api/backends`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/manifest`
- `GET /api/jobs/{job_id}/scene`
- `GET /api/jobs/{job_id}/assets/{path}`
- `GET /api/jobs/{job_id}/download`

`POST /api/jobs` accepts multipart form data:

- `mode`: `image`, `multi_image`, `video`, or `panorama`
- `geometry_backend`: `mock`, `vggt`, `colmap`, `colmap_vggt`, `project_3dgs`, `dust3r`, or `mast3r`
- `output_type`: `point_cloud`, `mesh`, or `gaussian_splat`
- `gaussian_trainer`: `graphdeco` (default), `project`, or experimental `mcmc`; used only with `project_3dgs + gaussian_splat`
- `sfm_feature_profile`: `sift_v1` (default) or experimental `aliked_n16rot_v1`; used by `colmap`, `colmap_vggt`, and the final COLMAP feature stage of `project_3dgs`
- `sfm_local_matcher`: `bruteforce` (default) or experimental `lightglue`; selects descriptor matching inside each chosen image pair
- `sfm_pairing`: `exhaustive` (default), video-only `sequential_loop`, or multi-image-only `vocab_tree`
- `sfm_geometric_verification`: `default_v1` (default) or experimental `guided_v1`; both keep COLMAP geometric verification enabled, and Guided changes only `FeatureMatching.guided_matching`
- `sfm_camera_calibration`: `shared_opencv_v1`, `shared_simple_radial_v1`, or multi-image-only experimental `auto_grouped_simple_radial_v1`; omitted requests preserve each backend's historical default (`project_3dgs` uses shared OPENCV, while `colmap` and `colmap_vggt` use shared SIMPLE_RADIAL)
- `gaussian_geometry_source`: `colmap` (default) or video-only `vggt_ba` (experimental/research-only)
- `gaussian_postprocess`: `none` (default) or `vggt_visibility_v1` (experimental/research-only)
- `gaussian_longest_edge`: 1280–3072px; used only with `project_3dgs + gaussian_splat`
- `video_keyframe_profile`: `standard_v1` (default) or explicit `standard_v2`; used only for bounded video jobs. The frontend continues to submit v1 until the v2 geometry gates pass.
- `video_rotation`: `auto`, `clockwise_90`, `counterclockwise_90`, or `180`
- `files`: one or more uploaded files

Implemented geometry paths:

- `geometry_backend=mock` with `output_type=point_cloud`
- `geometry_backend=vggt` with `output_type=point_cloud` or `mesh`, when the local VGGT repo and checkpoint are installed
- `geometry_backend=colmap` with `output_type=point_cloud` or `mesh`, when the `colmap` executable is installed
- `geometry_backend=colmap_vggt` with `output_type=point_cloud` or `mesh`, when both COLMAP and VGGT are installed
- `geometry_backend=project_3dgs` with `output_type=gaussian_splat`, when COLMAP and the selected CUDA trainer are available. `gaussian_trainer=project` uses Project v7: the fixed `standard_v1` profile runs 30,000 iterations with a 1280px default and a frontend-selectable longest edge through 3072px, 3NN RMS initialization, training-time screen-radius pruning disabled, and Validation-selected export. Experimental `gaussian_trainer=mcmc` uses the installed Apache-2.0 gsplat MCMC strategy in the same native/distributed lifecycle, with the frozen `mcmc_v1` method package and a 3,000,000-Gaussian global cap. `graphdeco` invokes its pinned isolated research environment and remains the default. Every trainer reuses the same geometry input and common SOR/Validation/export/manifest/Viewer path; MCMC is runnable but not promoted pending remote quality/resource evidence.
- `mode=video` with `project_3dgs + gaussian_splat`, when FFmpeg/ffprobe, COLMAP, and the selected trainer are available. Both profiles analyze one 10-second–10-minute MP4/MOV/M4V/WebM (606-second technical tolerance, up to 2 GiB) at 6 fps. The default historical `video_keyframes_standard_v1` selects up to 1,000 upright frames. Explicit `video_keyframes_standard_v2` selects a 4 fps uniform base and motion-adaptive frames up to 5 fps. Ordinary-COLMAP v2 runs Mapper on at most 1,000 uniformly spaced base frames, then performs up to two `image_registrator` + non-clearing `point_triangulator` passes against the complete selected-frame feature database before gap recovery. This bounds Mapper cost without discarding the additional v2 views. Gap recovery runs at most two local COLMAP registration rounds with viable gap-bridging candidates from the remaining 6 fps pool. Each round is limited to 25% of the initial selection and recovery is limited to 50% cumulatively. The first accepted triangulation can feed a zero-new-candidate propagation round so new 3D points register deeper gap frames; rounds without strict gap improvement stop before expensive triangulation. Initial expansion and accepted recovery rounds share one final CUDA bundle adjustment with one CPU fallback. Explicit v2 also bounds intermediate Mapper global-BA frequency while leaving v1 replay unchanged. The upload is staged in 8 MiB chunks; selected frames retain source PTS and generated JPEGs use EXIF Orientation 1 while authoritative metadata stays in JSON sidecars. Final registration must pass 12-frame, 70% registration, and 80% temporal-coverage gates before 3DGS starts; surviving gaps above 2 seconds remain a soft warning.

Video support is bounded offline reconstruction, not realtime SLAM or evidence of drift-free multi-room mapping. Coordinates remain normalized arbitrary units. FFmpeg and ffprobe are external executable dependencies; their absence disables only video ingestion, not image-based Project jobs.

Two orthogonal research options are available for Gaussian jobs:

- `gaussian_geometry_source=vggt_ba` is currently video-only. It runs fixed 8-frame/4-frame-overlap VGGT windows, classifies cameras with at least 32 reliable observations as strong, excludes weak cameras from local BA graph evidence, and makes one deterministic recovery-window attempt per adjacent disconnect (at most eight frames, with at least three reliable frames from each side). A surviving 12-camera/70%-support/80%-temporal-coverage component seeds COLMAP 4 SIFT/exhaustive triangulation, omitted-image registration, and global BA. Ordinary COLMAP Mapper runs only after one of three classified late quality failures: `vggt_graph_unusable_after_recovery`, `vggt_seed_geometry_insufficient`, or `vggt_registration_gate_failed`; CUDA/OOM, dependency, checkpoint, non-finite, I/O, cancellation, subprocess, and unexpected-code failures remain failed Jobs. Manifests separately record requested/effective source, fallback status, and reason. A fallback Job remains viewable but is not successful VGGT-BA A/B evidence. Missing verified nonlocal geometry remains `open_trajectory_unverified` and does not itself trigger fallback or support a loop-closure/bounded-drift claim.
- `gaussian_postprocess=vggt_visibility_v1` can follow either geometry source. After Original Validation/export, it uses at most 64 Train views and scale-aligned VGGT depth to conservatively remove multi-view free-space floaters and unsupported oversized Gaussians outside the capture envelope. It does not load Validation/Test to derive the mask, fill holes, create walls, retrain, or overwrite the Original model. Filtering is fail-soft: the Original job remains usable if the derivative is unavailable. When successful, the frontend provides Original/VGGT-filtered A/B viewing and separate Validation/export assets while the stable `scene_splat` role continues to identify Original.

Both options are experimental and research-only pending dependency, real-scene, resource, and license validation. VGGT-BA requires the pinned VGGT, DINOv2, LightGlue/ALIKED, VGGSfM tracker, PyCOLMAP, SciPy, and COLMAP dependencies; postprocessing requires the base VGGT repo/checkpoint. `GET /api/backends` reports these capabilities separately so missing BA dependencies do not disable ordinary COLMAP Gaussian jobs or base VGGT cleanup.

For a same-input visual comparison with retained Official job `20260806_060729_a5d1d377`, select `Multi-image` → `Project 3DGS` → `Gaussian splat` → `Project v7 (gsplat)` and upload the same 225 source images. The frontend shows `standard_v1 · v10` on the resulting job. New Project v7 and MCMC jobs stop after Train/Validation model selection and export; they do not load Test or create a Test-consumption record. Coordinates remain normalized arbitrary units, not metres.

Create a complete asynchronous MCMC video Job through the same public task lifecycle:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F mode=video \
  -F geometry_backend=project_3dgs \
  -F output_type=gaussian_splat \
  -F gaussian_trainer=mcmc \
  -F files=@room.mp4
```

Every successful native Project/MCMC preparation publishes `gaussian/replay/` with hardlinked (or copied across filesystems) registered images, cameras, dataset contract, and frozen initialization. It excludes the COLMAP database and matches. Before CUDA starts, all trainer paths write `gaussian/preparation/ATTEMPT/geometry_readiness.json` and reject catastrophic isolated camera poses or a collapsed 3NN-scale distribution with stable reason codes. Reuse a passing replay without rerunning geometry, sparse-point selection, or 3NN scale estimation:

```bash
uv run python scripts/run_gaussian_training.py \
  --dataset-contract outputs/jobs/JOB/gaussian/replay/dataset.json \
  --dataset-root outputs/jobs/JOB/gaussian/replay \
  --run-dir outputs/replays/JOB-mcmc \
  --trainer mcmc \
  --initialization frozen \
  --distributed
```

Append `--readiness-only` to the runner command to generate/check `geometry_readiness.json` and exit before any trainer or distributed CUDA process starts.

A retained failed workspace with one independently verified bad camera can be converted only through an explicit derivative; the tool does not choose an image or start training, and refuses to overwrite its output:

```bash
uv run python scripts/derive_gaussian_pose_repair.py \
  --dataset-contract WORKSPACE/dataset.json \
  --dataset-root WORKSPACE \
  --points WORKSPACE/colmap/undistorted/sparse_txt/points3D.txt \
  --exclude-image-id 61 \
  --output-dir outputs/replays/JOB-pose-repair
```

The new `repair.json` binds the original and derived hashes and states that surviving sparse coordinates are preserved without rerunning bundle adjustment. Its `replay/` may then be passed to `run_gaussian_training.py --initialization frozen`; it is a `repaired_derivative`, not a successful result for the original SfM arm.

DUSt3R, MASt3R, and panorama-to-geometry are still API contract placeholders and return a clear not implemented error.

Optional geometry backends are not downloaded with the base project. Check local backend availability with:

```bash
uv run python scripts/setup_model.py --backend vggt
```

VGGT setup is intentionally explicit because the base checkpoint is about 5GB and the full experimental VGGT-BA dependency set requires substantially more disk space. Install only after checking free space. The installer pins the VGGT, DINOv2, LightGlue, VGGSfM, and ALIKED sources, downloads the VGGT/DINOv2/VGGSfM/ALIKED weights into project-local ignored paths, installs PyCOLMAP/SciPy/LightGlue, and writes checkpoint hashes to `checkpoints/vggt/ba-dependencies.json`; jobs do not download at runtime:

```bash
uv run python scripts/setup_model.py --backend vggt --install
```

The backend also exposes `GET /api/backends` so the frontend can disable missing model/trainer integrations and show the required setup command.

The external Graphdeco trainer uses a separate pinned environment and is dry-run by default:

```bash
uv run python scripts/setup_gaussian_trainer.py --trainer graphdeco
```

After CUDA availability and free-space checks pass, install it explicitly:

```bash
uv run python scripts/setup_gaussian_trainer.py \
  --trainer graphdeco --install --accept-research-license
```

Graphdeco is licensed only for research/evaluation, so setup requires explicit acknowledgement. Its checkout, environment, and generated results stay ignored locally. This command never alters the project uv environment.

Install the isolated CUDA-enabled COLMAP 4.0.0 before using GPU SIFT. The setup script is a dry run unless `--install` is supplied and never invokes `sudo`:

```bash
uv run python scripts/setup_colmap_cuda.py
sudo apt-get update
sudo apt-get install -y \
  cuda-compiler-12-2 cuda-libraries-dev-12-2 libcudnn9-cuda-12 \
  cmake libfreeimage-dev libmetis-dev libgoogle-glog-dev \
  libceres-dev libsuitesparse-dev libopenimageio-dev \
  libopenexr-dev openimageio-tools
sudo mkdir -p /usr/include/opencv4
uv run python scripts/setup_colmap_cuda.py --install
```

The pinned production build is installed under ignored `external/colmap-4-cuda/` and does not replace `/usr/bin/colmap`. Runners resolve `IMAGE3D_COLMAP_BIN` first, then this project-local build, then `colmap` on `PATH`. The RTX 4060 development build uses CUDA 12.2, SM 89, ONNX Runtime, and cuDNN 9.

The experimental `aliked_n16rot_v1` feature and `lightglue` local matcher use pinned COLMAP ONNX assets. Setup is dry-run by default; `--install` preserves valid assets and repairs damaged project-managed files only after a replacement `.part` passes size/SHA-256 verification. Jobs verify local size/SHA-256 and never download models or silently change algorithms:

```bash
uv run python scripts/setup_colmap_learned_features.py
uv run python scripts/setup_colmap_learned_features.py --install
```

The two LightGlue models are the COLMAP release copies of Apache-2.0 LightGlue pretrained weights; ALIKED remains BSD-3-Clause upstream. Runtime model paths are project-local and git-ignored.

A geometry runner can then select feature extraction, local matching, and image pairing independently:

```bash
uv run python scripts/run_colmap_sparse.py \
  --image-dir INPUT \
  --output-dir OUTPUT \
  --feature-profile aliked_n16rot_v1 \
  --local-matcher lightglue \
  --pairing exhaustive \
  --geometric-verification guided_v1
```

The same controls are available through the asynchronous API:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F mode=multi_image \
  -F geometry_backend=colmap \
  -F output_type=point_cloud \
  -F sfm_feature_profile=aliked_n16rot_v1 \
  -F sfm_local_matcher=lightglue \
  -F sfm_pairing=exhaustive \
  -F sfm_geometric_verification=guided_v1 \
  -F sfm_camera_calibration=auto_grouped_simple_radial_v1 \
  -F files=@view-01.jpg \
  -F files=@view-02.jpg
```

Phase 3 exposes `sfm_pairing=exhaustive|sequential_loop|vocab_tree` independently from the local matcher. `exhaustive` remains the new-product default; `sequential_loop` is video-only and uses temporal neighbors plus loop detection, while `vocab_tree` is multi-image-only. Both retrieval profiles require a descriptor-compatible tree under ignored `external/colmap-vocab/`: the 256K-word SIFT tree and official 64K-word ALIKED N16Rot tree are installed and verified separately. The setup script is dry-run by default; explicit `--install` atomically repairs a damaged managed tree after validating its replacement. Jobs never download or substitute trees:

```bash
uv run python scripts/setup_colmap_vocab_tree.py
uv run python scripts/setup_colmap_vocab_tree.py --install
```

The legacy video field `colmap_matcher=exhaustive|sequential` remains accepted and maps to `exhaustive|sequential_loop`; conflicting old/new fields fail. The production frontend defaults remain `sift_v1 + bruteforce + exhaustive + default_v1 + incremental`, while camera defaults preserve the historical backend path: shared OPENCV for `project_3dgs` and shared SIMPLE_RADIAL for direct `colmap`/`colmap_vggt`. The 2026-08-13 2048-keypoint/1280px ETH3D run found SIFT-LightGlue nearly disconnected while ALIKED-LightGlue completed but cost substantially more matching time; those historical settings do not select the current 8192-keypoint default. Phase 3 pairing code is available but has no promotion-grade real A/B yet. Global Mapper remains unexposed pending Phase 6; Test cannot select these options. Graphdeco setup uses PyTorch `2.3.1+cu121` and compiles its extensions with `/usr/local/cuda-12.2`, while Project uses the existing `gsplat` cu121 wheel.

Phase 4 exposes `sfm_geometric_verification=default_v1|guided_v1` after pairing. Both profiles explicitly keep COLMAP geometric verification enabled and retain the selected build's default `TwoViewGeometry`/RANSAC parameters; `guided_v1` changes only `FeatureMatching.guided_matching=1`. The same profile reaches standard-v2 `matches_importer` recovery. Matching timing remains the combined local-matching + geometric-verification wall time because COLMAP does not publish a stable separate duration. Guided is experimental and has no promotion-grade real A/B yet.

Phase 5 exposes `sfm_camera_calibration=shared_opencv_v1|shared_simple_radial_v1|auto_grouped_simple_radial_v1` after geometric verification. Shared profiles create one camera with the named COLMAP model. Auto-grouped is limited to multi-image input and deterministically groups only complete normalized device + optional lens + focal-length + decoded-size + EXIF-orientation evidence; missing or invalid device/focal/orientation evidence leaves that image in its own camera group. It never uses filename, upload order, directory layout, GPS, or serial-number metadata. `colmap_vggt` rejects OPENCV because its current dense unprojection supports only pinhole/radial distortion. VGGT-BA remains a shared-camera video path and rejects auto-grouping; standard-v2 recovery frames inherit its existing shared camera. Camera plausibility warnings use COLMAP Mapper's focal-ratio/extra-parameter bounds as diagnostics only and do not become job gates. No profile has promotion-grade real A/B evidence.

Every current COLMAP-backed API/adapter job publishes `sfm_camera_calibration_diagnostics=diagnostics/sfm_camera_calibration.json` from the raw sparse model before Gaussian undistortion. It records planned/initial/final camera groups, prior-focal flags, registration, named focal/principal-point/distortion values, focal change, sparse track/reprojection summaries, COLMAP build, and soft warnings. The compact manifest/UI expose only profile, model, group count, and warning count; raw parameters are not controls. Gaussian `sfm_diagnostics` is schema 4 and adds the validated camera summary plus each database image's `camera_id`; schemas 1–3 remain readable as the historical shared-OPENCV path.

New Gaussian jobs publish SfM diagnostics schema 4 with a verified View Graph summary: non-empty `two_view_geometries` rows define edges, and the asset records edge/component/degree distributions, isolated nodes, candidate-surviving versus Guided-added correspondences, camera-calibration provenance, and optional video edge-span/soft-gap bridge evidence. This is diagnostic evidence, not a new hard gate. Analyze retained schema 1/2/3/4 diagnostics without rewriting the accepted Job:

```bash
uv run python scripts/analyze_sfm_view_graph.py --job-dir outputs/jobs/JOB
```

Run COLMAP sparse SfM directly for a local image folder:

```bash
.venv/bin/python scripts/run_colmap_sparse.py \
  --image-dir path/to/images \
  --output-dir outputs/colmap_run \
  --local-matcher bruteforce \
  --pairing exhaustive \
  --geometric-verification default_v1 \
  --camera-calibration shared_simple_radial_v1
```

COLMAP output is a sparse SfM reference: it estimates a global camera graph and sparse point cloud. Use it to compare whether VGGT multi-image drift is caused by windowed model inference or by weak image overlap / texture. Video extraction writes `diagnostics/video_keyframe_timing.json`; explicit ordinary-COLMAP v2 also writes `diagnostics/video_initial_registration_expansion.json` and `diagnostics/colmap_timing.json`. After same-source v1/v2 geometry-only runs and a second deterministic v2 extraction, `scripts/evaluate_video_v2_promotion.py` evaluates the frozen 95% registration, zero `>2s` gap, camera/point retention, recovery-budget, determinism, and `≤2×` time gates; a failed gate exits nonzero and never changes defaults.

Run COLMAP + VGGT dense fusion directly:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python scripts/run_colmap_vggt_dense.py \
  --image-dir path/to/images \
  --output-dir outputs/colmap_vggt_run \
  --pairing exhaustive \
  --geometric-verification default_v1 \
  --camera-calibration shared_simple_radial_v1 \
  --vggt-batch-size 4 \
  --vggt-overlap-size 2 \
  --vggt-grouping sequential \
  --fusion-mode points \
  --max-points 2000000 \
  --conf-percentile 50 \
  --confidence-threshold-scope global \
  --consistency-support-policy any_support \
  --point-budget-policy random \
  --device cuda
```

This path uses COLMAP for global camera poses and VGGT for dense depth. The first scale alignment baseline estimates a per-image depth scale from COLMAP sparse observations and VGGT depth samples, then fuses all depth maps in COLMAP's global frame.
For large image sets, increase `--max-points` to keep the fused cloud dense enough for inspection. Lower `--conf-percentile` keeps more VGGT depth samples but can introduce more noisy points. The stable points path applies that percentile globally. `--confidence-threshold-scope per_frame` is an experimental alternative for independently calibrated VGGT windows; it improved two of three ETH3D scenes but regressed `terrains`, so it is not the default. `--consistency-support-policy adaptive_two` is an experimental accuracy-priority filter: points visible in two or more usable neighbors require two supports, while points with zero or one usable neighbor retain the baseline requirement. It improved 2/5 cm F1 slightly on all three ETH3D scenes but remains opt-in because the gain is small and can reduce strict completeness. Combining both experimental modes preserved the same mixed pattern—improvements on `pipes` and `delivery_area`, but lower 2-50 cm F1 on `terrains`—so the stable filtering defaults remain `global + any_support`.

The final point cap uses deterministic seeded random sampling by default. `--point-budget-policy spatial_balanced` is the Phase 3 experimental alternative: it orders accepted points along a Morton space-filling curve and keeps equal-mass stratum midpoints, producing exactly the same requested point count without GT input. In strict paired 2M-point tests it improved 1/2/5 cm F1 by `+0.006265/+0.005373/+0.002104` on `terrains` and `+0.002567/+0.005658/+0.005174` on `delivery_area`; `pipes` was below the 2M cap and therefore unchanged, while a separate 1M activation check improved all six thresholds. The policy also remained positive when combined with either confidence scope and support policy. It is retained as the strongest Phase 3 candidate, but `random` remains the stable default pending validation on another capped non-ETH3D scene.

The frontend exposes the three COLMAP+VGGT ablation factors as independent controls, so all eight Phase 1×2×3 combinations remain available. The G1.26 review freezes the existing stable baseline as `Sequential + Points + Global + Any support + Random`; the complete command above specifies it without relying on implicit defaults. No experimental factor was promoted: the frozen cross-scene evidence is mixed, and the point-budget candidate lacks an active non-ETH3D validation. Consequently Stage 1 did not produce a new improved configuration that satisfies every Gate G1 condition; this retained configuration is the reproducible fallback, not a claim of metric-scale recovery or universal improvement. For a controlled private-dataset comparison, keep the images, depth batch, confidence percentile, maximum points, output type, and environment unchanged, and create separate jobs in this order:

```text
baseline: Global    + Any support       + Random
Phase 1:  Per frame + Any support       + Random
Phase 2:  Global    + Adaptive two-view + Random
Phase 3:  Global    + Any support       + Spatial balanced
all-on:   Per frame + Adaptive two-view + Spatial balanced  (optional)
```

`POST /api/jobs` persists and returns a queued job immediately; the local serial worker performs reconstruction in the background while the frontend polls job status. Upload bodies are still read into backend memory. For a 225-image folder, exhaustive COLMAP matching examines 25,200 unordered image pairs, and depth batch `4` requires roughly 57 VGGT groups if all images register. `Max points` limits only the final PLY after filtering—it does not bound peak candidate-array or cross-view-filtering memory. Validate the setup on a representative subset before committing to each full run.

COLMAP+VGGT runs also write `diagnostics/vggt_groups.json`. The shared schema records each group's members and first-member reference, source order, actual consecutive overlap, sparse shared-track count, camera-center distance in COLMAP's arbitrary reconstruction units, and camera view-axis angle. It labels zero-track and below-8-track reference links as `disconnected` and `weak`. Sequential grouping intentionally uses disjoint chunks, so a nonzero requested overlap is reported as `ignored_by_sequential_grouping` rather than as active overlap. For a retained job with COLMAP text outputs, generate or byte-check the same diagnostics without rerunning reconstruction:

```bash
uv run python scripts/generate_vggt_group_diagnostics.py \
  --job-dir outputs/jobs/{job_id} \
  --write  # use --check after freezing the artifact
```

`diagnostics/scale_disagreement.json` reports `abs(log(scale_a) - log(scale_b))` p50/p90/p95 over every unique image pair sharing at least 8 sparse COLMAP tracks, split into pairs that do or do not occur together in a VGGT group. Only pairs with sparse-COLMAP scale estimates at both ends are included; these arbitrary-scale consistency statistics are not metric-scale recovery. A retained job can be generated or byte-checked without reconstruction:

```bash
uv run python scripts/generate_scale_disagreement_diagnostics.py \
  --job-dir outputs/jobs/{job_id} \
  --write  # use --check after freezing the artifact
```

Analyze a generated point cloud before attempting coordinate alignment:

```bash
uv run python scripts/analyze_pointcloud.py \
  --input outputs/jobs/{job_id}/geometry/points.ply \
  --output outputs/jobs/{job_id}/diagnostics/geometry.json
```

The diagnostics include point count, bounding boxes, density estimates, and dominant RANSAC planes. Use this to verify whether a reliable ground/table/wall plane exists before applying automatic upright alignment.

Align a generated point cloud to its dominant plane:

```bash
uv run python scripts/align_pointcloud.py \
  --input outputs/jobs/{job_id}/geometry/points.ply \
  --output outputs/jobs/{job_id}/geometry/points_aligned.ply \
  --diagnostics-output outputs/jobs/{job_id}/diagnostics/alignment.json
```

The alignment step preserves the original point cloud and writes a separate aligned PLY. By default it rotates the strongest detected plane to the +Z axis and translates that plane to zero height. The point-cloud viewer uses the same Z-up convention and displays its grid on the XY plane.
Job creation also runs this alignment as a generic point-cloud postprocess. Any backend that returns a `point_cloud` asset can expose `point_cloud_aligned` in the manifest, and the frontend viewer lets the user switch between Raw and Aligned when the aligned asset exists.

Census free-space floater populations in an exported Gaussian splat:

```bash
uv run python scripts/analyze_gaussian_floaters.py \
  --gaussian-ply outputs/jobs/{job_id}/gaussian/export/train-001/scene.ply \
  --points outputs/jobs/{job_id}/geometry/points.ply \
  --cameras outputs/jobs/{job_id}/geometry/cameras.json \
  --output /tmp/floater_census.json
```

The census splits gaussians into opacity bands (haze < 0.05, core, thick ≥ 0.25) and reports each band's distance to the SfM surface and camera path in colmap_world, the hugging fraction (inside the SfM neighbor-spacing median), and the free-space count (beyond the neighbor-spacing p90 and farther than 1.0 world unit from every camera). Scales stay in the normalized training frame. These numbers are the reference gate for floater mitigation experiments (codex.md §17).

## ETH3D Geometry Evaluation

The optional ETH3D benchmark evaluates geometric reconstruction separately from the API and `JobStore`. Dataset files stay under the ignored `data/` directory and are not redistributed by this repository. The first frozen scene is `pipes`, defined by `benchmarks/eth3d-v1/pipes.json`.

Place the official `pipes` undistorted images, COLMAP calibration, and evaluation scan at:

```text
data/benchmarks/eth3d/pipes/
  images/dslr_images_undistorted/
  dslr_calibration_undistorted/{cameras,images,points3D}.txt
  dslr_scan_eval/{scan_alignment.mlp,scan1.ply}
```

The official evaluator is a separate C++ tool that requires Boost, Eigen3, and PCL development packages. Preview its setup without changing the machine:

```bash
uv run python scripts/setup_eth3d_evaluator.py
```

After installing the native prerequisites yourself, clone and build it explicitly with:

```bash
uv run python scripts/setup_eth3d_evaluator.py --install
```

Run the normal image-only reconstruction first. Do not pass ETH3D reference cameras or scans to this command:

```bash
env -u LD_LIBRARY_PATH uv run python scripts/run_colmap_vggt_dense.py \
  --image-dir data/benchmarks/eth3d/pipes/images/dslr_images_undistorted \
  --output-dir outputs/benchmarks/eth3d-v1/pipes/colmap_vggt_points/reconstruction \
  --pairing exhaustive \
  --geometric-verification default_v1 \
  --camera-calibration shared_simple_radial_v1 \
  --vggt-batch-size 4 \
  --vggt-grouping sequential \
  --fusion-mode points \
  --seed 42
```

Then align the reconstruction from corresponding estimated/reference camera centers and call the official evaluator:

```bash
uv run python scripts/evaluate_eth3d_scene.py \
  --benchmark benchmarks/eth3d-v1/pipes.json \
  --reconstruction-dir outputs/benchmarks/eth3d-v1/pipes/colmap_vggt_points/reconstruction \
  --evaluator-bin external/eth3d-multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  --output-dir outputs/benchmarks/eth3d-v1/pipes/colmap_vggt_points/evaluation \
  --camera-ransac-threshold 0.05 \
  --camera-ransac-iterations 1000 \
  --min-camera-inliers 8 \
  --seed 42
```

The evaluator applies one camera-center-derived Sim(3) to the raw reconstruction point cloud. It never fits the reconstruction to the laser scan with ICP. Reported Accuracy, Completeness, and F1 are therefore **GT-camera-Sim(3)-aligned geometry quality**, not evidence that the reconstruction recovered metric scale by itself. Do not use the generic Z-up `points_aligned.ply` as benchmark input; it has a different purpose.

Build a mesh from a generated point cloud:

```bash
uv run python scripts/mesh_from_pointcloud.py \
  outputs/jobs/{job_id}/geometry/points_aligned.ply \
  outputs/jobs/{job_id}/geometry/mesh.glb \
  --diagnostics-output outputs/jobs/{job_id}/diagnostics/mesh.json
```

Mesh output uses Open3D. Job creation runs the same mesh postprocess automatically when `output_type=mesh` is selected, preferring `points_aligned.ply` over the raw point cloud.
The mesh diagnostics JSON records point filtering, local point spacing, bounding boxes, connected components, long-edge triangle removal, and timings. Use it as the first check before tuning reconstruction parameters.

Useful first-round mesh A/B commands:

```bash
# Poisson with stricter bridge cleanup.
uv run python scripts/mesh_from_pointcloud.py \
  outputs/jobs/{job_id}/geometry/points_aligned.ply \
  outputs/jobs/{job_id}/geometry/mesh_poisson_clean.glb \
  --diagnostics-output outputs/jobs/{job_id}/diagnostics/mesh_poisson_clean.json \
  --voxel-size 0.04 \
  --poisson-depth 9 \
  --edge-trim-factor 1.6 \
  --max-triangles 250000

# Ball pivoting, useful when Poisson closes too many holes.
uv run python scripts/mesh_from_pointcloud.py \
  outputs/jobs/{job_id}/geometry/points_aligned.ply \
  outputs/jobs/{job_id}/geometry/mesh_bpa.glb \
  --diagnostics-output outputs/jobs/{job_id}/diagnostics/mesh_bpa.json \
  --method ball_pivoting \
  --voxel-size 0.035 \
  --edge-trim-factor 1.6 \
  --max-triangles 250000

# Alpha shape baseline, useful as a coarse geometry sanity check.
uv run python scripts/mesh_from_pointcloud.py \
  outputs/jobs/{job_id}/geometry/points_aligned.ply \
  outputs/jobs/{job_id}/geometry/mesh_alpha.glb \
  --diagnostics-output outputs/jobs/{job_id}/diagnostics/mesh_alpha.json \
  --method alpha_shape \
  --alpha 0.12 \
  --max-triangles 250000
```

When mesh jobs run through the API, the same options can be tuned with environment variables such as `IMAGE3D_MESH_METHOD`, `IMAGE3D_MESH_VOXEL_SIZE`, `IMAGE3D_MESH_POISSON_DEPTH`, `IMAGE3D_MESH_EDGE_TRIM_FACTOR`, and `IMAGE3D_MESH_MAX_TRIANGLES`.

Run VGGT directly for a local image folder:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python scripts/run_vggt_pointcloud.py \
  --image-dir path/to/images \
  --output-dir outputs/vggt_run \
  --max-images 4 \
  --batch-size 8 \
  --overlap-size 4 \
  --max-points 200000 \
  --device cuda \
  --precision auto
```

On 8GB GPUs, direct 4 to 5 image VGGT inference can run out of memory. The backend defaults to `IMAGE3D_VGGT_MAX_IMAGES=8`, `IMAGE3D_VGGT_BATCH_SIZE=8`, and `IMAGE3D_VGGT_OVERLAP_SIZE=4`, aligns adjacent groups with a shared-camera Sim3 estimate, and keeps the VGGT backbone in automatic half precision while leaving the camera/depth heads in fp32 for dtype stability. The frontend exposes per-job VGGT `Max images`, `Batch size`, and `Overlap` controls, so larger uploads such as 225 images can be attempted without changing environment variables.

`panorama` currently means one equirectangular 360 image. Real panorama reconstruction is not implemented yet; the backend records the mode and returns mock geometry through the same manifest contract.

The MVP writes results to `outputs/jobs/{job_id}/`, including `manifest.json`, `geometry/points.ply`, `scene_graph/scene.json`, and `logs/run.log`. VGGT jobs also write `geometry/cameras.json`.

The point-cloud viewer includes X/Y/Z axis flip controls. X/Y/Z buttons are display-only transforms, so they can correct viewer coordinate conventions without rerunning reconstruction.

## Frontend MVP

Install and start the web UI:

```bash
cd frontend
npm install
npm run test
npm run build
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`, so run the backend in a separate terminal before using the frontend.

Current frontend flow:

1. Choose `Image`, `Multi-image`, `Video`, or `Panorama`.
2. Choose a geometry backend and output type.
3. Upload local files.
4. Create a mock reconstruction job.
5. View job metrics, scene objects, output links, and the mock `.ply` point cloud.
6. Or load an existing job id.

## Development Status

The repository has a project skeleton, a mock backend job API, and a functional frontend MVP against the stable `manifest.json` output contract.
