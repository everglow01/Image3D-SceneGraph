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
- `geometry_backend`: `mock`, `vggt`, `colmap`, `colmap_vggt`, `dust3r`, `mast3r`, or `nerfstudio_3dgs`
- `output_type`: `point_cloud`, `mesh`, or `gaussian_splat`
- `files`: one or more uploaded files

Implemented geometry paths:

- `geometry_backend=mock` with `output_type=point_cloud`
- `geometry_backend=vggt` with `output_type=point_cloud` or `mesh`, when the local VGGT repo and checkpoint are installed
- `geometry_backend=colmap` with `output_type=point_cloud` or `mesh`, when the `colmap` executable is installed
- `geometry_backend=colmap_vggt` with `output_type=point_cloud` or `mesh`, when both COLMAP and VGGT are installed

DUSt3R, MASt3R, automatic 3DGS training, and video-to-geometry are still API contract placeholders and return a clear not implemented error until their adapters are added.

Optional geometry backends are not downloaded with the base project. Check local backend availability with:

```bash
uv run python scripts/setup_model.py --backend vggt
```

VGGT setup is intentionally explicit because the checkpoint is about 5GB and the full environment can require substantially more disk space. Install only after checking free space:

```bash
uv run python scripts/setup_model.py --backend vggt --install
```

The backend also exposes `GET /api/backends` so the frontend can disable missing model integrations and show the required setup command.

Install COLMAP before using the COLMAP baseline:

```bash
sudo apt install colmap
```

Run COLMAP sparse SfM directly for a local image folder:

```bash
.venv/bin/python scripts/run_colmap_sparse.py \
  --image-dir path/to/images \
  --output-dir outputs/colmap_run \
  --matcher sequential
```

COLMAP output is a sparse SfM reference: it estimates a global camera graph and sparse point cloud. Use it to compare whether VGGT multi-image drift is caused by windowed model inference or by weak image overlap / texture.

Run COLMAP + VGGT dense fusion directly:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python scripts/run_colmap_vggt_dense.py \
  --image-dir path/to/images \
  --output-dir outputs/colmap_vggt_run \
  --matcher exhaustive \
  --vggt-batch-size 4 \
  --max-points 2000000 \
  --conf-percentile 50 \
  --device cuda
```

This path uses COLMAP for global camera poses and VGGT for dense depth. The first scale alignment baseline estimates a per-image depth scale from COLMAP sparse observations and VGGT depth samples, then fuses all depth maps in COLMAP's global frame.
For large image sets, increase `--max-points` to keep the fused cloud dense enough for inspection. Lower `--conf-percentile` keeps more VGGT depth samples but can introduce more noisy points. The stable points path applies that percentile globally. `--confidence-threshold-scope per_frame` is an experimental alternative for independently calibrated VGGT windows; it improved two of three ETH3D scenes but regressed `terrains`, so it is not the default. `--consistency-support-policy adaptive_two` is an experimental accuracy-priority filter: points visible in two or more usable neighbors require two supports, while points with zero or one usable neighbor retain the baseline requirement. It improved 2/5 cm F1 slightly on all three ETH3D scenes but remains opt-in because the gain is small and can reduce strict completeness. Combining both experimental modes preserved the same mixed pattern—improvements on `pipes` and `delivery_area`, but lower 2-50 cm F1 on `terrains`—so the stable filtering defaults remain `global + any_support`.

The final point cap uses deterministic seeded random sampling by default. `--point-budget-policy spatial_balanced` is the Phase 3 experimental alternative: it orders accepted points along a Morton space-filling curve and keeps equal-mass stratum midpoints, producing exactly the same requested point count without GT input. In strict paired 2M-point tests it improved 1/2/5 cm F1 by `+0.006265/+0.005373/+0.002104` on `terrains` and `+0.002567/+0.005658/+0.005174` on `delivery_area`; `pipes` was below the 2M cap and therefore unchanged, while a separate 1M activation check improved all six thresholds. The policy also remained positive when combined with either confidence scope and support policy. It is retained as the strongest Phase 3 candidate, but `random` remains the stable default pending validation on another capped non-ETH3D scene.

The frontend exposes the three COLMAP+VGGT ablation factors as independent controls, so all eight Phase 1×2×3 combinations remain available. The stable baseline is `Global + Any support + Random`. For a controlled private-dataset comparison, keep the images, depth batch, confidence percentile, maximum points, output type, and environment unchanged, and create separate jobs in this order:

```text
baseline: Global    + Any support       + Random
Phase 1:  Per frame + Any support       + Random
Phase 2:  Global    + Adaptive two-view + Random
Phase 3:  Global    + Any support       + Spatial balanced
all-on:   Per frame + Adaptive two-view + Spatial balanced  (optional)
```

Record each job ID. A loaded job displays its effective policies and point-budget counts from persisted manifest metrics, and its log and diagnostics remain available in the downloadable bundle. For a 225-image folder, COLMAP+VGGT processes every image that COLMAP registers; the standalone VGGT `Max images` control does not apply. The current API is synchronous and reads uploads into backend memory, exhaustive COLMAP matching examines 25,200 unordered image pairs, and depth batch `4` requires roughly 57 VGGT groups if all images register. `Max points` limits only the final PLY after filtering—it does not bound peak candidate-array or cross-view-filtering memory. Validate the setup on a representative subset before committing to each full run.

COLMAP+VGGT runs also write `diagnostics/vggt_groups.json`. The shared schema records each group's members and first-member reference, source order, actual consecutive overlap, sparse shared-track count, camera-center distance in COLMAP's arbitrary reconstruction units, and camera view-axis angle. It labels zero-track and below-8-track reference links as `disconnected` and `weak`. Sequential grouping intentionally uses disjoint chunks, so a nonzero requested overlap is reported as `ignored_by_sequential_grouping` rather than as active overlap. For a retained job with COLMAP text outputs, generate or byte-check the same diagnostics without rerunning reconstruction:

```bash
uv run python scripts/generate_vggt_group_diagnostics.py \
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
  --matcher exhaustive \
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
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`, so run the backend in a separate terminal before using the frontend.

Current frontend flow:

1. Choose `Image`, `Multi-image`, `Video`, or `Panorama`.
2. Choose a geometry backend and output type.
3. Upload local files.
4. Create a mock reconstruction job.
5. View job metrics, scene objects, output links, and the mock `.ply` point cloud.
6. Or load an existing job id, including registered Gaussian splat jobs.

## Nerfstudio 3DGS Import

Existing Nerfstudio `splatfacto` checkpoints must be exported before the web viewer can load them.

Example export:

```bash
cd /home/owen/nerfstudio
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/ns-export gaussian-splat \
  --load-config outputs/drjohnson_hq/splatfacto/2026-06-22_161605/config.yml \
  --output-dir /home/owen/Image3D-SceneGraph/outputs/exports/drjohnson_hq \
  --output-filename splat.ply \
  --ply-color-mode sh_coeffs
```

Register the exported asset as an Image3D-SceneGraph job:

```bash
cd /home/owen/Image3D-SceneGraph
uv run python scripts/register_gaussian_splat.py \
  --splat outputs/exports/drjohnson_hq/splat.ply \
  --name drjohnson_hq
```

Then open the frontend and load the printed `job_id`.

## Development Status

The repository has a project skeleton, a mock backend job API, and a functional frontend MVP against the stable `manifest.json` output contract.
