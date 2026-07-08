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
For large image sets, increase `--max-points` to keep the fused cloud dense enough for inspection. Lower `--conf-percentile` keeps more VGGT depth samples but can introduce more noisy points.

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

Build a mesh from a generated point cloud:

```bash
uv run python scripts/mesh_from_pointcloud.py \
  outputs/jobs/{job_id}/geometry/points_aligned.ply \
  outputs/jobs/{job_id}/geometry/mesh.glb \
  --diagnostics-output outputs/jobs/{job_id}/diagnostics/mesh.json
```

Mesh output uses Open3D. Job creation runs the same mesh postprocess automatically when `output_type=mesh` is selected, preferring `points_aligned.ply` over the raw point cloud.

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
