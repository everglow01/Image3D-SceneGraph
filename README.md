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
- `geometry_backend`: `mock`, `vggt`, `dust3r`, `mast3r`, or `nerfstudio_3dgs`
- `output_type`: `point_cloud`, `mesh`, or `gaussian_splat`
- `files`: one or more uploaded files

Implemented geometry paths:

- `geometry_backend=mock` with `output_type=point_cloud`
- `geometry_backend=vggt` with `output_type=point_cloud`, when the local VGGT repo and checkpoint are installed

DUSt3R, MASt3R, mesh export, automatic 3DGS training, and video-to-geometry are still API contract placeholders and return a clear not implemented error until their adapters are added.

Optional geometry backends are not downloaded with the base project. Check local backend availability with:

```bash
uv run python scripts/setup_model.py --backend vggt
```

VGGT setup is intentionally explicit because the checkpoint is about 5GB and the full environment can require substantially more disk space. Install only after checking free space:

```bash
uv run python scripts/setup_model.py --backend vggt --install
```

The backend also exposes `GET /api/backends` so the frontend can disable missing model integrations and show the required setup command.

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
