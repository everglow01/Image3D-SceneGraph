# Stage 2A dependency audit

Audit date: 2026-07-29. Stage 2 may depend on a narrow rasterization primitive but owns the Gaussian model, losses, optimizer groups, schedules, densification/pruning, checkpoint/resume, evaluation, and export.

## Approved narrow candidates

### Differentiable rasterizer: `gsplat` 1.5.3

- Package: `gsplat==1.5.3`.
- Source tag: `v1.5.3`; commit `937e29912570c372bed6747a5c9bf85fed877bae`.
- Source/release: <https://github.com/nerfstudio-project/gsplat/releases/tag/v1.5.3>.
- License: Apache-2.0.
- Boundary: CUDA differentiable rasterization only, called behind the project trainer. The package's organization name does not permit importing Nerfstudio, Splatfacto, their configs, trainer, checkpoint, dataset, or command runtime.
- Direct package requirements: `torch`, `numpy`, `ninja`, `jaxtyping`, and `rich>=12`; the local install resolved against existing PyTorch without adding Nerfstudio.
- Native build: CUDA/C++ extension using Ninja and C++17. CPU-only real rendering is unsupported; CPU tests cover project contracts and analytic/reference checks only.
- Project packaging: it is isolated in the pinned `gpu` optional dependency group; API/CPU installation does not compile or install CUDA. Local setup/check commands are `env -u LD_LIBRARY_PATH uv sync --extra gpu --inexact` and `env -u LD_LIBRARY_PATH uv run --no-sync python scripts/check_gsplat.py`. `--inexact` preserves the separately installed VGGT research environment; a clean dedicated Stage 2 environment may omit it.

Rejected alternative: `graphdeco-inria/diff-gaussian-rasterization` has no stable release pin and its research/non-commercial license violates R2.0's production-compatible dependency boundary.

### Browser renderer: `@mkkellogg/gaussian-splats-3d` 0.4.7

- Package/lock: exact resolved `0.4.7`, integrity `sha512-0vy9/i9sJLFH/v3WJZ4axCsqjkToe8UsV3xY7bvK5EUC0akiRsWZODoCiSzpxhTLNyzSKTsyQKozIFeNA5RWRA==`.
- Upstream package commit recorded by npm: `2dfc83e497bd76e558fe970c54464b17b5f5c689`.
- Source: <https://github.com/mkkellogg/GaussianSplats3D>.
- License: MIT; no upstream NOTICE file was found. Peer dependency `three>=0.160.0`; this project resolves `three==0.185.1`, MIT.
- Boundary: browser display/decoding only. It is not differentiable and must not define the canonical Gaussian or training checkpoint format.
- Current settings avoid shared worker memory (`sharedMemoryForWorkers: false`), so Stage 2A does not add COOP/COEP requirements.

The caret declaration in `frontend/package.json` remains reproducible because `package-lock.json` pins the actual artifact. A future viewer upgrade is a new audited decision, not an implicit `npm update`.

### LPIPS decision for Stage 2D

- Upstream `richzhang/PerceptualSimilarity` package release is `lpips==0.1.4` (2021-08-25), repository license BSD-2-Clause.
- Default AlexNet LPIPS loads bundled learned calibration parameters but requests torchvision's pretrained AlexNet ImageNet trunk. In the frozen environment torchvision points to `https://download.pytorch.org/models/alexnet-owt-7be5be79.pth` and would download it when absent.
- The project has no approved local copy, SHA-256, redistribution/source-weight license record, or no-network installation evidence for that trunk. `lpips` is therefore not added to `pyproject.toml`/`uv.lock`; runtime never downloads the checkpoint.
- Evaluation records `lpips.status=not_run` and `reason=pretrained_weight_license_and_hash_not_audited`. PSNR/SSIM are not relabeled as LPIPS. Enabling LPIPS later requires a separate approved weight record and clean offline check.

## Local build/ABI evidence

Observed environment:

```text
OS: Linux 6.8.0-124-generic x86_64, glibc 2.35
Python: 3.10.12
PyTorch: 2.3.1+cu121
CUDA build/runtime requested by PyTorch: 12.1
NVIDIA driver: 535.309.01
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
compute capability: 8.9
reported VRAM: 8188 MiB
nvcc: /usr/local/cuda/bin/nvcc (CUDA 11.7; not used by the accepted binary-wheel path)
```

Completed:

```text
Source/JIT attempt: the PyPI py3 wheel imported, then failed correctly because local nvcc 11.7 cannot compile Ada `compute_89`.
Accepted path: official release binary `gsplat-1.5.3+pt23cu121-cp310-cp310-linux_x86_64.whl`.
env -u LD_LIBRARY_PATH uv sync --extra gpu --inexact
env -u LD_LIBRARY_PATH uv run --no-sync python scripts/check_gsplat.py
```

The accepted run produced finite `(1, 64, 64, 3)` RGB and `(1, 64, 64, 1)` alpha outputs on the RTX 4060, then finite gradients for means, quaternions, scales, opacities, and colors. The stable color derivative check returned analytic `32.960815`, central finite difference `32.958981`, relative error `5.5668204e-05`. Hiding CUDA returned the intended project-facing failure: `gsplat Stage 2 requires CUDA; torch.cuda.is_available() is false.`

The package is confined to the `gpu` optional dependency group; base installs do not select it. `uv.lock` records the exact official `gsplat==1.5.3+pt23cu121` binary URL and `torch==2.3.1+cu121` from the explicit PyTorch CUDA 12.1 index. The wheel is deliberately marked for Linux x86_64 and CPython 3.10; other Stage 2 GPU matrices require a separately audited artifact.

## Executable gate result

R2.2 passes on the frozen RTX 4060 development matrix. `scripts/check_gsplat.py` is the repeatable ABI/forward/backward/finite-difference check. Source compilation with the machine's CUDA 11.7 toolkit remains unsupported, but it is not the frozen installation route and no new CUDA toolkit download is required.

## Runtime dependency boundary check

At audit time:

- base installation: no PyTorch, gsplat, Nerfstudio, or Splatfacto is selected; the lockfile contains PyTorch/gsplat only for the explicit `gpu` extra;
- frontend lock: viewer plus Three.js, no Nerfstudio dependency;
- `scripts/register_gaussian_splat.py`: copies an existing export and is legacy/reference-only;
- any README command using a separate Nerfstudio tree describes the historical import route, not the Stage 2 runtime.

No full external 3DGS platform is approved for installation, training, evaluation, checkpointing, or export.
