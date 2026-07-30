# Project-owned Gaussian trainer and initialization contract

Stage 2C owns the 3D Gaussian model and optimization lifecycle. Runtime code must not invoke Nerfstudio, Splatfacto, or another full external trainer. The only external training primitive is the R2.2-pinned `gsplat==1.5.3+pt23cu121` rasterizer.

## Model and rendering

`image3d_scenegraph.gaussian.model.GaussianModel` owns these learnable tensors:

- normalized-space means `N x 3`;
- log scales `N x 3`, activated with `exp`;
- quaternion rotations `N x 4`, normalized before rendering;
- opacity logits `N`, activated with `sigmoid`;
- degree-3 spherical-harmonic coefficients `N x 16 x 3`.

Optimizer groups are position, feature, opacity, scaling, and rotation. Their effective rates, exponential schedules, SH schedule, loss weights, mutation cadence, and budgets come from the R2.4 resolved configuration. The renderer boundary is `gaussian.render.render_gaussians`; no project module imports an external trainer.

Training samples one deterministic train view per iteration, uses project-owned L1 + differentiable SSIM, validates finite gradients/parameters, tracks projected gradients/radii, and applies configured densification, opacity pruning, screen-size pruning, opacity reset, SH schedule, and Gaussian budget. CUDA OOM, cancellation, non-finite loss/gradient/parameter, invisible initialization, and unsupported input are explicit failures rather than successful partial output.

COLMAP `SIMPLE_RADIAL`, `RADIAL`, and `OPENCV` distortion is explicitly inverted by differentiable image resampling before pinhole rendering. This is necessary because gsplat 1.5.3's distortion UT projection is documented as non-differentiable to all inputs. Distortion is never silently discarded.

The trainer loads only `train` and `validation` IDs and asserts that no `test` ID enters runtime. LPIPS remains `not_run: dependency_not_audited_in_r2_10`; R2.11 owns dependency audit and final isolated test evaluation.

## Initialization schema v1

Sparse initialization parses project/COLMAP `points3D.txt` and records:

- source SHA-256;
- minimum track support and maximum reprojection error;
- non-finite/support/error/budget rejection counts;
- deterministic selected-row and complete selection hashes.

Dense initialization reads an RGB PLY and a matching Stage 1 support sidecar when provided. It applies, in order:

1. finite coordinate/confidence checks;
2. minimum support and confidence;
3. coordinate-wise quantile outlier rejection;
4. deterministic normalized-space voxel deduplication, preferring support then confidence;
5. deterministic Morton-stratified maximum budget.

It records every rejection count, source/sidecar hashes, settings, selected-row hash, and selection hash. Source PLY/sidecars are read-only. Selected points are transformed by the dataset contract's `normalized_from_world`; RGB initializes SH DC, higher bands are zero, rotations identity, and scales come from bounded nearest-neighbor spacing.

`dataset.with_initialization` returns a new hashed dataset contract pointing at a verified initialization asset. It never mutates a frozen source contract.

## Checkpoint and resume

The trainer serializes project-owned model tensors, optimizer, schedule iteration, densification accumulators, Python/NumPy/Torch CPU+CUDA RNG, and finite JSON metric history into the existing R2.5 `CheckpointState`. Attempts and checkpoints retain R2.5 atomic non-overwrite behavior and dataset/config/code/environment provenance checks.

Fresh/retry/resume remain distinct. Each attempt keeps its model/result/progress under immutable `attempts/{attempt_id}/artifacts`; a resume attempt cannot overwrite its parent. It loads only a provenance-compatible parent checkpoint. Iteration-indexed view sampling plus restored optimizer/model/RNG makes interrupted execution tolerance-reproducible without claiming cross-GPU bitwise identity.

## Worker and assets

`project_3dgs + gaussian_splat` is the project trainer identity. The adapter runs project COLMAP sparse preparation and `scripts/run_gaussian_training.py` as cancellable subprocesses under the R2.6 serial worker. `standard_v1` is resolved internally; raw Gaussian hyperparameters are not exposed by the public job form.

Complete jobs may publish:

- `gaussian_model`: project PyTorch model snapshot;
- `gaussian_training_result`: final iteration, loss, validation, resource, and checkpoint record;
- `gaussian_progress`: JSONL iteration/validation events;
- `gaussian_dataset`: effective dataset/initialization contract.

These are internal Stage 2C training assets, not the R2.12 canonical/browser Gaussian export. Failed/cancelled partial work stays under lifecycle attempt diagnostics and out of successful manifest assets.

## Verified Stage 2C evidence

Generated ignored evidence:

- CUDA learn/resume smoke: `outputs/experiments/r2_7/synthetic-smoke-20260730-d`.
  - 12 known-camera 64x64 views;
  - cancel at iteration 20, resume to iteration 40;
  - loss `0.14069837 -> 0.09866242`;
  - validation PSNR `16.01746`, SSIM `0.80934`;
  - 23,068,672 peak reserved bytes.
- R2.8 public indoor sparse evidence: `outputs/experiments/r2_8/room-sparse-100iter-v1`.
  - frozen `room` 26/3/3 contract, 20,000 budgeted sparse initialization;
  - 100-iteration train loss `0.54790366 -> 0.53736663`, final 18,765 Gaussians;
  - validation PSNR `2.22707`, SSIM `0.189397`, peak reserved 94,371,840 bytes;
  - same-config resume from iteration 50 to 100 reproduced final loss/PSNR exactly and SSIM within `1e-8`; model tensors differed at most `2.1e-7`, consistent with the frozen CUDA tolerance rather than bitwise identity.
- R2.10 validation-only ablation: `outputs/experiments/r2_10/terrains-validation-100iter-v8`.
  - frozen dataset hash `0c1c2d409aca649e1eba5e1aa1229601dd14bee031cb66d30a495f2d03459b33`;
  - identical config hash `125c31960d7c2185b2f30ee456c5aae5b281577454f5e5fc27a00d008cfa0624`, seed, train/validation split, resolution, and 100 iterations;
  - sparse: 8,745 initial, 8,387 final, loss `0.60263 -> 0.45565`, validation PSNR `5.89716`, SSIM `0.48045`, peak reserved 77,594,624 bytes;
  - dense: 2,000,000 source -> 1,830,684 supported -> 1,782,973 inlier -> 441,565 voxel -> 20,000 budgeted initial; 17,871 final, loss `0.49077 -> 0.36141`, validation PSNR `6.07229`, SSIM `0.47854`, peak reserved 98,566,144 bytes;
  - dense-minus-sparse: PSNR `+0.17513 dB`, SSIM `-0.001907`, Gaussian count `+9,484`, peak reserved `+20,971,520` bytes.

This short 100-iteration public geometry-development run proves executable sparse/dense policies and mixed validation/resource behavior; it does not promote dense initialization, establish final quality, substitute for the primary `room` indoor benchmark, or use held-out test metrics. R2.11–R2.16 still own full evaluation, canonical/browser export, frozen RTX profile, and upload-to-view delivery.
