"""Train-depth visibility filtering for frozen Gaussian model snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PROFILE_ID = "vggt_visibility_v1"
REASON_FRONT_FREE_SPACE = np.uint8(1)
REASON_OUTSIDE_OVERSIZED = np.uint8(2)


@dataclass(frozen=True)
class DepthEvidence:
    image_id: str
    image_name: str
    camera_from_normalized: np.ndarray
    intrinsic: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    confidence_threshold: float
    scale_x: float
    scale_y: float
    pad_left: float
    pad_top: float
    far_depth: float
    scale_observations: int
    scale_log_mad: float


@dataclass(frozen=True)
class GaussianFilterResult:
    keep: np.ndarray
    reasons: np.ndarray
    support_counts: np.ndarray
    contradiction_counts: np.ndarray
    envelope_counts: np.ndarray
    oversized: np.ndarray
    diagnostics: dict[str, Any]


class GaussianVggtFilterError(ValueError):
    """Raised when Train-depth evidence cannot safely derive a filtered model."""


def classify_gaussians(
    *,
    means: np.ndarray,
    scales: np.ndarray,
    quaternions: np.ndarray,
    opacities: np.ndarray,
    evidence: list[DepthEvidence],
    relative_depth_tolerance: float = 0.08,
    sigma_extent: float = 3.0,
    minimum_contradictions: int = 2,
    oversized_normalized_scale: float = 0.1,
    oversized_screen_ratio: float = 0.15,
    maximum_removal_fraction: float = 0.25,
    chunk_size: int = 100_000,
) -> GaussianFilterResult:
    points = np.asarray(means, dtype=np.float64)
    axis_scales = np.asarray(scales, dtype=np.float64)
    quats = np.asarray(quaternions, dtype=np.float64)
    alpha = np.asarray(opacities, dtype=np.float64)
    count = len(points)
    if points.shape != (count, 3) or axis_scales.shape != (count, 3):
        raise GaussianVggtFilterError("Gaussian means/scales must be N x 3")
    if quats.shape != (count, 4) or alpha.shape != (count,):
        raise GaussianVggtFilterError("Gaussian quaternions/opacities have invalid shapes")
    if count == 0 or not all(
        np.isfinite(value).all() for value in (points, axis_scales, quats, alpha)
    ):
        raise GaussianVggtFilterError("Gaussian tensors must be non-empty and finite")
    if (axis_scales <= 0).any():
        raise GaussianVggtFilterError("Gaussian scales must be positive")
    if len(evidence) < 2:
        raise GaussianVggtFilterError("VGGT filtering requires at least two Train depth views")
    if not 0 < relative_depth_tolerance < 1:
        raise GaussianVggtFilterError("relative depth tolerance must be between zero and one")
    if minimum_contradictions < 2:
        raise GaussianVggtFilterError("at least two independent contradictions are required")

    support = np.zeros(count, dtype=np.uint16)
    contradictions = np.zeros(count, dtype=np.uint16)
    envelope = np.zeros(count, dtype=np.uint16)
    screen_oversized = np.zeros(count, dtype=bool)
    normalized_oversized = axis_scales.max(axis=1) > oversized_normalized_scale
    rotation_matrices = quaternion_matrices(quats)
    view_records = []

    for view in evidence:
        _validate_evidence(view)
        view_support = 0
        view_contradictions = 0
        view_envelope = 0
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            point_chunk = points[start:stop]
            scale_chunk = axis_scales[start:stop]
            rotation_chunk = rotation_matrices[start:stop]
            camera = view.camera_from_normalized
            camera_points = point_chunk @ camera[:3, :3].T + camera[:3, 3]
            z = camera_points[:, 2]
            positive = z > 1e-8
            safe_z = np.maximum(z, 1e-8)
            u = view.intrinsic[0, 0] * camera_points[:, 0] / safe_z + view.intrinsic[0, 2]
            v = view.intrinsic[1, 1] * camera_points[:, 1] / safe_z + view.intrinsic[1, 2]
            canvas_u = u * view.scale_x + view.pad_left
            canvas_v = v * view.scale_y + view.pad_top
            ui = np.rint(canvas_u).astype(np.int64)
            vi = np.rint(canvas_v).astype(np.int64)
            inside = (
                positive
                & (ui >= 0)
                & (vi >= 0)
                & (ui < view.depth.shape[1])
                & (vi < view.depth.shape[0])
            )
            sampled_depth = np.zeros(stop - start, dtype=np.float64)
            sampled_confidence = np.zeros(stop - start, dtype=np.float64)
            indices = np.flatnonzero(inside)
            sampled_depth[indices] = view.depth[vi[indices], ui[indices]]
            sampled_confidence[indices] = view.confidence[vi[indices], ui[indices]]
            valid = (
                inside
                & np.isfinite(sampled_depth)
                & np.isfinite(sampled_confidence)
                & (sampled_depth > 1e-8)
                & (sampled_confidence >= view.confidence_threshold)
            )

            camera_z_axis = camera[2, :3]
            oriented_axes = rotation_chunk * scale_chunk[:, None, :]
            depth_axes = np.einsum("j,njk->nk", camera_z_axis, oriented_axes)
            depth_radius = sigma_extent * np.linalg.norm(depth_axes, axis=1)
            lower = z - depth_radius
            upper = z + depth_radius
            tolerance = relative_depth_tolerance * sampled_depth
            supported = valid & (lower <= sampled_depth + tolerance) & (
                upper >= sampled_depth - tolerance
            )
            contradicted = valid & (upper < sampled_depth - tolerance)
            within_envelope = inside & (lower <= view.far_depth * 1.15)

            world_scale = float(np.linalg.norm(camera[:3, 0]))
            world_axis_radius = sigma_extent * world_scale * scale_chunk.max(axis=1)
            projected_radius = (
                max(float(view.intrinsic[0, 0]), float(view.intrinsic[1, 1]))
                * world_axis_radius
                / safe_z
            )
            large_on_screen = inside & (
                projected_radius
                > oversized_screen_ratio
                * max(view.depth.shape[0] / view.scale_y, view.depth.shape[1] / view.scale_x)
            )

            support[start:stop] += supported.astype(np.uint16)
            contradictions[start:stop] += contradicted.astype(np.uint16)
            envelope[start:stop] += within_envelope.astype(np.uint16)
            screen_oversized[start:stop] |= large_on_screen
            view_support += int(supported.sum())
            view_contradictions += int(contradicted.sum())
            view_envelope += int(within_envelope.sum())
        view_records.append(
            {
                "image_id": view.image_id,
                "image_name": view.image_name,
                "scale_observations": view.scale_observations,
                "scale_log_mad": view.scale_log_mad,
                "confidence_threshold": view.confidence_threshold,
                "far_depth": view.far_depth,
                "surface_support_count": view_support,
                "free_space_contradiction_count": view_contradictions,
                "capture_envelope_count": view_envelope,
            }
        )

    oversized = normalized_oversized | screen_oversized
    free_space = (contradictions >= minimum_contradictions) & (support == 0)
    outside_large = (envelope == 0) & (support == 0) & oversized
    reasons = np.zeros(count, dtype=np.uint8)
    reasons[free_space] |= REASON_FRONT_FREE_SPACE
    reasons[outside_large] |= REASON_OUTSIDE_OVERSIZED
    keep = reasons == 0
    removed = int((~keep).sum())
    removal_fraction = removed / count
    if removal_fraction > maximum_removal_fraction:
        raise GaussianVggtFilterError(
            f"VGGT filter would remove {removal_fraction:.3%}, above the {maximum_removal_fraction:.1%} safety limit"
        )
    visible_kept = int(((alpha > 0.01) & keep).sum())
    if visible_kept < min(100, count):
        raise GaussianVggtFilterError("VGGT filter leaves too few visible Gaussians")
    diagnostics = {
        "schema_version": 1,
        "profile": PROFILE_ID,
        "settings": {
            "relative_depth_tolerance": relative_depth_tolerance,
            "sigma_extent": sigma_extent,
            "minimum_contradictions": minimum_contradictions,
            "oversized_normalized_scale": oversized_normalized_scale,
            "oversized_screen_ratio": oversized_screen_ratio,
            "maximum_removal_fraction": maximum_removal_fraction,
        },
        "counts": {
            "input": count,
            "kept": int(keep.sum()),
            "removed": removed,
            "removed_front_free_space": int(free_space.sum()),
            "removed_outside_oversized": int(outside_large.sum()),
            "removed_with_both_reasons": int((free_space & outside_large).sum()),
            "visible_kept": visible_kept,
        },
        "removal_fraction": removal_fraction,
        "train_depth_view_count": len(evidence),
        "train_image_ids": [view.image_id for view in evidence],
        "validation_image_ids": [],
        "test_image_ids": [],
        "views": view_records,
    }
    return GaussianFilterResult(
        keep=keep,
        reasons=reasons,
        support_counts=support,
        contradiction_counts=contradictions,
        envelope_counts=envelope,
        oversized=oversized,
        diagnostics=diagnostics,
    )


def quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise GaussianVggtFilterError("Gaussian quaternion cannot be zero")
    w, x, y, z = (values / norms).T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)


def _validate_evidence(view: DepthEvidence) -> None:
    camera = np.asarray(view.camera_from_normalized, dtype=np.float64)
    intrinsic = np.asarray(view.intrinsic, dtype=np.float64)
    depth = np.asarray(view.depth)
    confidence = np.asarray(view.confidence)
    if camera.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise GaussianVggtFilterError("depth evidence camera matrices are invalid")
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise GaussianVggtFilterError("depth/confidence evidence shapes do not match")
    if not np.isfinite(camera).all() or not np.isfinite(intrinsic).all():
        raise GaussianVggtFilterError("depth evidence cameras must be finite")
    if min(view.scale_x, view.scale_y, view.far_depth) <= 0:
        raise GaussianVggtFilterError("depth evidence transforms must be positive")
