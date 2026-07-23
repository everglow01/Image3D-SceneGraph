from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityRansacResult:
    transform: np.ndarray
    inliers: np.ndarray
    residuals: np.ndarray
    iterations: int
    threshold: float


def estimate_similarity_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source, target = _validate_correspondences(source_points, target_points)
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("Similarity correspondences must not be collinear")

    source_variance = float(np.sum(source_centered * source_centered))
    if source_variance <= 1e-12:
        raise ValueError("Similarity source points have zero variance")

    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    if np.linalg.det(rotation) <= 0:
        raise ValueError("Similarity estimation produced an improper rotation")

    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Similarity estimation produced a non-positive scale")
    translation = target_centroid - scale * rotation @ source_centroid

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def estimate_similarity_transform_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    threshold: float,
    iterations: int,
    min_inliers: int,
    seed: int,
) -> SimilarityRansacResult:
    source, target = _validate_correspondences(source_points, target_points)
    if threshold <= 0:
        raise ValueError("RANSAC threshold must be positive")
    if iterations <= 0:
        raise ValueError("RANSAC iterations must be positive")
    if min_inliers < 3 or min_inliers > len(source):
        raise ValueError("RANSAC min_inliers must be between 3 and the correspondence count")

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_median = float("inf")
    for _ in range(iterations):
        selected = rng.choice(len(source), size=3, replace=False)
        try:
            candidate = estimate_similarity_transform(source[selected], target[selected])
        except ValueError:
            continue
        residuals = np.linalg.norm(transform_points(source, candidate) - target, axis=1)
        inliers = residuals <= threshold
        count = int(inliers.sum())
        median = float(np.median(residuals[inliers])) if count else float("inf")
        if best_inliers is None or count > int(best_inliers.sum()) or (
            count == int(best_inliers.sum()) and median < best_median
        ):
            best_inliers = inliers
            best_median = median

    if best_inliers is None or int(best_inliers.sum()) < min_inliers:
        found = 0 if best_inliers is None else int(best_inliers.sum())
        raise ValueError(f"RANSAC found {found} inliers; at least {min_inliers} are required")

    transform = estimate_similarity_transform(source[best_inliers], target[best_inliers])
    residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    final_inliers = residuals <= threshold
    if int(final_inliers.sum()) < min_inliers:
        raise ValueError(
            f"Refined similarity has {int(final_inliers.sum())} inliers; at least {min_inliers} are required"
        )
    if not np.array_equal(final_inliers, best_inliers):
        transform = estimate_similarity_transform(source[final_inliers], target[final_inliers])
        residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        final_inliers = residuals <= threshold

    return SimilarityRansacResult(
        transform=transform,
        inliers=final_inliers,
        residuals=residuals,
        iterations=iterations,
        threshold=threshold,
    )


def transform_points(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    values = np.asarray(points)
    transform = np.asarray(target_from_source, dtype=np.float64)
    if values.shape[-1:] != (3,):
        raise ValueError("Points must have shape (..., 3)")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Similarity transform must be a finite 4x4 matrix")
    if not np.isfinite(values).all():
        raise ValueError("Points must contain only finite values")
    flat = values.reshape(-1, 3).astype(np.float64, copy=False)
    transformed = flat @ transform[:3, :3].T + transform[:3, 3]
    output_dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
    return transformed.reshape(values.shape).astype(output_dtype, copy=False)


def decompose_similarity_transform(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Similarity transform must be a finite 4x4 matrix")
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    if scale <= 0:
        raise ValueError("Similarity transform must have a positive scale")
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) <= 0:
        raise ValueError("Similarity transform does not contain a proper rotation")
    return scale, rotation, matrix[:3, 3].copy()


def _validate_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("Similarity correspondences must have matching shape (N, 3)")
    if len(source) < 3:
        raise ValueError("At least three similarity correspondences are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity correspondences must contain only finite values")
    return source, target
