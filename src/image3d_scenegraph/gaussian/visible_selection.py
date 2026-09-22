"""Conservative front-layer picking; no expected-depth or center-only visibility claims."""

from __future__ import annotations

import time

import numpy as np

from .cloud_render import CloudRenderError

MIN_WEIGHT = 1 / 255
MIN_COVERAGE = 0.5
MAX_SECONDS = 20
MAX_INTERSECTIONS = 24_000_000
MAX_PAIRS = 16_000_000
ROI_SIZE = 128


def polygon_pixels(polygon, x, y):
    """Pixel-center polygon test, including the boundary (same image coordinate frame)."""
    p = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(np.shape(x), dtype=bool)
    edge = inside.copy()
    for a, b in zip(p, np.roll(p, -1, axis=0), strict=True):
        dx, dy = b - a
        edge |= (
            (np.abs((x - a[0]) * dy - (y - a[1]) * dx) < 1e-7)
            & (x >= min(a[0], b[0]))
            & (x <= max(a[0], b[0]))
            & (y >= min(a[1], b[1]))
            & (y <= max(a[1], b[1]))
        )
        if dy != 0:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < a[0] + (y - a[1]) * dx / dy)
    return inside | edge


class FrontLayer:
    """Bounded CPU reference reducer shared by the GPU index adapter.

    The first significant contribution anchors a depth layer. Insufficient opacity
    in that layer abstains; it never searches farther back for an opaque surface.
    """

    def __init__(self, pixels, tolerance):
        self.trans = np.ones(pixels, dtype=np.float64)
        self.depth = np.full(pixels, np.inf)
        self.coverage = np.zeros(pixels)
        self.done = np.zeros(pixels, dtype=bool)
        self.candidates = []
        self.tolerance = tolerance

    def add(self, ids, pixels, depths, alphas):
        ids, pixels, depths, alphas = map(np.asarray, (ids, pixels, depths, alphas))
        if not len(ids):
            return
        order = np.lexsort((depths, pixels))
        ids, pixels, depths, alphas = (a[order] for a in (ids, pixels, depths, alphas))
        alphas = np.clip(alphas, 0, 0.999)
        logs = np.log1p(-alphas)
        sums = np.cumsum(logs)
        starts = np.r_[0, np.flatnonzero(np.diff(pixels)) + 1]
        lengths = np.diff(np.r_[starts, len(pixels)])
        offsets = np.repeat(np.r_[0.0, sums[starts[1:] - 1]], lengths)
        before = np.exp(sums - logs - offsets) * self.trans[pixels]
        weights = before * alphas
        significant = (weights >= MIN_WEIGHT) & ~self.done[pixels]
        np.minimum.at(self.depth, pixels[significant], depths[significant])
        width = np.maximum(0.001, self.depth[pixels] * self.tolerance)
        front = significant & (depths <= self.depth[pixels] + width)
        np.add.at(self.coverage, pixels[front], weights[front])
        self.candidates.append((ids[front], pixels[front]))
        ends = starts + lengths - 1
        self.trans[pixels[ends]] *= np.exp(sums[ends] - offsets[ends])
        # Once past the front layer, more distant splats cannot become its substitute.
        beyond = np.isfinite(self.depth[pixels]) & (depths > self.depth[pixels] + width)
        self.done[pixels[beyond]] = True
        self.done |= self.trans <= 1e-4

    def result(self):
        confident = (self.coverage >= MIN_COVERAGE) & np.isfinite(self.depth)
        ids = [g[confident[p]] for g, p in self.candidates]
        return (
            np.unique(np.concatenate(ids)) if ids else np.empty(0, dtype=np.int64)
        ), confident


class VisiblePicker:
    def __init__(self, model, camera, base_ids):
        import torch
        from gsplat.cuda._wrapper import (
            fully_fused_projection,
            isect_tiles,
            isect_offset_encode,
        )

        started = time.monotonic()
        self.camera, self.base_ids = camera, base_ids
        means, quats, scales, opacity, _ = model.activated()
        device = means.device
        view = torch.as_tensor(
            camera.camera_from_normalized.copy(), device=device, dtype=torch.float32
        )[None]
        k = torch.as_tensor(
            camera.intrinsic.copy(), device=device, dtype=torch.float32
        )[None]
        radii, self.means, self.depths, self.conics, _ = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            view,
            k,
            camera.width,
            camera.height,
            packed=False,
            opacities=opacity,
        )
        self.opacity = opacity.reshape(1, -1)
        self.tile_size = 16
        tw, th = (camera.width + 15) // 16, (camera.height + 15) // 16
        # Upper bound before allocating the sorted intersection list.
        extents = torch.ceil((2 * radii.to(torch.float32)) / 16).to(torch.int64) + 1
        estimate = (
            (
                (extents[..., 0].clamp(max=tw) * extents[..., 1].clamp(max=th))
                * (radii > 0).all(-1)
            )
            .sum()
            .item()
        )
        if estimate > MAX_INTERSECTIONS:
            raise CloudRenderError(
                "可见选择投影超出内存预算，请缩小画面分辨率；不会改用穿透选择"
            )
        _, intersections, self.flat = isect_tiles(
            self.means, radii, self.depths, 16, tw, th, packed=False
        )
        if len(self.flat) > MAX_INTERSECTIONS:
            raise CloudRenderError("可见选择瓦片数超出预算；未返回部分选择")
        self.offsets = isect_offset_encode(intersections, 1, tw, th)
        offsets = self.offsets.flatten()
        lengths = torch.diff(torch.cat((offsets, offsets.new_tensor([len(self.flat)]))))
        self.batches = (int(lengths.max().item()) + 255) // 256 if len(lengths) else 0
        self.projection_ms = (time.monotonic() - started) * 1000
        if self.projection_ms > 10000:
            raise CloudRenderError(
                "可见选择投影超时，请缩小画面分辨率；不会改用穿透选择"
            )

    def pick(self, polygon, depth_range, tolerance=0.02):
        import torch
        from gsplat.cuda._wrapper import rasterize_to_indices_in_range

        c = self.camera
        p = np.asarray(polygon, dtype=np.float64)
        if (
            p.ndim != 2
            or p.shape[1:] != (2,)
            or not 3 <= len(p) <= 128
            or not np.isfinite(p).all()
        ):
            raise CloudRenderError("无效的可见选区")
        if (p < 0).any() or (p > [c.width, c.height]).any() or not 0 <= tolerance <= 1:
            raise CloudRenderError("可见选区超出固定图片")
        depth = np.asarray(depth_range, dtype=np.float64)
        if (
            depth.shape != (2,)
            or not np.isfinite(depth).all()
            or not 0.01 <= depth[0] < depth[1] <= 1e6
        ):
            raise CloudRenderError("选择深度区间无效")
        started, total = time.monotonic(), 0
        found, depth_samples = [], []
        confident_count = candidate_count = 0
        coverage_tenths = coverage_quarters = 0
        coverage_max = 0.0
        trans = torch.zeros((1, c.height, c.width), device=self.means.device)
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int), [c.width, c.height])
        for top in range(y0, y1, ROI_SIZE):
            for left in range(x0, x1, ROI_SIZE):
                if time.monotonic() - started > MAX_SECONDS:
                    raise CloudRenderError("可见选择超时，请缩小选区；未返回部分选择")
                w, h = min(ROI_SIZE, x1 - left), min(ROI_SIZE, y1 - top)
                yy, xx = np.mgrid[top : top + h, left : left + w]
                roi = polygon_pixels(p, xx + 0.5, yy + 0.5)
                if not roi.any():
                    continue
                layer = FrontLayer(w * h, tolerance)
                layer.done = ~roi.ravel()
                candidate_count += int(roi.sum())
                trans.zero_()
                for batch in range(self.batches):
                    if time.monotonic() - started > MAX_SECONDS:
                        raise CloudRenderError(
                            "可见选择超时，请缩小选区；未返回部分选择"
                        )
                    values = np.where(layer.done, 0, layer.trans).reshape(h, w)
                    if not values.any():
                        break
                    trans[0, top : top + h, left : left + w] = torch.as_tensor(
                        values, device=trans.device, dtype=trans.dtype
                    )
                    ids, pixels, _ = rasterize_to_indices_in_range(
                        batch,
                        batch + 1,
                        trans,
                        self.means,
                        self.conics,
                        self.opacity,
                        c.width,
                        c.height,
                        16,
                        self.offsets,
                        self.flat,
                    )
                    total += len(ids)
                    if total > MAX_PAIRS:
                        raise CloudRenderError(
                            "可见选择贡献数超出预算，请缩小选区；未返回部分选择"
                        )
                    if not len(ids):
                        continue
                    if (
                        (ids < 0).any()
                        or (ids >= self.means.shape[1]).any()
                        or (pixels < 0).any()
                        or (pixels >= c.width * c.height).any()
                    ):
                        raise CloudRenderError("可见选择贡献索引无效；未返回部分选择")
                    x = pixels % c.width + 0.5
                    y = pixels // c.width + 0.5
                    delta = torch.stack((x, y), -1) - self.means[0, ids]
                    conic = self.conics[0, ids]
                    sigma = (
                        0.5
                        * (
                            conic[:, 0] * delta[:, 0] ** 2
                            + conic[:, 2] * delta[:, 1] ** 2
                        )
                        + conic[:, 1] * delta[:, 0] * delta[:, 1]
                    )
                    alpha = (self.opacity[0, ids] * torch.exp(-sigma)).clamp(max=0.999)
                    local = (
                        ((pixels // c.width - top) * w + pixels % c.width - left)
                        .cpu()
                        .numpy()
                    )
                    if (
                        (local < 0).any()
                        or (local >= w * h).any()
                        or layer.done[local].any()
                    ):
                        raise CloudRenderError("可见选择返回区域外贡献；未返回部分选择")
                    depths = self.depths[0, ids].cpu().numpy()
                    alphas = alpha.cpu().numpy()
                    if not np.isfinite(depths).all() or not np.isfinite(alphas).all():
                        raise CloudRenderError(
                            "可见选择贡献包含非有限值；未返回部分选择"
                        )
                    layer.add(ids.cpu().numpy(), local, depths, alphas)
                ids, confident = layer.result()
                confident_count += int(confident.sum())
                coverage_tenths += int(((layer.coverage >= 0.1) & roi.ravel()).sum())
                coverage_quarters += int(((layer.coverage >= 0.25) & roi.ravel()).sum())
                coverage_max = max(coverage_max, float(layer.coverage.max()))
                # Depth limits constrain selection, never remove foreground occluders.
                if len(ids):
                    gpu_ids = torch.as_tensor(
                        ids, device=self.depths.device, dtype=torch.long
                    )
                    z = self.depths[0, gpu_ids]
                    keep = (z >= depth_range[0]) & (z <= depth_range[1])
                    found.append(self.base_ids[gpu_ids[keep]].cpu().numpy())
                depth_samples.extend(layer.depth[confident].tolist())
        return {
            "ids": np.unique(np.concatenate(found))
            if found
            else np.empty(0, dtype=np.int64),
            "depth": float(np.median(depth_samples)) if depth_samples else None,
            "confident_pixels": confident_count,
            "uncertain_pixels": candidate_count - confident_count,
            "candidate_pixels": candidate_count,
            "contribution_pairs": total,
            "depth_batches": self.batches,
            "front_coverage_ge_0_1": coverage_tenths,
            "front_coverage_ge_0_25": coverage_quarters,
            "front_coverage_max": round(coverage_max, 4),
            "selection_ms": (time.monotonic() - started) * 1000,
            "projection_ms": self.projection_ms,
        }
