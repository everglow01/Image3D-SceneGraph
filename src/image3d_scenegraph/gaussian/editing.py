"""Non-destructive Gaussian selections and file-backed edit documents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

import numpy as np

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gpu_lease import FileLease

from .cloud_render import CloudCamera, MAX_GAUSSIANS, validate_ply_source


class GaussianEditError(ValueError):
    pass


class EditConflict(GaussianEditError):
    pass


def _points(value) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not 1 <= len(points) <= MAX_GAUSSIANS
        or not np.isfinite(points).all()
    ):
        raise GaussianEditError(
            "points must be finite N x 3 within the Gaussian budget"
        )
    return points


def select_box(means, minimum, maximum) -> np.ndarray:
    points = _points(means)
    low, high = (
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
    )
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or not np.isfinite([low, high]).all()
        or not (low < high).all()
    ):
        raise GaussianEditError("box bounds must be finite, ordered 3-vectors")
    return ((points >= low) & (points <= high)).all(axis=1)


def select_polygon(
    means, camera: CloudCamera, polygon, depth_range, *, radii=None
) -> np.ndarray:
    """Select centers, or conservative 3-sigma sphere bounds when radii are supplied."""
    points = _points(means)
    polygon = np.asarray(polygon, dtype=np.float64)
    depth = np.asarray(depth_range, dtype=np.float64)
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or not 3 <= len(polygon) <= 128
        or not np.isfinite(polygon).all()
    ):
        raise GaussianEditError("polygon must have 3..128 finite pixel vertices")
    if (polygon < 0).any() or (polygon > [camera.width, camera.height]).any():
        raise GaussianEditError("polygon lies outside the frozen image")
    area = (
        abs(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], 1)
                - polygon[:, 1] * np.roll(polygon[:, 0], 1)
            )
        )
        / 2
    )
    if area < 1:
        raise GaussianEditError("polygon area is too small")
    if (
        depth.shape != (2,)
        or not np.isfinite(depth).all()
        or not 0.01 <= depth[0] < depth[1] <= 1e6
    ):
        raise GaussianEditError("depth interval must satisfy 0.01 <= near < far <= 1e6")
    if radii is not None:
        radii = np.asarray(radii, dtype=np.float64)
        if (
            radii.shape != (len(points),)
            or not np.isfinite(radii).all()
            or (radii < 0).any()
        ):
            raise GaussianEditError("support radii must be finite nonnegative N values")
    result = np.zeros(len(points), dtype=bool)
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    for start in range(0, len(points), 65536):
        stop = min(start + 65536, len(points))
        xyz = (
            points[start:stop] @ camera.camera_from_normalized[:3, :3].T
            + camera.camera_from_normalized[:3, 3]
        )
        z = xyz[:, 2]
        if radii is None:
            valid = (z >= depth[0]) & (z <= depth[1])
            uv = xyz[:, :2] / np.maximum(z[:, None], 0.01)
            uv = (
                uv * [camera.intrinsic[0, 0], camera.intrinsic[1, 1]]
                + camera.intrinsic[:2, 2]
            )
            inside = np.zeros(len(xyz), dtype=bool)
            boundary = inside.copy()
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                delta = b - a
                cross = (uv[:, 0] - a[0]) * delta[1] - (uv[:, 1] - a[1]) * delta[0]
                boundary |= (
                    (np.abs(cross) < 1e-7)
                    & (uv >= np.minimum(a, b)).all(axis=1)
                    & (uv <= np.maximum(a, b)).all(axis=1)
                )
                if delta[1] != 0:
                    inside ^= ((a[1] > uv[:, 1]) != (b[1] > uv[:, 1])) & (
                        uv[:, 0] < a[0] + (uv[:, 1] - a[1]) * delta[0] / delta[1]
                    )
            result[start:stop] = valid & (inside | boundary)
        else:
            radius = radii[start:stop]
            valid = (z + radius >= depth[0]) & (z - radius <= depth[1])
            near, far = np.maximum(z - radius, 0.01), np.maximum(z + radius, 0.01)
            intersects = valid.copy()
            for axis in range(2):
                numerators = np.stack((xyz[:, axis] - radius, xyz[:, axis] + radius))
                projected = np.concatenate((numerators / near, numerators / far))
                projected = (
                    projected * camera.intrinsic[axis, axis] + camera.intrinsic[axis, 2]
                )
                # Bounds overlapping the near plane are deliberately conservative candidates.
                intersects &= (z - radius <= 0.01) | (
                    (projected.max(axis=0) >= low[axis])
                    & (projected.min(axis=0) <= high[axis])
                )
            result[start:stop] = intersects
    return result


def resolve_edit_source(
    store,
    job_id: str,
    *,
    variant_id: str | None = None,
    asset_role: str = "scene_splat",
) -> dict:
    manifest = store.get_manifest(job_id)
    if (
        manifest.get("job_id") != job_id
        or manifest.get("status") != "done"
        or manifest.get("output_type") != "gaussian_splat"
    ):
        raise GaussianEditError("editing requires a completed Gaussian result")
    if variant_id is not None:
        if asset_role != "scene_splat":
            raise GaussianEditError("variant and derivative role cannot be combined")
        matches = [
            row
            for row in manifest.get("gaussian_variants", [])
            if row.get("id") == variant_id
        ]
        if len(matches) != 1:
            raise GaussianEditError("unknown or duplicate Gaussian variant")
        ply_asset, metadata_asset = (
            matches[0]["scene_splat"],
            matches[0]["export_metadata"],
        )
    else:
        metadata_role = {
            "scene_splat": "gaussian_export_metadata",
            "scene_splat_vggt_filtered": "gaussian_vggt_filtered_export_metadata",
        }.get(asset_role)
        if metadata_role is None:
            raise GaussianEditError("unsupported editable asset role")
        assets = manifest.get("assets", {})
        ply_asset, metadata_asset = assets.get(asset_role), assets.get(metadata_role)
    if not isinstance(ply_asset, str) or not isinstance(metadata_asset, str):
        raise GaussianEditError("source assets are incomplete")
    for asset in (ply_asset, metadata_asset):
        if Path(asset).is_absolute() or ".." in Path(asset).parts:
            raise GaussianEditError("source assets must be contained relative paths")
    ply = store.get_asset_path(job_id, ply_asset)
    metadata_path = store.get_asset_path(job_id, metadata_asset)
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("coordinate_frame"),
        metadata.get("world_units"),
        metadata.get("sh_degree"),
    ) != ("normalized", "arbitrary", 3):
        raise GaussianEditError("source coordinate or SH contract is unsupported")
    transform = np.asarray(metadata.get("world_from_normalized"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise GaussianEditError("source normalization is invalid")
    linear = transform[:3, :3]
    scale = np.linalg.norm(linear[:, 0])
    if (
        scale <= 0
        or np.linalg.det(linear) <= 0
        or not np.allclose(linear.T @ linear, np.eye(3) * scale**2, atol=1e-7)
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
    ):
        raise GaussianEditError("source normalization must be a positive similarity")
    count = metadata.get("gaussian_count")
    digest = metadata.get("browser_sha256", "")
    validate_ply_source(ply, digest, count)
    return {
        "job_id": job_id,
        "variant_id": variant_id,
        "asset_role": asset_role,
        "ply_asset": ply_asset,
        "metadata_asset": metadata_asset,
        "ply_sha256": digest,
        "metadata_sha256": sha256_file(metadata_path),
        "gaussian_count": count,
        "world_from_normalized": transform.tolist(),
    }


def load_edit_source_rows(store, source: dict) -> np.ndarray:
    from .export import PLY_FIELDS, read_gaussian_ply

    current = resolve_edit_source(
        store,
        source["job_id"],
        variant_id=source["variant_id"],
        asset_role=source["asset_role"],
    )
    if current != source:
        raise EditConflict("the frozen source has changed")
    path = store.get_asset_path(source["job_id"], source["ply_asset"])
    fields = read_gaussian_ply(path)
    rows = np.column_stack([fields[name] for name in PLY_FIELDS])
    if (
        len(rows) != source["gaussian_count"]
        or sha256_file(path) != source["ply_sha256"]
    ):
        raise EditConflict("source count/hash changed during loading")
    return rows


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n"
    ).encode()


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    _write_new(temporary, _json_bytes(value))
    os.replace(temporary, path)
    _sync_dir(path.parent)


class GaussianEditStore:
    def __init__(self, job_store, root: Path | None = None):
        self.jobs = job_store
        self.root = (
            Path(root) if root is not None else job_store.output_root.parent / "edits"
        )

    def _directory(self, edit_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", edit_id):
            raise GaussianEditError("invalid edit ID")
        path = self.root / edit_id
        if path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise GaussianEditError("edit directory escapes its root")
        if not (path / "edit.json").is_file():
            raise FileNotFoundError(edit_id)
        for child in ("edit.json", "operations", "versions"):
            candidate = path / child
            if candidate.is_symlink() or candidate.resolve().parent != path.resolve():
                raise GaussianEditError("edit assets escape their directory")
        return path

    def get(self, edit_id: str) -> dict:
        directory = self._directory(edit_id)
        state = json.loads((directory / "edit.json").read_text())
        if state.get("schema_version") != 1 or state.get("edit_id") != edit_id:
            raise GaussianEditError("edit document identity mismatch")
        return state

    def _mask_path(self, directory: Path, row: dict) -> Path:
        relative = row["path"]
        if not re.fullmatch(r"operations/[0-9a-f]{32}\.bin", relative):
            raise GaussianEditError("invalid mask path")
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise GaussianEditError("mask path escapes edit directory")
        return path

    def _read_mask(self, directory: Path, row: dict, count: int) -> np.ndarray:
        path = self._mask_path(directory, row)
        if (
            type(count) is not int
            or not 1 <= count <= MAX_GAUSSIANS
            or path.stat().st_size != (count + 7) // 8
        ):
            raise GaussianEditError("mask size/count mismatch")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise GaussianEditError("mask hash mismatch")
        mask = np.unpackbits(
            np.frombuffer(content, dtype=np.uint8), bitorder="little", count=count
        ).astype(bool)
        if np.packbits(mask, bitorder="little").tobytes() != content:
            raise GaussianEditError("mask has nonzero padding")
        return mask

    def _new_mask(self, directory: Path, visible: np.ndarray) -> dict:
        content = np.packbits(visible, bitorder="little").tobytes()
        name = f"operations/{uuid.uuid4().hex}.bin"
        _write_new(directory / name, content)
        _sync_dir(directory / "operations")
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    def visible(self, edit_id: str) -> np.ndarray:
        state = self.get(edit_id)
        return self._read_mask(
            self._directory(edit_id),
            state["history"][state["cursor"]],
            state["source"]["gaussian_count"],
        )

    def create(
        self,
        job_id: str,
        *,
        variant_id: str | None = None,
        asset_role: str = "scene_splat",
    ) -> dict:
        source = resolve_edit_source(
            self.jobs, job_id, variant_id=variant_id, asset_role=asset_role
        )
        rows = load_edit_source_rows(self.jobs, source)
        del rows
        self.root.mkdir(parents=True, exist_ok=True)
        edit_id = uuid.uuid4().hex
        staging = self.root / f".{edit_id}.staging"
        staging.mkdir()
        (staging / "operations").mkdir()
        (staging / "versions").mkdir()
        mask = self._new_mask(staging, np.ones(source["gaussian_count"], dtype=bool))
        state = {
            "schema_version": 1,
            "edit_id": edit_id,
            "profile": "manual_trim_v1",
            "source": source,
            "revision": 0,
            "cursor": 0,
            "history": [mask],
            "requests": {},
            "quality_role": "manual_edit_not_evaluated",
            "navigation_status": "not_inherited",
        }
        _write_new(staging / "edit.json", _json_bytes(state))
        _sync_dir(staging)
        staging.rename(self.root / edit_id)
        _sync_dir(self.root)
        return state

    def apply(
        self,
        edit_id: str,
        *,
        operation_id: str,
        expected_revision: int,
        kind: str,
        selected: np.ndarray | None = None,
        confirm_large: bool = False,
        api_request_sha256: str | None = None,
    ) -> dict:
        if api_request_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", api_request_sha256
        ):
            raise GaussianEditError("invalid API request digest")
        if not isinstance(operation_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", operation_id
        ):
            raise GaussianEditError("invalid operation ID")
        if type(confirm_large) is not bool:
            raise GaussianEditError("large-deletion confirmation must be boolean")
        if (
            kind not in {"delete", "undo", "redo"}
            or type(expected_revision) is not int
            or expected_revision < 0
        ):
            raise GaussianEditError("invalid operation/revision")
        if kind == "delete" and (
            not isinstance(selected, np.ndarray)
            or selected.dtype != np.bool_
            or selected.ndim != 1
        ):
            raise GaussianEditError("delete requires a boolean selection")
        if kind != "delete" and selected is not None:
            raise GaussianEditError("undo/redo cannot include a selection")
        directory = self._directory(edit_id)
        request = {
            "kind": kind,
            "expected_revision": expected_revision,
            "confirm_large": confirm_large,
            "selection_count": len(selected) if selected is not None else None,
            "selection_sha256": hashlib.sha256(
                np.packbits(selected, bitorder="little").tobytes()
            ).hexdigest()
            if selected is not None
            else None,
        }
        request_hash = hashlib.sha256(_json_bytes(request)).hexdigest()
        with FileLease(directory / ".edit.lock"):
            state = self.get(edit_id)
            previous = state["requests"].get(operation_id)
            if previous is not None:
                if (
                    previous["request_hash"] != request_hash
                    or previous.get("api_request_sha256") != api_request_sha256
                ):
                    raise EditConflict(
                        "operation ID was reused with different parameters"
                    )
                return previous["result"]
            if state["revision"] != expected_revision:
                raise EditConflict("edit revision has changed")
            if len(state["requests"]) >= 1000:
                raise GaussianEditError(
                    "document operation budget reached; saved versions remain available"
                )
            count = state["source"]["gaussian_count"]
            visible = self._read_mask(
                directory, state["history"][state["cursor"]], count
            )
            if kind == "delete":
                if selected.shape != (count,):
                    raise GaussianEditError("selection length does not match source")
                removed = selected & visible
                if not removed.any() or np.array_equal(removed, visible):
                    raise GaussianEditError(
                        "deletion must remove some but not all visible Gaussians"
                    )
                if removed.sum() / visible.sum() > 0.5 and not confirm_large:
                    raise GaussianEditError(
                        "deleting more than half the visible model requires confirmation"
                    )
                visible = visible & ~removed
                state["history"] = state["history"][: state["cursor"] + 1] + [
                    self._new_mask(directory, visible)
                ]
                state["history"] = state["history"][-101:]
                state["cursor"] = len(state["history"]) - 1
            else:
                cursor = state["cursor"] + (-1 if kind == "undo" else 1)
                if not 0 <= cursor < len(state["history"]):
                    raise GaussianEditError(f"nothing to {kind}")
                state["cursor"] = cursor
                visible = self._read_mask(directory, state["history"][cursor], count)
            state["revision"] += 1
            result = {
                "edit_id": edit_id,
                "revision": state["revision"],
                "visible_count": int(visible.sum()),
                "can_undo": state["cursor"] > 0,
                "can_redo": state["cursor"] + 1 < len(state["history"]),
            }
            state["requests"][operation_id] = {
                "request_hash": request_hash,
                "api_request_sha256": api_request_sha256,
                "result": result,
            }
            _replace_json(directory / "edit.json", state)
            return result

    def source_identity(self, edit_id: str) -> dict:
        source = self.get(edit_id)["source"]
        return {key: source[key] for key in (
            "ply_sha256", "metadata_sha256", "gaussian_count"
        )}

    def check_source(self, edit_id: str, identity: dict) -> dict:
        if (
            not isinstance(identity, dict)
            or set(identity) != {"ply_sha256", "metadata_sha256", "gaussian_count"}
            or type(identity["gaussian_count"]) is not int
            or any(not isinstance(identity[key], str) or not re.fullmatch(
                r"[0-9a-f]{64}", identity[key]
            ) for key in ("ply_sha256", "metadata_sha256"))
        ):
            raise GaussianEditError("invalid local source identity")
        source = self.get(edit_id)["source"]
        if identity != self.source_identity(edit_id):
            raise EditConflict("local source identity does not match document")
        current = resolve_edit_source(
            self.jobs, source["job_id"], variant_id=source["variant_id"],
            asset_role=source["asset_role"],
        )
        if current != source:
            raise EditConflict("the frozen source has changed")
        return source

    def local_snapshot(self, edit_id: str, identity: dict) -> tuple[dict, bytes]:
        directory = self._directory(edit_id)
        with FileLease(directory / ".edit.lock"):
            self.check_source(edit_id, identity)
            state = self.get(edit_id)
            visible = self._read_mask(
                directory, state["history"][state["cursor"]],
                state["source"]["gaussian_count"],
            )
            return {
                "revision": state["revision"], "visible_count": int(visible.sum()),
            }, np.packbits(visible, bitorder="little").tobytes()

    def submit_snapshot(
        self, edit_id: str, *, identity: dict, content: bytes,
        expected_revision: int, operation_id: str, confirm_large: bool = False,
    ) -> dict:
        if (
            type(expected_revision) is not int or expected_revision < 0
            or type(confirm_large) is not bool
            or not isinstance(operation_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", operation_id)
            or not isinstance(content, bytes) or len(content) > 512 * 1024
        ):
            raise GaussianEditError("invalid local snapshot request")
        directory = self._directory(edit_id)
        with FileLease(directory / ".edit.lock"):
            source = self.check_source(edit_id, identity)
            count = source["gaussian_count"]
            if len(content) != (count + 7) // 8:
                raise GaussianEditError("mask size/count mismatch")
            visible = np.unpackbits(
                np.frombuffer(content, dtype=np.uint8), bitorder="little", count=count
            ).astype(bool)
            if np.packbits(visible, bitorder="little").tobytes() != content:
                raise GaussianEditError("mask has nonzero padding")
            if not visible.any():
                raise GaussianEditError("local snapshot cannot hide the entire model")
            digest = hashlib.sha256(_json_bytes({
                "kind": "local_snapshot_v1", "identity": identity,
                "expected_revision": expected_revision, "confirm_large": confirm_large,
                "mask_sha256": hashlib.sha256(content).hexdigest(),
            })).hexdigest()
            state = self.get(edit_id)
            previous = state["requests"].get(operation_id)
            if previous is not None:
                if previous["request_hash"] != digest:
                    raise EditConflict("operation ID was reused with different parameters")
                return previous["result"]
            if state["revision"] != expected_revision:
                raise EditConflict("edit revision has changed")
            if len(state["requests"]) >= 1000:
                raise GaussianEditError("document operation budget reached")
            old = self._read_mask(directory, state["history"][state["cursor"]], count)
            if (old & ~visible).sum() > old.sum() / 2 and not confirm_large:
                raise GaussianEditError(
                    "deleting more than half the visible model requires confirmation"
                )
            state["history"] = (state["history"][:state["cursor"] + 1] + [
                self._new_mask(directory, visible)
            ])[-101:]
            state["cursor"] = len(state["history"]) - 1
            state["revision"] += 1
            result = {
                "edit_id": edit_id, "revision": state["revision"],
                "visible_count": int(visible.sum()),
                "can_undo": state["cursor"] > 0, "can_redo": False,
            }
            state["requests"][operation_id] = {
                "request_hash": digest, "result": result,
            }
            _replace_json(directory / "edit.json", state)
            return result

    def acknowledged_request(
        self, edit_id: str, operation_id: str, digest: str
    ) -> dict | None:
        previous = self.get(edit_id)["requests"].get(operation_id)
        if previous is None:
            return None
        if previous.get("api_request_sha256") != digest:
            raise EditConflict("operation ID was reused with different parameters")
        return previous["result"]

    def save_version(self, edit_id: str, *, expected_revision: int) -> dict:
        directory = self._directory(edit_id)
        with FileLease(directory / ".edit.lock"):
            state = self.get(edit_id)
            if (
                type(expected_revision) is not int
                or state["revision"] != expected_revision
            ):
                raise EditConflict("edit revision has changed")
            visible = self.visible(edit_id)
            version = f"v{expected_revision:08d}"
            destination = directory / "versions" / version
            if destination.exists():
                return self.version(edit_id, version)
            staging = directory / "versions" / f".{uuid.uuid4().hex}.staging"
            staging.mkdir()
            _write_new(
                staging / "visible-mask.bin",
                np.packbits(visible, bitorder="little").tobytes(),
            )
            manifest = {
                "schema_version": 1,
                "edit_id": edit_id,
                "version": version,
                "revision": expected_revision,
                "source": state["source"],
                "visible_count": int(visible.sum()),
                "mask_sha256": sha256_file(staging / "visible-mask.bin"),
                "mask_encoding": "packbits_little_v1",
                "quality_role": "manual_edit_not_evaluated",
                "navigation_status": "not_inherited",
            }
            _write_new(staging / "edit-manifest.json", _json_bytes(manifest))
            _sync_dir(staging)
            staging.rename(destination)
            _sync_dir(destination.parent)
            return manifest

    def version(self, edit_id: str, version: str) -> dict:
        if not re.fullmatch(r"v[0-9]{8}", version):
            raise GaussianEditError("invalid version ID")
        directory = self._directory(edit_id)
        path = directory / "versions" / version
        if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise GaussianEditError("version escapes edit directory")
        manifest_path = path / "edit-manifest.json"
        if manifest_path.is_symlink():
            raise GaussianEditError("version manifest cannot be a symlink")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("edit_id") != edit_id
            or manifest.get("version") != version
            or manifest.get("schema_version") != 1
            or manifest.get("revision") != int(version[1:])
            or manifest.get("source") != self.get(edit_id)["source"]
        ):
            raise GaussianEditError("version identity mismatch")
        return manifest

    def describe(self, edit_id: str) -> dict:
        state = self.get(edit_id)
        directory = self._directory(edit_id)
        versions = []
        for path in sorted((directory / "versions").glob("v*")):
            if not re.fullmatch(r"v[0-9]{8}", path.name):
                continue
            version = self.version(edit_id, path.name)
            try:
                for name in ("scene.ply", "export.json", "bundle.zip"):
                    self.export_asset(edit_id, path.name, name)
                version["exported"] = True
            except FileNotFoundError:
                version["exported"] = False
            versions.append(version)
        visible = self._read_mask(
            directory,
            state["history"][state["cursor"]],
            state["source"]["gaussian_count"],
        )
        return {
            key: state[key]
            for key in (
                "edit_id",
                "source",
                "revision",
                "quality_role",
                "navigation_status",
            )
        } | {
            "visible_count": int(visible.sum()),
            "can_undo": state["cursor"] > 0,
            "can_redo": state["cursor"] + 1 < len(state["history"]),
            "versions": versions,
        }

    def list_documents(self, job_id: str) -> list[dict]:
        result = []
        for path in sorted(self.root.glob("*")):
            if re.fullmatch(r"[0-9a-f]{32}", path.name):
                state = self.get(path.name)
                if state["source"]["job_id"] == job_id:
                    result.append(self.describe(path.name))
        return result

    def version_visible(self, edit_id: str, version: str) -> np.ndarray:
        manifest = self.version(edit_id, version)
        path = self._directory(edit_id) / "versions" / version / "visible-mask.bin"
        count = manifest["source"]["gaussian_count"]
        if (
            path.is_symlink()
            or path.stat().st_size != (count + 7) // 8
            or sha256_file(path) != manifest["mask_sha256"]
        ):
            raise GaussianEditError("saved mask identity mismatch")
        content = path.read_bytes()
        visible = np.unpackbits(
            np.frombuffer(content, dtype=np.uint8), bitorder="little", count=count
        ).astype(bool)
        if (
            int(visible.sum()) != manifest["visible_count"]
            or not visible.any()
            or np.packbits(visible, bitorder="little").tobytes() != content
        ):
            raise GaussianEditError("saved mask count/padding mismatch")
        return visible

    def export_asset(self, edit_id: str, version: str, name: str) -> Path:
        self.version(edit_id, version)
        if name not in {"scene.ply", "export.json", "bundle.zip"}:
            raise GaussianEditError("unsupported export asset")
        base = self._directory(edit_id) / "versions" / version
        directory = base / "export"
        path = directory / name
        if (
            directory.is_symlink()
            or path.is_symlink()
            or path.resolve().parent != directory.resolve()
            or directory.resolve().parent != base.resolve()
        ):
            raise GaussianEditError("export path escapes version")
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def export_version(self, edit_id: str, version: str) -> dict:
        from .export import write_binary_ply, write_deterministic_zip

        directory = self._directory(edit_id)
        with FileLease(directory / ".edit.lock"):
            manifest = self.version(edit_id, version)
            base = directory / "versions" / version
            destination = base / "export"
            if destination.exists():
                raise FileExistsError(destination)
            source = manifest["source"]
            mask_path = base / "visible-mask.bin"
            if (
                mask_path.is_symlink()
                or mask_path.stat().st_size != (source["gaussian_count"] + 7) // 8
                or sha256_file(mask_path) != manifest["mask_sha256"]
            ):
                raise GaussianEditError("saved mask identity mismatch")
            visible = np.unpackbits(
                np.frombuffer(mask_path.read_bytes(), dtype=np.uint8),
                bitorder="little",
                count=source["gaussian_count"],
            ).astype(bool)
            if int(visible.sum()) != manifest["visible_count"] or not visible.any():
                raise GaussianEditError("saved count does not match mask")
            source_path = self.jobs.get_asset_path(
                source["job_id"], source["ply_asset"]
            )
            required = 3 * source_path.stat().st_size + 1024**3
            if shutil.disk_usage(directory).free < required:
                raise GaussianEditError("insufficient disk for immutable edit export")
            rows = load_edit_source_rows(self.jobs, source)
            staging = base / f".export-{uuid.uuid4().hex}.staging"
            staging.mkdir()
            write_binary_ply(staging / "scene.ply", rows[visible])
            record = {
                "schema_version": 1,
                "profile": "manual_trim_export_v1",
                "edit_id": edit_id,
                "version": version,
                "source_ply_sha256": source["ply_sha256"],
                "mask_sha256": manifest["mask_sha256"],
                "browser_sha256": sha256_file(staging / "scene.ply"),
                "gaussian_count": int(visible.sum()),
                "removed_count": int((~visible).sum()),
                "sh_degree": 3,
                "coordinate_frame": "normalized",
                "world_units": "arbitrary",
                "world_from_normalized": source["world_from_normalized"],
                "quality_role": "manual_edit_not_evaluated",
                "navigation_status": "not_inherited",
            }
            _write_new(staging / "export.json", _json_bytes(record))
            write_deterministic_zip(
                staging / "bundle.zip",
                {
                    "scene.ply": staging / "scene.ply",
                    "export.json": staging / "export.json",
                    "edit-manifest.json": base / "edit-manifest.json",
                    "visible-mask.bin": mask_path,
                },
            )
            if sha256_file(source_path) != source["ply_sha256"]:
                raise EditConflict("source changed during export")
            for path in staging.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            _sync_dir(staging)
            staging.rename(destination)
            _sync_dir(base)
            return record
