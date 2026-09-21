"""Bounded native rendering primitives; no network service is started here."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu


MAX_GAUSSIANS = 3_000_000
MAX_PLY_BYTES = MAX_GAUSSIANS * 62 * 4 + 65536


class CloudRenderError(ValueError):
    pass


@dataclass(frozen=True)
class CloudCamera:
    camera_from_normalized: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        for size in (self.width, self.height):
            if type(size) is not int or not 16 <= size <= 1920:
                raise CloudRenderError(
                    "camera dimensions must be integers in [16, 1920]"
                )
        if self.width * self.height > 1920 * 1080:
            raise CloudRenderError("camera exceeds the pixel budget")
        pose = np.array(self.camera_from_normalized, dtype=np.float64, copy=True)
        intrinsic = np.array(self.intrinsic, dtype=np.float64, copy=True)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise CloudRenderError("camera pose must be finite 4 x 4")
        rotation = pose[:3, :3]
        if (
            not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 1e-5
            or np.abs(pose[:3, 3]).max() > 1e6
        ):
            raise CloudRenderError("camera pose must be a bounded rigid transform")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise CloudRenderError("intrinsic must be finite 3 x 3")
        if (
            not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-8, rtol=0)
            or abs(intrinsic[0, 1]) > 1e-8
            or abs(intrinsic[1, 0]) > 1e-8
            or not 1 <= intrinsic[0, 0] <= 1e6
            or not 1 <= intrinsic[1, 1] <= 1e6
            or not 0 <= intrinsic[0, 2] <= self.width
            or not 0 <= intrinsic[1, 2] <= self.height
        ):
            raise CloudRenderError("intrinsic must be a bounded pinhole camera")
        pose.flags.writeable = False
        intrinsic.flags.writeable = False
        object.__setattr__(self, "camera_from_normalized", pose)
        object.__setattr__(self, "intrinsic", intrinsic)

    @classmethod
    def from_json(cls, value: dict) -> CloudCamera:
        if not isinstance(value, dict) or set(value) != {
            "camera_from_normalized",
            "intrinsic",
            "width",
            "height",
        }:
            raise CloudRenderError("invalid camera fields")
        return cls(**value)

    def to_json(self) -> dict:
        return {
            "camera_from_normalized": self.camera_from_normalized.tolist(),
            "intrinsic": self.intrinsic.tolist(),
            "width": self.width,
            "height": self.height,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_json(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


class FrozenFrameTickets:
    """One frozen frame per session; replacing it invalidates its old selections."""

    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._record: dict | None = None

    def issue(
        self, *, source_sha256: str, revision: int, camera_seq: int, camera: CloudCamera
    ) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise CloudRenderError("invalid source hash")
        if any(type(value) is not int or value < 0 for value in (revision, camera_seq)):
            raise CloudRenderError("invalid frame revision/sequence")
        token = secrets.token_urlsafe(32)
        self._record = {
            "token": token,
            "source_sha256": source_sha256,
            "revision": revision,
            "camera_seq": camera_seq,
            "camera_hash": camera.digest,
            "expires": self._clock() + 300,
        }
        return token

    def invalidate(self) -> None:
        self._record = None

    def validate(
        self,
        token: str,
        *,
        source_sha256: str,
        revision: int,
        camera_seq: int,
        camera: CloudCamera,
    ) -> None:
        record = self._record
        if (
            record is None
            or self._clock() >= record["expires"]
            or not isinstance(token, str)
            or not token.isascii()
            or len(token) > 128
            or not secrets.compare_digest(record["token"], token)
            or source_sha256 != record["source_sha256"]
            or revision != record["revision"]
            or camera_seq != record["camera_seq"]
            or camera.digest != record["camera_hash"]
        ):
            raise CloudRenderError("stale or foreign frozen frame")


def validate_ply_source(path: Path, expected_sha256: str, count: int) -> None:
    if type(count) is not int or not 1 <= count <= MAX_GAUSSIANS:
        raise CloudRenderError("Gaussian count exceeds the supported budget")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise CloudRenderError("invalid source hash")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_PLY_BYTES:
        raise CloudRenderError("source PLY is missing or exceeds the size budget")
    if sha256_file(path) != expected_sha256:
        raise CloudRenderError("source PLY hash mismatch")


class CloudRenderProcess:
    """Single-flight IPC. The media layer must coalesce camera inputs before render()."""

    def __init__(
        self,
        *,
        source: Path,
        source_sha256: str,
        count: int,
        lease_path: Path,
        scratch_root: Path,
    ):
        self.source = Path(source)
        self.source_sha256 = source_sha256
        self.count = count
        self.lease_path = Path(lease_path)
        self.scratch_root = Path(scratch_root)
        self._process = None
        self._pipe = None
        self._lock = threading.Lock()

    def start(self, *, timeout: float = 120) -> None:
        with self._lock:
            if self._process is not None:
                raise CloudRenderError("render process already started")
            validate_ply_source(self.source, self.source_sha256, self.count)
            if not self.scratch_root.is_dir() or not self.lease_path.parent.is_dir():
                raise CloudRenderError(
                    "render directories must be provisioned before startup"
                )
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(
                target=_render_worker,
                args=(
                    child,
                    self.source,
                    self.source_sha256,
                    self.count,
                    self.lease_path,
                    self.scratch_root,
                ),
                daemon=True,
            )
            self._process, self._pipe = process, parent
            try:
                process.start()
                child.close()
                self._receive(timeout, "ready")
            except BaseException:
                child.close()
                self._close()
                raise

    def render(
        self,
        camera: CloudCamera,
        *,
        visible: np.ndarray | None = None,
        timeout: float = 30,
    ) -> dict:
        with self._lock:
            if self._pipe is None:
                raise CloudRenderError("render process is not started")
            if visible is not None:
                if (
                    visible.dtype != np.bool_
                    or visible.shape != (self.count,)
                    or not visible.any()
                ):
                    raise CloudRenderError("invalid or empty visible mask")
            try:
                self._pipe.send({"camera": camera.to_json(), "visible": visible})
                result = self._receive(timeout, "frame")
                if result["camera_hash"] != camera.digest:
                    raise CloudRenderError("render response camera mismatch")
                return result
            except BaseException:
                self._close()
                raise

    def _receive(self, timeout: float, expected: str) -> dict:
        if not self._pipe.poll(timeout):
            raise CloudRenderError("render process timed out")
        try:
            result = self._pipe.recv()
        except EOFError as exc:
            raise CloudRenderError("render process exited") from exc
        if result.get("status") != expected:
            raise CloudRenderError(result.get("error", "invalid render response"))
        return result

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._pipe.close()
            self._pipe = None
        if self._process is not None:
            if self._process.pid is not None:
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join()
                self._process.close()
            self._process = None


def _render_worker(pipe, source, expected_hash, count, lease_path, scratch_root):
    try:
        # The child retains this raw descriptor until process exit releases its CUDA context.
        _lease = FileLease(lease_path).acquire()
        require_idle_gpu()
        with tempfile.TemporaryDirectory(
            prefix="cloud-render-", dir=scratch_root
        ) as temporary:
            validate_ply_source(source, expected_hash, count)
            import torch

            from .evaluation import load_model_snapshot
            from .importer import import_inria_ply
            from .model import GaussianModel
            from .render import RenderCamera, render_gaussians

            if not torch.cuda.is_available():
                raise CloudRenderError(
                    "cloud rendering requires the authorized server CUDA environment"
                )
            snapshot = Path(temporary) / "source.pt"
            imported = import_inria_ply(source, snapshot)
            if (
                imported["gaussian_count"] != count
                or imported["source_sha256"] != expected_hash
            ):
                raise CloudRenderError("imported source identity mismatch")
            device = torch.device("cuda:0")
            base = load_model_snapshot(snapshot, device).eval().requires_grad_(False)
            active = base
            pipe.send({"status": "ready", "gaussian_count": count})
            with torch.inference_mode():
                while True:
                    request = pipe.recv()
                    if request is None:
                        break
                    camera = CloudCamera.from_json(request["camera"])
                    visible = request["visible"]
                    if visible is not None:
                        if (
                            visible.dtype != np.bool_
                            or visible.shape != (count,)
                            or not visible.any()
                        ):
                            raise CloudRenderError("invalid visible mask")
                        if visible.all():
                            active = base
                        else:
                            indices = torch.from_numpy(np.flatnonzero(visible)).to(
                                device
                            )
                            state = {
                                key: value[indices]
                                for key, value in base.state_dict().items()
                            }
                            active = (
                                GaussianModel(**state, max_sh_degree=3)
                                .eval()
                                .requires_grad_(False)
                            )
                    rendered = render_gaussians(
                        active,
                        RenderCamera(
                            image_id="cloud",
                            camera_from_normalized=torch.tensor(
                                camera.camera_from_normalized.copy(),
                                dtype=torch.float32,
                                device=device,
                            ),
                            intrinsic=torch.tensor(
                                camera.intrinsic.copy(),
                                dtype=torch.float32,
                                device=device,
                            ),
                            width=camera.width,
                            height=camera.height,
                        ),
                        sh_degree=3,
                    )
                    rgb = rendered.image.clamp(0, 1).mul(255).byte().cpu().numpy()
                    pipe.send(
                        {
                            "status": "frame",
                            "camera_hash": camera.digest,
                            "rgb": rgb,
                            "gaussian_count": active.count,
                        }
                    )
    except EOFError:
        pass
    except Exception as exc:
        try:
            pipe.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        pipe.close()
