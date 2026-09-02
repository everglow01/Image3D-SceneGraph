from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


class FileIntegrityError(ValueError):
    """Raised when a managed file does not match its pinned metadata."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str = "asset",
) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise FileIntegrityError(
            f"{label} size mismatch for {path}: expected {expected_size}, "
            f"got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise FileIntegrityError(
            f"{label} SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )


def install_verified_file(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str = "asset",
    attempts: int = 3,
) -> bool:
    """Install a pinned file atomically; return whether it was downloaded."""
    if destination.is_file():
        try:
            verify_file(
                destination,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                label=label,
            )
        except FileIntegrityError:
            pass
        else:
            return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        for attempt in range(1, attempts + 1):
            try:
                urllib.request.urlretrieve(url, temporary)
                verify_file(
                    temporary,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    label=label,
                )
                os.replace(temporary, destination)
                return True
            except (urllib.error.URLError, FileIntegrityError) as exc:
                temporary.unlink(missing_ok=True)
                if attempt == attempts:
                    raise FileIntegrityError(str(exc)) from exc
                print(
                    f"download_retry={attempt}/{attempts} "
                    f"asset={destination.name} error={exc}",
                    file=sys.stderr,
                )
    finally:
        temporary.unlink(missing_ok=True)
    raise AssertionError("unreachable")
