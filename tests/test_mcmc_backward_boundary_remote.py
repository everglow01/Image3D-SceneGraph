"""Remote opt-in boundary check reusing the native first-divergence fixture."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if socket.gethostname() != "i-94B8D131" or os.environ.get("IMAGE3D_AUTHORIZE_BACKWARD_BOUNDARY_TESTS") != "1":
    pytest.skip("仅明确授权远端运行反向边界检查", allow_module_level=True)

from scripts import mcmc_first_divergence as snapshots  # noqa: E402
from scripts import run_mcmc_backward_boundary as gate  # noqa: E402


def boundary_worker(local_rank, rank, world_size, args):
    path = Path(__file__).with_name("test_mcmc_first_divergence_remote.py")
    spec = importlib.util.spec_from_file_location("first_divergence_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    with patch.object(snapshots, "FirstDivergenceProbe", gate.BoundaryProbe):
        fixture.native_worker(local_rank, rank, world_size, args)
    record = json.loads((args.root / f"boundary-rank-{rank}.json").read_text())
    assert record["order"] == list(gate.ORDER)
    assert record["original_raster_backward_calls"] == 1
    for stage in gate.BOUNDARIES:
        folder = args.root / f"rank-{rank}" / stage
        gate.validate_boundary_schema(snapshots.load_snapshot(folder), stage)
        rows = snapshots.compare_snapshots(folder, folder)
        assert all(v["status"] == "bitwise_equal" for v in rows.values())


def test_native_raster_backward_boundaries_and_no_extra_call(tmp_path):
    import torch
    from gsplat.distributed import cli
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    from scripts.run_mcmc_stability_ablation import PROJECT
    if torch.cuda.device_count() != 2:
        pytest.skip("需要两张空闲GPU")
    if os.environ.get("IMAGE3D_AUTHORIZE_FIRST_DIVERGENCE_TESTS") != "1":
        raise RuntimeError("native fixture requires explicit opt-in")
    with FileLease(PROJECT / "outputs/.gpu.lock"):
        require_idle_gpu()
        cli(boundary_worker, SimpleNamespace(root=Path(tmp_path)), verbose=True)
