"""Remote opt-in only; importing this module locally never imports Torch or NumPy."""
from __future__ import annotations

import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

if socket.gethostname() != "i-94B8D131" or os.environ.get("IMAGE3D_AUTHORIZE_FIRST_DIVERGENCE_TESTS") != "1":
    pytest.skip("仅明确授权远端运行首差张量/CUDA检查", allow_module_level=True)

from scripts import mcmc_first_divergence as probe  # noqa: E402


def test_tensor_metrics_match_independent_reference():
    import numpy as np
    a = np.array([0., -0., 1e-8, 2.0005, float("nan"), float("inf")], dtype=np.float64)
    b = np.array([0., 0., 0., 2., float("nan"), float("inf")], dtype=np.float64)
    report = probe.compare_arrays(a, b)
    reference = [probe.numeric_pair(float(x), float(y)) for x, y in zip(a, b)]
    assert report["num_mismatch"] == sum(v["mismatch"] for v in reference)
    assert report["num_nonfinite"] == 2 and report["relative_unbounded_count"] == 1
    assert report["num_strict_unequal"] == sum(v["strict_unequal"] for v in reference)
    assert report["num_bitwise_unequal"] == 3
    assert report["first_mismatch_index"] == [3]
    assert report["first_strict_unequal_index"] == [2]
    assert report["max_rel_error"] is None and report["relative_unbounded"]
    assert probe.compare_arrays(np.array([2**63-1]), np.array([2**63-2]))["max_abs_error"] == 1
    assert probe.compare_arrays(np.array(1.), np.array(2.))["first_mismatch_index"] == []
    assert probe.compare_arrays(np.empty((0, 3)), np.empty((0, 3)))["status"] == "bitwise_equal"
    assert probe.compare_arrays(np.array([1.]), np.array([1.], dtype=np.float32))["status"] == "structure_mismatch"
    assert probe.compare_arrays(np.array([1.00000001]), np.array([1.]))["status"] == "numerically_equivalent"
    assert probe.compare_arrays(np.array([1.00000001]), np.array([1.]), exact=True)["status"] == "numerically_inconsistent"


def test_cpu_snapshot_is_independent_and_tampering_rejected(tmp_path):
    import numpy as np
    import torch
    value = torch.tensor([[1., 2.], [3., 4.]])
    probe.save_snapshot(tmp_path / "a", {"value": value, "rng": np.arange(3, dtype=np.uint32)})
    value.add_(10)
    saved = probe.load_snapshot(tmp_path / "a")
    array = np.load(tmp_path / "a" / saved["tree"]["items"][1][1]["file"], allow_pickle=False)
    assert array[0, 0] == 1.
    probe.save_snapshot(tmp_path / "b", {"value": value, "rng": np.arange(3, dtype=np.uint32)})
    assert probe.compare_snapshots(tmp_path / "a", tmp_path / "b")["value"]["status"] == "numerically_inconsistent"
    tensor = next((tmp_path / "a").glob("*.npy"))
    tensor.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        probe.load_snapshot(tmp_path / "a")


def native_worker(local_rank, rank, world_size, args):
    import json
    import numpy as np
    import torch
    from unittest.mock import patch
    from image3d_scenegraph.gaussian import trainer
    from image3d_scenegraph.gaussian.dataset import contract_hash, write_contract
    from image3d_scenegraph.gaussian.initialization import InitializationResult
    from scripts.run_camera_size_lifecycle import lifecycle_config
    from scripts.smoke_gaussian_trainer import generate_scene

    assert world_size == 2 and rank == local_rank
    torch.cuda.set_device(rank)
    dataset, training = args.root / "dataset", args.root / "training"
    if rank == 0:
        contract = generate_scene(dataset)
        contract["splits"] = {"train": [str(i) for i in range(1, 9)], "validation": ["9", "10"], "test": ["11", "12"]}
        contract["dataset_hash"] = contract_hash(contract)
        write_contract(dataset / "dataset.json", contract)
    torch.distributed.barrier()
    contract = json.loads((dataset / "dataset.json").read_text())
    init = InitializationResult(points=np.random.RandomState(7).uniform(-.15, .15, (60, 3)).astype(np.float32),
        colors=np.full((60, 3), 128, dtype=np.uint8), scales=np.full(60, .02, dtype=np.float32), diagnostics={"kind": "probe_test"})
    kwargs = dict(contract=contract, dataset_root=dataset, initialization=init, resolved_config=lifecycle_config(),
                  run_dir=training, local_rank=rank, world_rank=rank, world_size=2, save_intermediate_previews=False)
    trainer.train_gaussians(**kwargs, stop_after_iteration=1)
    instance = probe.FirstDivergenceProbe(args.root, rank, iteration=2, expected_count=30)
    def forbidden(*a, **kw):
        raise AssertionError("probe continued into Validation/checkpoint")
    with patch.object(trainer, "evaluate_views", forbidden), patch.object(trainer, "_write_streaming_checkpoint", forbidden):
        with pytest.raises(probe.ProbeComplete):
            with instance.installed():
                trainer.train_gaussians(**kwargs, attempt_id="probe-001", attempt_kind="resume",
                    parent_attempt_id="train-001", resume_iteration=1, stop_after_iteration=2)
    instance.finish()
    assert instance.stages == list(probe.STAGES)
    for stage in probe.STAGES:
        report = probe.compare_snapshots(args.root / f"rank-{rank}" / stage, args.root / f"rank-{rank}" / stage)
        assert all(v["status"] == "bitwise_equal" for v in report.values())
    assert not (training / "attempts/probe-001/artifacts/result.json").exists()
    assert not (training / "attempts/probe-001/checkpoints").exists()
    torch.distributed.barrier()


def test_native_restore_five_stages_and_probe_only_exit(tmp_path):
    import torch
    from gsplat.distributed import cli
    from image3d_scenegraph.gpu_lease import FileLease, require_idle_gpu
    from scripts.run_mcmc_stability_ablation import PROJECT
    if torch.cuda.device_count() != 2:
        pytest.skip("需要两张空闲GPU")
    with FileLease(PROJECT / "outputs/.gpu.lock"):
        require_idle_gpu()
        cli(native_worker, SimpleNamespace(root=Path(tmp_path)), verbose=True)
