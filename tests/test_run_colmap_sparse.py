from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_colmap_sparse
from scripts.run_colmap_sparse import colmap_version, find_largest_sparse_model, read_sparse_model_counts


def _write_binary_count(path, count: int) -> None:
    path.write_bytes(struct.pack("<Q", count))


def test_sparse_model_selection_prefers_registered_images_then_points(tmp_path):
    first = tmp_path / "0"
    second = tmp_path / "1"
    third = tmp_path / "2"
    for path in (first, second, third):
        path.mkdir()

    _write_binary_count(first / "images.bin", 12)
    _write_binary_count(first / "points3D.bin", 5_000)
    _write_binary_count(second / "images.bin", 15)
    _write_binary_count(second / "points3D.bin", 1_000)
    _write_binary_count(third / "images.bin", 15)
    _write_binary_count(third / "points3D.bin", 2_000)

    assert read_sparse_model_counts(first) == (12, 5_000)
    assert find_largest_sparse_model(tmp_path) == (third, 15, 2_000)


def test_sparse_model_counts_support_text_models(tmp_path):
    model = tmp_path / "0"
    model.mkdir()
    (model / "images.txt").write_text(
        "# images\n1 pose.jpg\n1 2 3\n2 pose.jpg\n4 5 6\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text(
        "# points\n1 0 0 0 1 2 3 0.1 1 0\n2 0 0 0 1 2 3 0.1 1 0\n",
        encoding="utf-8",
    )

    assert read_sparse_model_counts(model) == (2, 2)


def test_colmap_version_includes_cuda_build_line(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess(
            [],
            0,
            "COLMAP 4.0.0 -- Structure-from-Motion and Multi-View Stereo\n"
            "(Commit 8bac7b9 on 2026-03-15 with CUDA)\n",
            "",
        ),
    )

    assert colmap_version("colmap") == (
        "COLMAP 4.0.0 -- Structure-from-Motion and Multi-View Stereo "
        "(Commit 8bac7b9 on 2026-03-15 with CUDA)"
    )


@pytest.mark.parametrize(
    ("help_output", "expected"),
    [
        ("  --Mapper.image_list_path arg\n", "--Mapper.image_list_path"),
        ("  --image_list_path arg\n", "--image_list_path"),
    ],
)
def test_mapper_image_list_option_supports_both_colmap_clis(
    monkeypatch, help_output, expected
):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess([], 0, help_output, ""),
    )

    assert run_colmap_sparse.mapper_image_list_option("colmap") == expected


def test_run_command_preserves_colmap_stderr(monkeypatch):
    command = ["colmap", "mapper"]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: (_ for _ in ()).throw(
            subprocess.CalledProcessError(
                1,
                command,
                output="mapper stdout",
                stderr="unrecognised option",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="unrecognised option"):
        run_colmap_sparse.run_command(command)


def test_gpu_indices_accept_multiple_visible_devices():
    assert run_colmap_sparse.parse_gpu_indices("0,1") == "0,1"
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        run_colmap_sparse.parse_gpu_indices("0,-1")


def test_runner_applies_thread_limit_and_writes_progress(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    progress_path = tmp_path / "progress.json"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"image")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 1)
            _write_binary_count(model / "points3D.bin", 1)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            path = output_dir / "geometry" / "points.ply"
            path.write_text("ply\nelement vertex 1\nend_header\n", encoding="utf-8")
        elif command[1] == "model_converter" and command[-1] == "TXT":
            path = output_dir / "colmap" / "sparse_txt"
            (path / "cameras.txt").write_text(
                "1 PINHOLE 64 64 50 50 32 32\n", encoding="utf-8"
            )
            (path / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 frame.jpg\n\n", encoding="utf-8"
            )
        return "ok"

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(
        run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0 with CUDA"
    )
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--no-use-gpu",
            "--gpu-index",
            "0",
            "--num-threads",
            "4",
            "--progress-file",
            str(progress_path),
        ],
    )

    run_colmap_sparse.main()

    feature, matcher, mapper = commands[:3]
    assert feature[feature.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert feature[feature.index("--FeatureExtraction.gpu_index") + 1] == "0"
    assert feature[feature.index("--FeatureExtraction.num_threads") + 1] == "4"
    assert matcher[matcher.index("--FeatureMatching.use_gpu") + 1] == "0"
    assert matcher[matcher.index("--FeatureMatching.gpu_index") + 1] == "0"
    assert matcher[matcher.index("--FeatureMatching.num_threads") + 1] == "4"
    assert mapper[mapper.index("--Mapper.num_threads") + 1] == "4"
    assert "--Mapper.ba_global_frames_ratio" not in mapper
    assert "--Mapper.image_list_path" not in mapper
    assert "--image_list_path" not in mapper
    assert json.loads(progress_path.read_text()) == {"stage": "colmap_mapping"}
    log = (output_dir / "logs" / "run.log").read_text()
    assert "colmap_executable=" in log
    assert "colmap_build=COLMAP 4.0.0 with CUDA\n" in log
    assert "use_gpu=False\n" in log
    assert "gpu_index=0\n" in log
    assert "num_threads=4\n" in log


def test_gaussian_runner_caps_undistorted_images_and_uses_all_visible_gpus(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"image")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 12)
            _write_binary_count(model / "points3D.bin", 100)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            path = output_dir / "geometry" / "points.ply"
            path.write_text("ply\nelement vertex 1\nend_header\n", encoding="utf-8")
        return "ok"

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_camera_payload",
        lambda _: {
            "cameras": [{"model": "PINHOLE"}],
            "images": [{"image_id": index} for index in range(12)],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--matcher",
            "exhaustive",
            "--gaussian-baseline",
            "--max-image-size",
            "3072",
        ],
    )

    run_colmap_sparse.main()

    feature, matcher = commands[:2]
    undistorter = next(command for command in commands if command[1] == "image_undistorter")
    assert "--FeatureExtraction.gpu_index" not in feature
    assert "--FeatureMatching.gpu_index" not in matcher
    assert undistorter[undistorter.index("--max_image_size") + 1] == "3072"
    log = (output_dir / "logs" / "run.log").read_text()
    assert "gpu_index=all_visible\n" in log
    assert "max_image_size=3072\n" in log


def test_gaussian_sequential_matcher_enables_vocab_tree_loop_detection(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    vocab_tree = tmp_path / "vocab_tree.bin"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"image")
    commands = []

    def fake_run(command):
        commands.append(command)
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 12)
            _write_binary_count(model / "points3D.bin", 100)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            path = output_dir / "geometry" / "points.ply"
            path.write_text("ply\nelement vertex 1\nend_header\n", encoding="utf-8")
        return "ok"

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_camera_payload",
        lambda _: {
            "cameras": [{"model": "PINHOLE"}],
            "images": [{"image_id": index} for index in range(12)],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--matcher",
            "sequential",
            "--gaussian-baseline",
            "--vocab-tree-path",
            str(vocab_tree),
        ],
    )

    run_colmap_sparse.main()

    matcher = commands[1]
    assert matcher[1] == "sequential_matcher"
    assert matcher[matcher.index("--SequentialMatching.loop_detection") + 1] == "1"
    assert (
        matcher[matcher.index("--SequentialMatching.vocab_tree_path") + 1]
        == str(vocab_tree)
    )
    log = (output_dir / "logs" / "run.log").read_text()
    assert f"vocab_tree={vocab_tree}\n" in log


def test_v2_runner_sets_dynamic_overlap_and_recovers_before_undistortion(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    vocab_tree = tmp_path / "vocab_tree.bin"
    video_source = tmp_path / "video.mp4"
    selection_path = tmp_path / "selection.json"
    image_dir.mkdir()
    for index in range(21):
        (image_dir / f"frame_{index}.jpg").write_bytes(b"image")
    video_source.write_bytes(b"video")
    selection_path.write_text(
        json.dumps(
            {
                "profile": "video_keyframes_standard_v2",
                "selected": [
                    {
                        "path": f"frames/frame_{index}.jpg",
                        "time_seconds": index / 5,
                        "pts": index,
                    }
                    for index in range(21)
                ],
            }
        ),
        encoding="utf-8",
    )
    commands = []
    events = []
    recovery_call = {}

    def fake_run(command):
        commands.append(command)
        events.append(command[1])
        if command[1] == "mapper":
            model = output_dir / "colmap" / "sparse" / "0"
            model.mkdir(parents=True)
            _write_binary_count(model / "images.bin", 12)
            _write_binary_count(model / "points3D.bin", 100)
        elif command[1] == "model_converter" and command[-1] == "PLY":
            (output_dir / "geometry" / "points.ply").write_text(
                "ply\nelement vertex 1\nend_header\n", encoding="utf-8"
            )
        return "ok"

    def fake_expansion(**kwargs):
        events.append("video_initial_registration_expansion")
        return (
            kwargs["initial_model"],
            {
                "status": "no_progress",
                "accepted_pass_count": 0,
            },
            ["expansion"],
        )

    def fake_recovery(**kwargs):
        recovery_call.update(kwargs)
        events.append("video_registration_recovery")
        return kwargs["initial_model"], {"status": "recovered", "rounds": [{}]}, ["recovery"]

    monkeypatch.setattr(
        run_colmap_sparse, "resolve_colmap_executable", lambda: tmp_path / "colmap"
    )
    monkeypatch.setattr(run_colmap_sparse, "colmap_version", lambda _: "COLMAP 4.0.0")
    monkeypatch.setattr(
        run_colmap_sparse,
        "mapper_image_list_option",
        lambda _: "--Mapper.image_list_path",
    )
    monkeypatch.setattr(run_colmap_sparse, "run_command", fake_run)
    monkeypatch.setattr(
        run_colmap_sparse,
        "expand_v2_initial_registration",
        fake_expansion,
    )
    monkeypatch.setattr(run_colmap_sparse, "recover_video_registration", fake_recovery)
    monkeypatch.setattr(
        run_colmap_sparse,
        "build_camera_payload",
        lambda _: {
            "cameras": [{"model": "PINHOLE"}],
            "images": [{"image_id": index} for index in range(12)],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--matcher",
            "sequential",
            "--gaussian-baseline",
            "--vocab-tree-path",
            str(vocab_tree),
            "--video-source",
            str(video_source),
            "--video-selection",
            str(selection_path),
        ],
    )

    run_colmap_sparse.main()

    matcher = commands[1]
    mapper = commands[2]
    assert matcher[matcher.index("--SequentialMatching.overlap") + 1] == "21"
    assert mapper[mapper.index("--Mapper.ba_global_frames_ratio") + 1] == "1.5"
    assert mapper[mapper.index("--Mapper.ba_global_points_ratio") + 1] == "1.5"
    assert mapper[mapper.index("--Mapper.ba_global_frames_freq") + 1] == "1000"
    assert mapper[mapper.index("--Mapper.ba_global_points_freq") + 1] == "1000000"
    assert mapper[mapper.index("--Mapper.ba_global_max_refinements") + 1] == "1"
    seed_path = Path(mapper[mapper.index("--Mapper.image_list_path") + 1])
    assert seed_path.read_text().splitlines() == [
        f"frame_{index}.jpg" for index in range(21)
    ]
    assert events.index("video_initial_registration_expansion") < events.index(
        "video_registration_recovery"
    )
    assert events.index("video_registration_recovery") < events.index("image_undistorter")
    assert recovery_call["database_path"] == output_dir / "colmap" / "database.db"
    assert recovery_call["selection_path"] == selection_path
    log = (output_dir / "logs" / "run.log").read_text()
    assert "sequential_overlap=21\n" in log
    assert "num_images=21\n" in log
    assert "initial_input_count=21\n" in log
    assert "registration_ratio=0.571429\n" in log
    assert "video_registration_recovery_status=recovered\n" in log
    timing = json.loads(
        (output_dir / "diagnostics" / "colmap_timing.json").read_text()
    )
    assert timing["video_profile"] == "video_keyframes_standard_v2"
    assert set(timing["stage_elapsed_seconds"]) == {
        "feature_extraction",
        "feature_matching",
        "mapping",
        "initial_registration_expansion",
        "registration_recovery",
        "undistortion",
        "point_cloud_conversion",
        "text_conversion",
    }
    assert timing["v2_mapper_options"] == run_colmap_sparse.v2_mapper_options(
        json.loads(selection_path.read_text())
    )
    assert timing["total_elapsed_seconds"] >= sum(
        timing["stage_elapsed_seconds"].values()
    )


def test_gaussian_baseline_sequential_requires_vocab_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--matcher",
            "sequential",
            "--gaussian-baseline",
        ],
    )

    with pytest.raises(SystemExit, match="vocab-tree-path"):
        run_colmap_sparse.main()


def test_runner_rejects_nonpositive_thread_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colmap_sparse.py",
            "--image-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--num-threads",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        run_colmap_sparse.main()
