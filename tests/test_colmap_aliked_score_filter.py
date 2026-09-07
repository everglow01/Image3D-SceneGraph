from pathlib import Path
import shutil
import subprocess

import pytest


PATCH = Path(__file__).parents[1] / "scripts/patches/colmap-4-aliked-positive-scores-v1.patch"


def test_colmap_patch_keeps_positive_scores_and_original_indices(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "src/colmap/feature/aliked.cc"
    source.parent.mkdir(parents=True)
    source.write_text(r'''
#include <algorithm>
#include <memory>
#include <vector>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <iostream>
#define THROW_CHECK(x) if (!(x)) throw std::runtime_error("check")
#define LOG(x) std::clog
struct Tensor {
  std::vector<float> values;
  template <typename T> const T* GetTensorData() const { return values.data(); }
};
struct ValidKeypoint { float x, y; int index; };
void extract(const std::vector<Tensor>& output_tensors,
             std::vector<ValidKeypoint>* keypoints) {
    const int num_keypoints = output_tensors[2].values.size();
    const float* keypoints_data = output_tensors[0].GetTensorData<float>();
    const float* descriptors_data = output_tensors[1].GetTensorData<float>();

    (void)descriptors_data;
    const float scale_x = 4.5f, scale_y = 4.5f;
    const int width = 10, height = 10;
    std::vector<ValidKeypoint> valid_keypoints;
    valid_keypoints.reserve(num_keypoints);
    for (int i = 0; i < num_keypoints; ++i) {
      const float norm_x = keypoints_data[2 * i + 0];
      const float norm_y = keypoints_data[2 * i + 1];
      const float px = (norm_x + 1.0f) * scale_x + 0.5f;
      const float py = (norm_y + 1.0f) * scale_y + 0.5f;
      if (px >= 0.0f && px < width && py >= 0.0f && py < height) {
        valid_keypoints.push_back({px, py, i});
      }
    }
    const int num_valid = static_cast<int>(valid_keypoints.size());
    keypoints->resize(num_valid);
    *keypoints = valid_keypoints;
}
int main() {
  std::vector<Tensor> tensors = {
    {{-1,-1, 0,0, 0,0, 3,3, 0,0}}, {{10,11,12,13,14}}, {{0,0.1f,0.2f,0.5f,1}}
  };
  std::vector<ValidKeypoint> points;
  extract(tensors, &points);
  assert(points.size() == 3);
  assert(points[0].index == 1 && points[1].index == 2 && points[2].index == 4);
  assert(points[0].x == 5 && points[0].y == 5);
  assert(tensors[1].values[points[2].index] == 14);
  tensors[2].values.assign(5, 0);
  extract(tensors, &points);
  assert(points.empty());
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
    tensors[2].values[0] = invalid;
    bool failed = false;
    try { extract(tensors, &points); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
  }
}
''')
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_experiment_detects_descriptor_index_corruption(tmp_path, monkeypatch):
    import runpy
    import sqlite3

    import numpy as np

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "run_aliked_score_filter_experiment.py"))
    for name, indices in (("old", [0, 1, 2]), ("new", [2, 1]), ("broken", [2, 1])):
        with sqlite3.connect(tmp_path / f"{name}.db") as db:
            db.executescript("""
                CREATE TABLE images(image_id INTEGER, name TEXT, camera_id INTEGER);
                CREATE TABLE cameras(camera_id INTEGER, width INTEGER, height INTEGER);
                CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
                CREATE TABLE descriptors(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
                CREATE TABLE matches(rows INTEGER);
                CREATE TABLE two_view_geometries(rows INTEGER);
                INSERT INTO images VALUES (1, 'frame.jpg', 1);
                INSERT INTO cameras VALUES (1, 100, 100);
            """)
            xy = np.array([[1, 2], [10, 20], [30, 40]], dtype="<f4")[indices]
            desc = np.eye(3, dtype="<f4")[indices]
            if name == "broken":
                desc = desc[::-1].copy()
            db.execute("INSERT INTO keypoints VALUES (1,?,2,?)", (len(indices), xy.tobytes()))
            db.execute("INSERT INTO descriptors VALUES (1,?,12,?)", (len(indices), desc.tobytes()))
    compare = audit["compare_databases"]
    assert compare(tmp_path / "old.db", tmp_path / "new.db", ["frame.jpg"])[0]["numerical_subset_passed"]
    assert not compare(tmp_path / "old.db", tmp_path / "broken.db", ["frame.jpg"])[0]["numerical_subset_passed"]
    assert audit["database_summary"](tmp_path / "new.db")["feature_counts"] == {"frame.jpg": 2}
    with sqlite3.connect(tmp_path / "new.db") as db:
        db.execute("UPDATE descriptors SET rows=1")
    with pytest.raises(ValueError, match="row mismatch"):
        audit["database_summary"](tmp_path / "new.db")


def test_comparison_maps_filtered_indices_and_aligns_camera_gauges(tmp_path, monkeypatch):
    import runpy

    import numpy as np

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "run_aliked_score_filter_experiment.py"))
    original = {
        "keypoints": np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32),
        "descriptors": np.array([np.eye(3)], dtype=np.float32),
    }
    positive = {key: value[:, [0, 2]] for key, value in original.items()}
    mapping, record = audit["map_positive_feature_indices"](original, positive)
    assert mapping.tolist() == [0, 2]
    assert record["removed_count"] == 1
    positive["descriptors"] = positive["descriptors"].copy()
    positive["descriptors"][0, 1] = 1
    with pytest.raises(ValueError, match="ordered subset"):
        audit["map_positive_feature_indices"](original, positive)

    original_dir, positive_dir = tmp_path / "original", tmp_path / "positive"
    original_dir.mkdir()
    positive_dir.mkdir()
    centers = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    for directory, values in ((original_dir, centers), (positive_dir, centers * 2 + 5)):
        lines = ["# images"]
        for index, center in enumerate(values, 1):
            translation = -center
            lines.extend((
                f"{index} 1 0 0 0 {translation[0]} {translation[1]} {translation[2]} 1 {index}.jpg",
                "",
            ))
        (directory / "images.txt").write_text("\n".join(lines) + "\n")
    alignment = audit["align_camera_centers"](original_dir, positive_dir)
    assert alignment["similarity_scale"] == pytest.approx(0.5)
    assert alignment["residual_to_original_median_radius"]["max"] < 1e-12


def test_bruteforce_database_copy_clears_matches_only(tmp_path, monkeypatch):
    import runpy
    import sqlite3

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "run_aliked_score_filter_experiment.py"))
    source, destination = tmp_path / "source.db", tmp_path / "destination.db"
    with sqlite3.connect(source) as db:
        db.executescript("""
            CREATE TABLE images(image_id INTEGER, name TEXT);
            CREATE TABLE matches(pair_id INTEGER);
            CREATE TABLE two_view_geometries(pair_id INTEGER);
            INSERT INTO images VALUES (1, 'preserved.jpg');
            INSERT INTO matches VALUES (12);
            INSERT INTO two_view_geometries VALUES (12);
        """)
    audit["copy_feature_database"](source, destination)
    with sqlite3.connect(source) as db:
        assert db.execute("SELECT count(*) FROM matches").fetchone()[0] == 1
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT name FROM images").fetchone()[0] == "preserved.jpg"
        assert db.execute("SELECT count(*) FROM matches").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM two_view_geometries").fetchone()[0] == 0
    with pytest.raises(ValueError, match="refusing to overwrite"):
        audit["copy_feature_database"](source, destination)


def test_gpu_telemetry_is_fail_soft(monkeypatch):
    import runpy

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    audit = runpy.run_path(str(scripts / "run_aliked_score_filter_experiment.py"))

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 2)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert audit["sample_gpu_memory_mib"]() is None


def test_candidate_runtime_copy_preserves_soname_links(tmp_path, monkeypatch):
    import runpy

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    build = runpy.run_path(str(scripts / "build_colmap_aliked_score_filter.py"))
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "libonnxruntime.so.1.2.3").write_bytes(b"runtime")
    (source / "libonnxruntime.so.1").symlink_to("libonnxruntime.so.1.2.3")
    (source / "libonnxruntime.so").symlink_to("libonnxruntime.so.1")
    records = build["copy_runtime_libraries"](source, destination)
    assert records["libonnxruntime.so"] == {"symlink": "libonnxruntime.so.1"}
    assert (destination / "libonnxruntime.so").resolve().read_bytes() == b"runtime"
    (destination / "libonnxruntime.so.1.2.3").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="runtime library mismatch"):
        build["copy_runtime_libraries"](source, destination)
