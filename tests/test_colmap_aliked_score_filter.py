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
