#!/usr/bin/env python3
"""Freeze unseen-by-renderer-audit Validation IDs before any candidate rendering."""
import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED = {'1108', '1718', '317', '425', '508', '1684'}


def select_views(contract):
    ids = sorted(set(contract['splits']['validation']) - EXCLUDED, key=int)
    if len(ids) < 12 or set(ids) & set(contract['splits']['test']):
        raise ValueError('requires 12 Validation-only candidates disjoint from Test')
    return [ids[i * (len(ids) - 1) // 11] for i in range(12)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    contract = json.loads(raw)
    if contract['dataset_hash'] != '9de071505b4f175b0f219c3ffbb4ecfd8e20a1832c68051ba5515b6183bb367a':
        raise ValueError('requires the frozen comparison dataset')
    ids = select_views(contract)
    result = {'profile': 'spark_gpu_acceptance_v1', 'image_ids': ids, 'split': 'validation',
              'excluded_ids': sorted(EXCLUDED, key=int),
              'selection': '12 equally spaced ranks of numerically sorted Validation IDs excluding original six; no quality scores',
              'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'dataset_hash': contract['dataset_hash'],
              'ply_sha256': '59cf39a46948ef447d35b299d77ffc14878a9e902a3e7460df0700a0b1df3a38',
              'model_sha256': '054bbb6ca1563de47e8201b9b713bc112dc29a1a24d34b4cab5658a1d7d31586',
              'test_rgb': 'not_loaded', 'training': 'not_run',
              'quality_thresholds': {'mean_native_mae_reduction_min': 0.10,
                                     'mean_reference_psnr_delta_min': -0.1,
                                     'mean_reference_ssim_delta_min': -0.002},
              'resource_thresholds': {'p95_frame_time_ratio_max': 1.2, 'load_time_ratio_max': 1.2},
              'motion': {'base_image_id': ids[0], 'frames': 240, 'warmup': 60, 'trials': 3,
                         'translation_normalized_units': 0.01, 'yaw_degrees': 3,
                         'sample_every': 20, 'thumbnail_width': 320,
                         'max_sample_live_settled_mae': 0.01,
                         'max_sample_live_settled_mae_delta': 0.002,
                         'scope': 'local closed loop around first selected camera, per-frame displacement; no ground-truth moving video'},
              'scope': 'independent of six renderer-debug views only; same scene/development Validation, not unseen-scene or Test'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
