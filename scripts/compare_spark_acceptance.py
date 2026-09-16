#!/usr/bin/env python3
"""Compare frozen Validation PNGs and hardware motion records without loading models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from audit_gaussian_render_consistency import image_difference


def run(root):
    torch.set_num_threads(4)
    hashes = {}

    def load_json(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def image(path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert('RGB'), copy=True)).float() / 255

    protocol = load_json(root / 'protocol.json')
    native = load_json(root / 'native/audit.json')
    reports = {arm: load_json(root / f'browser-user/{arm}/{filename}')
               for arm, filename in [('spark', 'spark.json'), ('legacy', 'browser.json')]}
    assert native['status'] == 'completed' and native['source_unchanged'] and native['split'] == 'validation'
    assert set(native['image_ids']) == set(protocol['image_ids'])
    assert not set(native['image_ids']) & set(protocol['excluded_ids'])
    for report in reports.values():
        assert report['status'] == 'completed' and report['source_unchanged'] and report['test_rgb'] == 'not_loaded'
        assert report['ply_sha256'] == protocol['ply_sha256'] == native['source_sha256']['ply']
        assert report['motion']['base_image_id'] == protocol['motion']['base_image_id']
    rows = []
    for image_id in protocol['image_ids']:
        views = {arm: next(c for c in report['captures'] if c['image_id'] == image_id and c['requested_sh'] == 3)
                 for arm, report in reports.items()}
        for arm, view in views.items():
            assert view['count'] == native['model_count'] and view['gl_error'] == 0
            assert 'NVIDIA' in view['gpu'] and 'L2' in view['gpu'] and 'SwiftShader' not in view['gpu']
            assert view['effective_sh'] == (3 if arm == 'spark' else 2)
        for key in ('projection', 'world_to_camera'):
            assert np.allclose(views['spark'][key], views['legacy'][key], atol=1e-7, rtol=0)
        reference = image(root / f'native/reference-{image_id}.png')
        original = image(root / f'native/native-sh3-{image_id}.png')
        row = {'image_id': image_id}
        for arm, view in views.items():
            png = image(root / f'browser-user/{arm}' / (view['name'] + '.png'))
            row[arm + '_native'] = image_difference(png, original)
            row[arm + '_reference'] = image_difference(png, reference)
        rows.append(row)
    means = {key: {metric: float(np.mean([row[key][metric] for row in rows]))
                   for metric in ('mae', 'psnr', 'ssim')}
             for key in ('spark_native', 'legacy_native', 'spark_reference', 'legacy_reference')}
    performance, temporal = {}, {}
    for arm, report in reports.items():
        trials = report['motion']['trials']
        perf = [t for t in trials if not t['capture_images']]
        assert len(perf) == 3 and all(len(t['raf_ms']) == 240 for t in perf)
        assert all('NVIDIA' in t['device'] and 'SwiftShader' not in t['device'] for t in trials)
        stats = []
        for t in perf:
            stats.append({'raf_p50_ms': float(np.median(t['raf_ms'])),
                          'raf_p95_ms': float(np.percentile(t['raf_ms'], 95)),
                          'raf_mean_ms': float(np.mean(t['raf_ms'])),
                          'cpu_p95_ms': float(np.percentile(t['cpu_submission_ms'], 95)),
                          'gpu_p95_ms': float(np.percentile([q['ms'] for q in t['gpu_queries']], 95))
                          if t['gpu_timer_available'] and not t['gpu_disjoint'] and t['gpu_queries_missing'] == 0 else None})
        performance[arm] = {'trials': stats, 'median_raf_p95_ms': float(np.median([s['raf_p95_ms'] for s in stats])),
                            'scene_load_wall_ms': report['captures'][0]['scene_load_wall_ms']}
        sampled = [t for t in trials if t['capture_images']]
        assert len(sampled) == 1
        differences = []
        for sample in sampled[0]['samples']:
            live = image(root / f'browser-user/{arm}' / sample['png'])
            settled = image(root / f'browser-user/{arm}' / sample['settled_png'])
            differences.append({'frame': sample['frame'], **image_difference(live, settled)})
        assert len(differences) == 13
        temporal[arm] = {'samples': differences, 'max_mae': max(d['mae'] for d in differences),
                         'mean_mae': float(np.mean([d['mae'] for d in differences]))}
    deltas = {'native_mae_reduction': 1 - means['spark_native']['mae'] / means['legacy_native']['mae'],
              'reference_psnr': means['spark_reference']['psnr'] - means['legacy_reference']['psnr'],
              'reference_ssim': means['spark_reference']['ssim'] - means['legacy_reference']['ssim'],
              'p95_frame_time_ratio': performance['spark']['median_raf_p95_ms'] / performance['legacy']['median_raf_p95_ms'],
              'load_time_ratio': performance['spark']['scene_load_wall_ms'] / performance['legacy']['scene_load_wall_ms']}
    return {'profile': 'spark_gpu_acceptance_comparison_v1', 'scope': protocol['scope'],
            'test_rgb': 'not_loaded', 'training': 'not_run', 'rows': rows, 'means': means,
            'performance': performance, 'temporal': temporal, 'deltas': deltas, 'input_sha256': hashes,
            'gates': {'native_parity': deltas['native_mae_reduction'] >= 0.10,
                      'reference_psnr': deltas['reference_psnr'] >= -0.1,
                      'reference_ssim': deltas['reference_ssim'] >= -0.002,
                      'frame_time': deltas['p95_frame_time_ratio'] <= 1.2,
                      'load_time': deltas['load_time_ratio'] <= 1.2,
                      'sampled_motion': temporal['spark']['max_mae'] <= 0.01 and
                      temporal['spark']['max_mae'] - temporal['legacy']['max_mae'] <= 0.002},
            'limitations': ['headless controlled renderer, not deployed interactive UI',
                           'motion thumbnails compare to same-renderer settled frames, not ground truth',
                           'one scene, one local path, no default promotion',
                           'cold loading measured once per arm; order/cache effects remain']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    with (args.root / 'comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'means': result['means'], 'deltas': result['deltas'], 'gates': result['gates']}))
