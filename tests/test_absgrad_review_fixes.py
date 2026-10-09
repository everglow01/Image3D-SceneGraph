"""Regression checks using file/process mocks only; never import a model runtime."""
from __future__ import annotations

import __future__
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pytest

from image3d_scenegraph.gaussian import absgrad_experiment as experiment
from image3d_scenegraph.gaussian import absgrad_resources as resources
from scripts.evaluate_absgrad_pair import resource_report

ROOT = Path(__file__).resolve().parents[1]


def extracted(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def test_nested_validation_preserves_peak_and_independent_evaluation_can_reset():
    class BeforeRendering(Exception):
        pass
    class Stop:
        def __enter__(self):
            raise BeforeRendering()
        def __exit__(self, *args):
            pass
    cuda = SimpleNamespace(peak=19)
    cuda.reset_peak_memory_stats = lambda _: setattr(cuda, 'peak', 2)
    evaluate = extracted('src/image3d_scenegraph/gaussian/evaluation.py', 'evaluate_model', {
        'torch': SimpleNamespace(cuda=cuda, zeros=lambda *a, **k: None, bool=bool, no_grad=Stop),
        'render_gaussians': lambda *a, **k: None,
    })
    model = SimpleNamespace(count=1, means=SimpleNamespace(is_cuda=True, device=0))
    with pytest.raises(BeforeRendering):
        evaluate(model, [1], split='validation', sh_degree=3, reset_memory_peak=False)
    assert cuda.peak == 19
    with pytest.raises(BeforeRendering):
        evaluate(model, [1], split='validation', sh_degree=3)
    assert cuda.peak == 2
    trainer = ast.parse((ROOT / 'src/image3d_scenegraph/gaussian/trainer.py').read_text())
    wrapper = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_views')
    call = next(n for n in ast.walk(wrapper) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'evaluate_model')
    assert any(k.arg == 'reset_memory_peak' and k.value.value is False for k in call.keywords)


def test_atomic_model_writer_no_overwrite_and_failed_replace_keeps_old_file(tmp_path):
    def serialize(value, stream):
        stream.write(value['bytes'])
        if value.get('fail'):
            raise OSError('injected write failure')
    write = extracted('src/image3d_scenegraph/gaussian/model_io.py', 'save_torch_file', {
        'torch': SimpleNamespace(save=serialize), 'os': os, 'uuid': uuid,
    })
    path = tmp_path / 'model.pt'
    write({'bytes': b'old'}, path)
    with pytest.raises(FileExistsError):
        write({'bytes': b'bad'}, path)
    with pytest.raises(OSError, match='injected'):
        write({'bytes': b'partial', 'fail': True}, path, replace=True)
    assert path.read_bytes() == b'old'
    assert not list(tmp_path.glob('.*.tmp-*'))
    write({'bytes': b'new'}, path, replace=True)
    assert path.read_bytes() == b'new'
    link = tmp_path / 'link'
    link.symlink_to(path)
    with pytest.raises(FileExistsError):
        write({'bytes': b'bad'}, link, replace=True)
    assert path.read_bytes() == b'new'


def test_operational_model_io_avoids_full_bytes_and_provenance_covers_moved_code():
    tree = ast.parse((ROOT / 'src/image3d_scenegraph/gaussian/trainer.py').read_text())
    for name in ('train_gaussians', 'final_fit_gaussians', '_load_contiguous_model_shard', '_merge_model_shards', '_write_streaming_checkpoint'):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_model_bytes' for n in ast.walk(fn))
    provenance = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training_provenance')
    names = {n.value for n in ast.walk(provenance) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {'model_io.py', 'evaluation.py', 'checkpoint.py', 'absgrad_resources.py'} <= names


def test_admission_and_post_exit_verification_have_structured_failures(tmp_path, monkeypatch):
    def denied():
        raise ValueError('admission denied')
    with monkeypatch.context() as m:
        m.setattr(resources.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not spawn'))
        with pytest.raises(ValueError, match='admission denied'):
            resources.run_stage(tmp_path, 'admission', ['unused'], cwd=tmp_path, admission=denied)
    first = json.loads((tmp_path / 'admission.exit.json').read_text())
    assert first['returncode'] is None and first['phase'] == 'admission'
    def invalid():
        raise ValueError('invalid result identity')
    with pytest.raises(ValueError, match='invalid result identity'):
        resources.run_stage(tmp_path, 'verify', [sys.executable, '-I', '-c', 'pass'], cwd=tmp_path, verify=invalid)
    last = json.loads((tmp_path / 'verify.exit.json').read_text())
    assert last['returncode'] == 0 and last['phase'] == 'verification'
    assert 'invalid result identity' in last['resource_failure']
    assert (tmp_path / 'verify.failure.json').exists()


def test_memory_composition_and_process_data_are_retained(tmp_path, monkeypatch):
    (tmp_path / 'memory.events').write_text('oom 0\noom_kill 0\n')
    (tmp_path / 'memory.current').write_text('4096')
    (tmp_path / 'memory.stat').write_text('anon 1024\nfile 3072\n')
    (tmp_path / 'cgroup.procs').write_text(str(os.getpid()))
    monkeypatch.setattr(resources, 'available_host_bytes', lambda: 24 * 1024**3)
    snapshot = resources.host_snapshot(tmp_path)
    assert snapshot['memory_stat'] == {'anon': 1024, 'file': 3072}
    assert snapshot['processes'][0]['VmRSS_bytes'] > 0
    resources.run_stage(tmp_path, 'sample', [sys.executable, '-I', '-c', 'pass'], cwd=tmp_path, host_group=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / 'sample.memory.jsonl').read_text().splitlines()]
    assert [row['phase'] for row in rows] == ['admission', 'execution']
    assert all(row['memory_stat']['file'] == 3072 for row in rows)


def test_training_phase_is_explicit_opt_in_and_survives_telemetry_context(tmp_path):
    resources.record_training_phase('outside')
    with resources.memory_telemetry(tmp_path, 0, 0, sample=lambda: 1):
        resources.record_training_phase('validation', 3000)
    phase = json.loads((tmp_path / 'rank-0.phase.json').read_text())
    assert phase['phase'] == 'validation' and phase['iteration'] == 3000
    assert phase['pid'] == os.getpid()
    events = [json.loads(line) for line in (tmp_path / "rank-0.phase.jsonl").read_text().splitlines()]
    assert [event["phase"] for event in events] == ["initialization", "validation"]
    resources.record_training_phase('outside_again')
    assert json.loads((tmp_path / 'rank-0.phase.json').read_text()) == phase


def test_resource_report_separates_matched_and_historical_denominators():
    native = {'per_rank_peak_reserved_bytes': [6, 8], 'elapsed_seconds': 30}
    signed_native = {'per_rank_peak_reserved_bytes': [3, 4], 'elapsed_seconds': 20}
    candidate_stage = {'elapsed_seconds': 40, 'resources': {'max_observed_global_gaussians': 300,
        'per_rank_peak_reserved_bytes': [10, 12]}}
    signed_stage = {'elapsed_seconds': 25, 'resources': {'max_observed_global_gaussians': 200,
        'per_rank_peak_reserved_bytes': [5, 6]}}
    historical = {'max_observed_global_gaussians': 200, 'max_per_rank_peak_reserved_bytes': [4, 4],
        'max_main_stage_wall_seconds': 40}
    report = resource_report(native, signed_native, candidate_stage, signed_stage, historical)
    assert report['historical_reference']['gaussian_ratio'] == 3
    assert report['matched_pair']['gaussian_ratio'] == 1.5
    assert report['matched_pair']['native_reserved_ratio'] == [2, 2]
    assert report['matched_pair']['telemetry_reserved_ratio'] == [2, 2]
    assert resource_report(native, signed_native, candidate_stage, None, historical)['matched_pair'] is None


def test_pipeline_admission_failure_does_not_create_success_marker(tmp_path):
    def denied(*args, **kwargs):
        raise ValueError('busy')
    with pytest.raises(ValueError, match='busy'):
        experiment.run_pipeline(tmp_path, tmp_path / 'replay', lease_fd=0, require_resources=denied)
    assert json.loads((tmp_path / 'absolute/train.exit.json').read_text())['phase'] == 'admission'
    assert not (tmp_path / 'complete.json').exists()


def test_shared_experiment_is_not_a_cli_import_and_launch_shell_is_versionable():
    for name in ('run_absgrad_candidate.py', 'run_absgrad_matched_pair.py', 'evaluate_absgrad_pair.py'):
        source = (ROOT / 'scripts' / name).read_text()
        assert 'import_module(' not in source
        assert 'from run_video_4k_comparison' not in source
    shell = (ROOT / 'scripts/launch_absgrad_matched_pair.sh').read_text()
    assert 'MemoryMax=12G' in shell and 'MemorySwapMax=0' in shell
    assert '--execute-authorized' in shell and '--watch' in shell


@pytest.mark.parametrize('fail_absolute', [False, True])
def test_matched_driver_runs_both_arms_and_records_failure_without_models(tmp_path, monkeypatch, fail_absolute):
    import types
    from image3d_scenegraph.gaussian.config import resolve_internal_config, resolved_config_record

    (tmp_path / 'outputs/experiments').mkdir(parents=True)
    overlay = tmp_path / 'rendering.py'
    overlay.write_text('mock rendering identity; never imported')
    source = {'source': str(tmp_path / 'source'), 'source_protocol_sha256': 'source-sha',
        'replay': str(tmp_path / 'replay'), 'dataset_hash': 'dataset',
        'overlay_sha256': experiment.sha256_file(overlay)}
    baseline = resolved_config_record(resolve_internal_config('absgrad_ablation_v1', {
        'resolution': {'longest_edge': 1920}, 'opacity_reset': {'recovery_prune': {'enabled': True}}}))
    gate = experiment.matched_gate_template()
    gate.update(status='APPROVED_FOR_CANDIDATE_EXECUTION', absolute_training_authorized=True)
    evidence = list(experiment.SAVE_EVIDENCE.values())
    records = {'gate-sha': gate, experiment.SIGNED_PROTOCOL_SHA256: source,
        experiment.SIGNED_CONFIG_SHA256: baseline,
        experiment.SIGNED_FINAL_RECORD_SHA256: {'environment_hash': 'env'},
        experiment.QUALITY_SHA256: {}, experiment.ROI_SHA256: {}, 'source-sha': {'files': {}},
        evidence[0]: {'cases': [{'returncode': 0}]}, evidence[1]: {'status': 'passed'},
        evidence[2]: {'provenance': {'code_hash': 'core', 'environment_hash': 'env'}}}
    monkeypatch.setattr(experiment, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'quality_cgroup', lambda **kw: tmp_path / 'cgroup')
    monkeypatch.setattr(experiment, 'available_host_bytes', lambda: 24 * 1024**3)
    monkeypatch.setattr(experiment, 'revision', lambda: 'revision')
    monkeypatch.setattr(experiment, 'checked_json', lambda path, digest: records[digest])
    monkeypatch.setattr(experiment, 'require_resources', lambda *args, **kw: None)
    monkeypatch.setattr(experiment.subprocess, 'check_output', lambda *a, **kw: 'GPU 0: NVIDIA L2\nGPU 1: NVIDIA L2\n')
    stubs = {
        'gsplat': {'rendering': SimpleNamespace(__file__=str(overlay))},
        'image3d_scenegraph.gaussian.trainer': {'training_provenance': lambda **kw: SimpleNamespace(code_hash='core', environment_hash='env')},
        'image3d_scenegraph.gaussian.render': {'require_distributed_absgrad': lambda: None},
        'image3d_scenegraph.gaussian.replay': {'validate_replay_bundle': lambda _: None},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    calls = []
    def pipeline(root, replay, **kwargs):
        arm = kwargs['arm']
        calls.append(arm)
        kwargs['require_resources'](root, minimum_free_gib=8)
        if arm == 'absolute' and fail_absolute:
            raise ValueError('simulated absolute verification failure')
        for rel in (f'{arm}/train-only/record.json', f'{arm}/train-only/model.pt',
                    f'{arm}/train-only/evaluation.json', f'{arm}/selection/evaluation.json', 'complete.json'):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'code_hash': 'core', 'environment_hash': 'env'}))
    def paired(root, name, command, **kwargs):
        assert name == 'paired-quality'
        kwargs['admission']()
        target = Path(command[command.index('--output-dir') + 1])
        target.mkdir()
        (target / 'report.json').write_text('{}')
        calls.append('paired')
    monkeypatch.setattr(experiment, 'run_pipeline', pipeline)
    monkeypatch.setattr(experiment, 'run_stage', paired)
    output = tmp_path / 'outputs/experiments/new'
    arguments = (output, tmp_path / 'old-signed', tmp_path / 'gate', 'gate-sha')
    if fail_absolute:
        with pytest.raises(ValueError, match='absolute verification'):
            experiment.execute_matched(*arguments, expected_revision='revision')
        assert not (output / 'complete.json').exists()
        assert 'absolute verification' in json.loads((output / 'failure.json').read_text())['reason']
        assert calls == ['signed', 'absolute']
    else:
        experiment.execute_matched(*arguments, expected_revision='revision')
        assert calls == ['signed', 'absolute', 'paired']
        assert json.loads((output / 'complete.json').read_text())['promotion_eligible'] is False
        protocol = json.loads((output / 'absolute-candidate/protocol.json').read_text())
        assert protocol['matched_signed'] == experiment.receipt(output / 'signed-control')


def test_model_path_loader_uses_mmap_without_reading_full_bytes():
    import io
    calls = []
    state = {name: object() for name in ('means', 'log_scales', 'quats', 'opacity_logits', 'sh_coeffs')}
    def load(source, **kwargs):
        calls.append((source, kwargs))
        return {'state_dict': state, 'max_sh_degree': 3}
    class Model:
        def __init__(self, **kwargs):
            self.state = kwargs
        def to(self, device):
            return self
        def validate(self):
            pass
    read = extracted('src/image3d_scenegraph/gaussian/model_io.py', 'load_model_snapshot', {
        'Path': Path, 'io': io, 'torch': SimpleNamespace(load=load), 'GaussianModel': Model,
    })
    path = Path('/nonexistent/mock-only.pt')
    result = read(path, SimpleNamespace(type='cpu'))
    assert calls[0][0] == path and calls[0][1]['mmap'] is True
    assert calls[0][1]['weights_only'] is True
    assert result.state['means'] is state['means']
    read(b'legacy mocked bytes', SimpleNamespace(type='cpu'))
    assert isinstance(calls[1][0], io.BytesIO)
    assert 'mmap' not in calls[1][1]
