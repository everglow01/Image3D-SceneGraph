from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from image3d_scenegraph.gaussian.checkpoint import (
    AttemptRecord,
    CheckpointContractError,
    CheckpointProvenance,
    CheckpointRecord,
    CheckpointState,
    create_attempt,
    load_attempt,
    load_checkpoint,
    prune_attempt_checkpoints,
    write_checkpoint,
)


def provenance(seed: str = "a") -> CheckpointProvenance:
    values = [char * 64 for char in (seed, "b", "c", "d")]
    return CheckpointProvenance(*values)


def state(history: tuple[dict, ...] = ({"iteration": 10, "loss": 0.5},)) -> CheckpointState:
    return CheckpointState(
        model=b"model-state",
        optimizer=b"optimizer-state",
        scheduler=b"scheduler-state",
        densification=b"densification-state",
        rng=b"rng-state",
        metric_history=history,
    )


def create_fresh(tmp_path, *, value: CheckpointProvenance | None = None) -> AttemptRecord:
    return create_attempt(
        tmp_path,
        attempt_id="attempt-001",
        kind="fresh",
        provenance=value or provenance(),
    )


def test_atomic_checkpoint_round_trip_and_no_overwrite(tmp_path):
    expected_provenance = provenance()
    create_fresh(tmp_path, value=expected_provenance)

    record = write_checkpoint(
        tmp_path,
        attempt_id="attempt-001",
        iteration=10,
        purpose="periodic",
        provenance=expected_provenance,
        state=state(),
    )
    loaded = load_checkpoint(
        tmp_path,
        "attempt-001",
        10,
        expected_provenance=expected_provenance,
    )

    assert loaded.record == record
    assert loaded.state == state()
    assert len(record.checkpoint_hash) == 64
    checkpoint_path = tmp_path / "attempts" / "attempt-001" / "checkpoints" / "iteration_000000010"
    assert sorted(path.name for path in checkpoint_path.iterdir()) == [
        "checkpoint.json",
        "densification.bin",
        "metrics.json",
        "model.bin",
        "optimizer.bin",
        "rng.bin",
        "scheduler.bin",
    ]
    assert not any(path.name.startswith(".") for path in checkpoint_path.parent.iterdir())

    with pytest.raises(CheckpointContractError, match="already exists"):
        write_checkpoint(
            tmp_path,
            attempt_id="attempt-001",
            iteration=10,
            purpose="periodic",
            provenance=expected_provenance,
            state=state(),
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("model.bin", "file size mismatch: model"),
        ("optimizer.bin", "file size mismatch: optimizer"),
        ("metrics.json", "file size mismatch: metric_history"),
        ("checkpoint.json", "cannot read checkpoint metadata"),
    ],
)
def test_checkpoint_rejects_corrupted_or_partial_state(tmp_path, target, message):
    expected_provenance = provenance()
    create_fresh(tmp_path, value=expected_provenance)
    write_checkpoint(
        tmp_path,
        attempt_id="attempt-001",
        iteration=10,
        purpose="periodic",
        provenance=expected_provenance,
        state=state(),
    )
    target_path = (
        tmp_path
        / "attempts"
        / "attempt-001"
        / "checkpoints"
        / "iteration_000000010"
        / target
    )
    target_path.write_bytes(b"tampered")

    with pytest.raises(CheckpointContractError, match=message):
        load_checkpoint(tmp_path, "attempt-001", 10)


def test_checkpoint_rejects_same_size_hash_tampering(tmp_path):
    expected_provenance = provenance()
    create_fresh(tmp_path, value=expected_provenance)
    write_checkpoint(
        tmp_path,
        attempt_id="attempt-001",
        iteration=10,
        purpose="periodic",
        provenance=expected_provenance,
        state=state(),
    )
    model = (
        tmp_path
        / "attempts"
        / "attempt-001"
        / "checkpoints"
        / "iteration_000000010"
        / "model.bin"
    )
    model.write_bytes(b"tampered-st")

    with pytest.raises(CheckpointContractError, match="file hash mismatch: model"):
        load_checkpoint(tmp_path, "attempt-001", 10)


def test_temporary_and_missing_checkpoint_are_not_loadable(tmp_path):
    create_fresh(tmp_path)
    temporary = (
        tmp_path
        / "attempts"
        / "attempt-001"
        / "checkpoints"
        / ".iteration_000000010.tmp-interrupted"
    )
    temporary.mkdir(parents=True)
    (temporary / "model.bin").write_bytes(b"partial")

    with pytest.raises(CheckpointContractError, match="does not exist"):
        load_checkpoint(tmp_path, "attempt-001", 10)


def test_attempt_kinds_have_distinct_lineage_and_never_overwrite(tmp_path):
    expected_provenance = provenance()
    fresh = create_fresh(tmp_path, value=expected_provenance)
    write_checkpoint(
        tmp_path,
        attempt_id=fresh.attempt_id,
        iteration=10,
        purpose="periodic",
        provenance=expected_provenance,
        state=state(),
    )
    retry = create_attempt(
        tmp_path,
        attempt_id="attempt-002",
        kind="retry",
        provenance=expected_provenance,
        parent_attempt_id=fresh.attempt_id,
    )
    resumed = create_attempt(
        tmp_path,
        attempt_id="attempt-003",
        kind="resume",
        provenance=expected_provenance,
        parent_attempt_id=fresh.attempt_id,
        resume_iteration=10,
    )

    assert retry.parent_attempt_id == fresh.attempt_id
    assert retry.resume_checkpoint is None
    assert resumed.resume_checkpoint is not None
    assert resumed.resume_checkpoint.iteration == 10
    assert resumed.resume_checkpoint.checkpoint_hash == load_checkpoint(
        tmp_path, fresh.attempt_id, 10
    ).record.checkpoint_hash
    assert load_attempt(tmp_path, resumed.attempt_id) == resumed

    with pytest.raises(CheckpointContractError, match="attempt already exists"):
        create_attempt(
            tmp_path,
            attempt_id="attempt-003",
            kind="retry",
            provenance=expected_provenance,
            parent_attempt_id=fresh.attempt_id,
        )
    with pytest.raises(CheckpointContractError, match="fresh attempt must be the first"):
        create_attempt(tmp_path, attempt_id="attempt-004", kind="fresh", provenance=expected_provenance)
    with pytest.raises(CheckpointContractError, match="retry attempt cannot load"):
        create_attempt(
            tmp_path,
            attempt_id="attempt-004",
            kind="retry",
            provenance=expected_provenance,
            parent_attempt_id=fresh.attempt_id,
            resume_iteration=10,
        )


def test_attempt_descriptor_rejects_tampering(tmp_path):
    create_fresh(tmp_path)
    descriptor = tmp_path / "attempts" / "attempt-001" / "attempt.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["kind"] = "retry"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointContractError, match="descriptor hash mismatch"):
        load_attempt(tmp_path, "attempt-001")


@pytest.mark.parametrize(
    "field",
    ["dataset_hash", "effective_config_hash", "code_hash", "environment_hash"],
)
def test_resume_rejects_every_provenance_mismatch(tmp_path, field):
    expected_provenance = provenance()
    create_fresh(tmp_path, value=expected_provenance)
    write_checkpoint(
        tmp_path,
        attempt_id="attempt-001",
        iteration=10,
        purpose="periodic",
        provenance=expected_provenance,
        state=state(),
    )
    mismatch = replace(expected_provenance, **{field: "e" * 64})

    with pytest.raises(CheckpointContractError, match=field + " mismatch"):
        create_attempt(
            tmp_path,
            attempt_id="attempt-002",
            kind="resume",
            provenance=mismatch,
            parent_attempt_id="attempt-001",
            resume_iteration=10,
        )


def test_retry_requires_matching_dataset_and_config_but_not_environment(tmp_path):
    expected_provenance = provenance()
    create_fresh(tmp_path, value=expected_provenance)
    changed_environment = replace(expected_provenance, environment_hash="e" * 64)
    create_attempt(
        tmp_path,
        attempt_id="attempt-002",
        kind="retry",
        provenance=changed_environment,
        parent_attempt_id="attempt-001",
    )

    with pytest.raises(CheckpointContractError, match="dataset_hash mismatch"):
        create_attempt(
            tmp_path,
            attempt_id="attempt-003",
            kind="retry",
            provenance=replace(expected_provenance, dataset_hash="e" * 64),
            parent_attempt_id="attempt-001",
        )


@pytest.mark.parametrize("attempt_id", ["../escape", ".hidden", "with/slash", ""])
def test_attempt_identifier_rejects_path_escape(tmp_path, attempt_id):
    with pytest.raises(CheckpointContractError, match="invalid attempt_id"):
        create_attempt(tmp_path, attempt_id=attempt_id, kind="fresh", provenance=provenance())


def test_prune_attempt_checkpoints_keeps_only_requested_committed_rows(tmp_path):
    expected_provenance = provenance()
    create_fresh(tmp_path, value=expected_provenance)
    for iteration in (10, 20, 30):
        write_checkpoint(
            tmp_path,
            attempt_id="attempt-001",
            iteration=iteration,
            purpose="periodic",
            provenance=expected_provenance,
            state=state(),
        )
    root = tmp_path / "attempts" / "attempt-001" / "checkpoints"
    temporary = root / ".iteration_000000040.tmp-interrupted"
    temporary.mkdir()
    unrelated = root / "notes"
    unrelated.mkdir()

    prune_attempt_checkpoints(tmp_path, "attempt-001", keep_iterations=(30,))

    assert sorted(path.name for path in root.iterdir()) == [
        ".iteration_000000040.tmp-interrupted",
        "iteration_000000030",
        "notes",
    ]
    assert load_checkpoint(tmp_path, "attempt-001", 30).record.iteration == 30


def test_prune_attempt_checkpoints_rejects_unknown_attempt(tmp_path):
    with pytest.raises(CheckpointContractError, match="attempt does not exist"):
        prune_attempt_checkpoints(tmp_path, "attempt-001", keep_iterations=())


def _advance(value: float, velocity: float, rng: random.Random, start: int, end: int):
    history = []
    for iteration in range(start, end):
        gradient = value - 3.0 + rng.uniform(-0.01, 0.01)
        velocity = 0.9 * velocity + gradient
        value -= 0.1 * velocity
        history.append({"iteration": iteration + 1, "value": value})
    return value, velocity, history


def test_reference_resume_matches_uninterrupted_run(tmp_path):
    expected_provenance = provenance()
    uninterrupted_rng = random.Random(7)
    uninterrupted = _advance(0.0, 0.0, uninterrupted_rng, 0, 20)

    first_rng = random.Random(7)
    value, velocity, first_history = _advance(0.0, 0.0, first_rng, 0, 10)
    create_fresh(tmp_path, value=expected_provenance)
    write_checkpoint(
        tmp_path,
        attempt_id="attempt-001",
        iteration=10,
        purpose="periodic",
        provenance=expected_provenance,
        state=CheckpointState(
            model=json.dumps(value).encode(),
            optimizer=json.dumps(velocity).encode(),
            scheduler=b"schedule-v1",
            densification=b"stats-v1",
            rng=json.dumps(first_rng.getstate()).encode(),
            metric_history=tuple(first_history),
        ),
    )
    create_attempt(
        tmp_path,
        attempt_id="attempt-002",
        kind="resume",
        provenance=expected_provenance,
        parent_attempt_id="attempt-001",
        resume_iteration=10,
    )
    loaded = load_checkpoint(tmp_path, "attempt-001", 10, expected_provenance=expected_provenance)
    resumed_rng = random.Random()
    rng_state = json.loads(loaded.state.rng)
    resumed_rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
    resumed_tail = _advance(
        json.loads(loaded.state.model),
        json.loads(loaded.state.optimizer),
        resumed_rng,
        10,
        20,
    )

    assert resumed_tail[0] == uninterrupted[0]
    assert resumed_tail[1] == uninterrupted[1]
    assert list(loaded.state.metric_history) + resumed_tail[2] == uninterrupted[2]
