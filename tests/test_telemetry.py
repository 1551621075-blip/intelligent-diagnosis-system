"""主动告警按事件时间推进；改变采样周期、分批和重启不能漏掉告警。"""

import json

import pytest

from intelligent_diagnosis.schemas import Domain, DomainError, SignalSample, TelemetryEvent
from intelligent_diagnosis.service import DiagnosisService
from intelligent_diagnosis.storage import Store


def run_batches(root, samples, cuts=()):
    store = Store(root)
    actor = store.authenticate(store.provision_user("operator", "operator"))
    service = DiagnosisService(store, root / "models")
    responses = []
    start = 0
    for i, end in enumerate((*cuts, len(samples))):
        # 每批重建服务，持续段和冷却状态必须真正持久化。
        service = DiagnosisService(Store(root), root / "models")
        event = TelemetryEvent(
            event_id=f"batch-{i}",
            device_id="device",
            domain=Domain.LOW_VOLTAGE,
            samples=samples[start:end],
        )
        response = service.ingest(event, actor)
        assert service.ingest(event, actor) == response
        responses.append(response)
        start = end
    with store.connect() as db:
        requests = [
            json.loads(r[0])
            for r in db.execute("SELECT request_json FROM diagnoses ORDER BY rowid")
        ]
    return requests, responses


def points(times, warnings=None):
    warnings = [1] * len(times) if warnings is None else warnings
    return [
        SignalSample(timestamp_ms=t, values={"warning_active": w})
        for t, w in zip(times, warnings, strict=True)
    ]


@pytest.mark.parametrize(
    "interval,first_trigger", [(300, 2100), (600, 2400), (700, 2100), (1000, 2000)]
)
def test_non_divisor_sampling_triggers_without_future_samples(tmp_path, interval, first_trigger):
    samples = points(list(range(0, 6000, interval)))
    requests, responses = run_batches(tmp_path, samples, tuple(range(1, len(samples))))
    assert len(requests) == 1
    assert requests[0]["fault_time_ms"] == first_trigger - 1000
    assert max(s["timestamp_ms"] for s in requests[0]["samples"]) == first_trigger
    assert sum(r["trigger_count"] for r in responses) == 1


@pytest.mark.parametrize("mask", range(8))
def test_recovery_at_batch_end_does_not_hide_earlier_alarm(tmp_path, mask):
    samples = points([0, 1000, 2000, 3000], [1, 1, 1, 0])
    whole, _ = run_batches(tmp_path / "whole", samples)
    cuts = tuple(i for i in range(1, 4) if mask & (1 << (i - 1)))
    split, responses = run_batches(tmp_path / "split", samples, cuts)
    assert split == whole
    assert len(split) == 1
    assert sum(r["trigger_count"] for r in responses) == 1


def test_long_batch_returns_every_trigger_and_preserves_cooldown(tmp_path):
    samples = points(list(range(0, 126_000, 1000)))
    whole, responses = run_batches(tmp_path / "whole", samples)
    split, _ = run_batches(tmp_path / "split", samples, (1, 3, 60, 62, 63, 120, 123))
    assert whole == split
    assert [r["fault_time_ms"] for r in whole] == [1000, 61_000, 121_000]
    assert responses[0]["trigger_count"] == 3
    assert len(responses[0]["diagnoses"]) == 3
    assert responses[0]["diagnosis"] == responses[0]["diagnoses"][0]


@pytest.mark.parametrize("warnings", [[1, 0, 1], [1, None, 1], [1, 2, 1]])
def test_missing_invalid_or_cleared_warning_breaks_continuity(tmp_path, warnings):
    requests, _ = run_batches(tmp_path, points([0, 1000, 2000], warnings))
    assert not requests


def test_large_gap_cannot_be_interpreted_as_continuous_alarm(tmp_path):
    requests, _ = run_batches(tmp_path, points([0, 1000, 2100, 3000, 4000]), (2,))
    assert not requests


def test_overlapping_sample_conflict_rolls_back(tmp_path):
    store = Store(tmp_path)
    service = DiagnosisService(store, tmp_path / "models")
    actor = store.authenticate(store.provision_user("operator", "operator"))
    first = TelemetryEvent(
        event_id="a", device_id="d", domain="low_voltage", samples=points([0, 1000])
    )
    service.ingest(first, actor)
    conflicting = first.model_copy(
        update={"event_id": "b", "samples": points([1000, 2000], [0, 1])}
    )
    with pytest.raises(DomainError) as error:
        service.ingest(conflicting, actor)
    assert error.value.code == "sample_conflict"
    with store.connect() as db:
        assert db.execute("SELECT latest_ms FROM device_events").fetchone()[0] == 1000
        assert db.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0] == 1


def test_density_limit_is_not_bypassed_by_single_point_batches(tmp_path):
    store = Store(tmp_path)
    service = DiagnosisService(store, tmp_path / "models")
    actor = store.authenticate(store.provision_user("operator", "operator"))
    first = TelemetryEvent(
        event_id="a", device_id="d", domain="low_voltage", samples=points(list(range(1000)))
    )
    service.ingest(first, actor)
    second = first.model_copy(update={"event_id": "b", "samples": points([1000])})
    with pytest.raises(DomainError) as error:
        service.ingest(second, actor)
    assert error.value.code == "sample_density"


def test_maximum_sqlite_timestamp_is_safe(tmp_path):
    maximum = 2**63 - 1
    requests, responses = run_batches(tmp_path, points([maximum - 2000, maximum - 1000, maximum]))
    assert len(requests) == 1
    assert responses[0]["status"] == "triggered"
