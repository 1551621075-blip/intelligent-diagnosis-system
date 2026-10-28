"""真实模型、知识门与SQLite服务贯穿测试；没有伪造成功的推理响应。"""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from intelligent_diagnosis.knowledge import KnowledgeBase
from intelligent_diagnosis.ml import train_models
from intelligent_diagnosis.schemas import (
    AssignmentClose,
    DispatchRequest,
    DomainError,
    Entry,
    FeedbackRequest,
    ModelActivation,
    ModelRollback,
    Outcome,
    TelemetryEvent,
)
from intelligent_diagnosis.service import DiagnosisService
from intelligent_diagnosis.storage import Store
from intelligent_diagnosis.synthetic import CLASS_DOMAIN, make_request


@pytest.fixture(scope="module")
def trained_service_models(tmp_path_factory):
    models = tmp_path_factory.mktemp("service-models")
    meta = train_models(models, n_devices=96)
    return models, meta


@pytest.fixture()
def setup(tmp_path, trained_service_models):
    models, meta = trained_service_models
    store = Store(tmp_path)
    actors = {
        name: store.authenticate(store.provision_user(name, role))
        for name, role in (("operator", "operator"), ("other", "operator"), ("expert", "expert"), ("admin", "admin"))
    }
    service = DiagnosisService(store, models)
    service.register_model(meta, actors["admin"])
    service.activate_model(ModelActivation(version=meta["version"], action_id="initial"), actors["admin"])
    return service, actors, meta


@pytest.mark.parametrize("label", tuple(CLASS_DOMAIN))
def test_real_diagnosis_and_frozen_result(setup, label):
    service, actors, _ = setup
    request = make_request(label, seed=912, device_id="new-field-synthetic-device")
    result = service.diagnose(request, actors["operator"])
    assert result.prediction is not None
    assert result.prediction.class_id == label
    assert result.knowledge_version == "knowledge-v1"
    assert result.stages == ["perception", "routing", "execution", "gating", "feedback_pending"]
    assert result.outcome == (Outcome.REVIEW if label in {"circuit_open", "circuit_short", "safety_sensor_invalid"} else Outcome.RESOLVED)
    assert result.evidence[0].status == "supported"
    assert service.diagnose(request, actors["operator"]) == result
    assert service.get(result.diagnosis_id, actors["operator"]) == result
    assert service.get(result.diagnosis_id, actors["expert"]) == result
    with pytest.raises(DomainError) as error:
        service.get(result.diagnosis_id, actors["other"])
    assert error.value.status == 404
    changed = request.model_copy(update={"complaint": request.complaint + " 更改内容"})
    with pytest.raises(DomainError, match="请求ID"):
        service.diagnose(changed, actors["operator"])


def test_missing_model_and_unknown_complaint_fail_closed(tmp_path):
    store = Store(tmp_path)
    actor = store.authenticate(store.provision_user("operator", "operator"))
    service = DiagnosisService(store, tmp_path / "empty-models")
    result = service.diagnose(make_request("battery_depleted"), actor)
    assert result.outcome == Outcome.UNKNOWN
    assert "无已批准模型" in result.reason
    result = service.diagnose(make_request("battery_depleted", request_id="unknown").model_copy(update={"complaint": "我想看电影和买咖啡", "domain": None}), actor)
    assert result.outcome == Outcome.UNKNOWN
    assert result.attribution is None


def test_missing_ood_and_conflict_are_not_resolved(setup):
    service, actors, _ = setup
    request = make_request("battery_depleted", request_id="missing")
    request.samples[0].values.pop("voltage")
    missing = service.diagnose(request, actors["operator"])
    assert missing.outcome == Outcome.UNKNOWN
    assert missing.prediction is None
    request = make_request("battery_depleted", request_id="extreme")
    for sample in request.samples:
        sample.values["voltage"] = 10000.0
    ood = service.diagnose(request, actors["operator"])
    assert ood.outcome == Outcome.UNKNOWN
    assert not ood.prediction.accepted
    request = make_request("battery_depleted", request_id="conflicting").model_copy(update={"complaint": "安全气囊报警灯持续亮起"})
    assert service.diagnose(request, actors["operator"]).outcome == Outcome.UNKNOWN


def test_concurrent_duplicate_creates_one_diagnosis(setup):
    service, actors, _ = setup
    request = make_request("battery_depleted", request_id="concurrent")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: service.diagnose(request, actors["operator"]), range(8)))
    assert len({r.diagnosis_id for r in results}) == 1
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0] == 1


def event_from_request(request, event_id="event"):
    return TelemetryEvent(event_id=event_id, device_id=request.device_id, domain=request.domain, samples=request.samples)


def test_telemetry_replay_cooldown_and_late_event(setup):
    service, actors, _ = setup
    request = make_request("battery_depleted", seed=912, entry=Entry.TELEMETRY)
    event = event_from_request(request)
    first = service.ingest(event, actors["operator"])
    assert first["status"] == "triggered"
    assert first["diagnosis"]["entry"] == "telemetry"
    assert first["diagnosis"]["prediction"]["class_id"] == "battery_depleted"
    assert first == service.ingest(event, actors["operator"])
    with pytest.raises(DomainError, match="事件ID"):
        service.ingest(event.model_copy(update={"latitude": 0.0}), actors["operator"])
    late = event.model_copy(update={"event_id": "late"})
    assert service.ingest(late, actors["operator"])["reason"] == "late_or_duplicate"
    shifted = request.model_copy(deep=True)
    for sample in shifted.samples:
        sample.timestamp_ms += 3000
    cooldown = service.ingest(event_from_request(shifted, "cooldown"), actors["operator"])
    assert cooldown["reason"] == "cooldown"


def test_telemetry_cross_batch_buffer_and_restart(setup):
    service, actors, _ = setup
    request = make_request("charging_failure", seed=912, entry=Entry.TELEMETRY)
    middle = len(request.samples) // 2
    first = event_from_request(request, "part1").model_copy(update={"samples": request.samples[:middle]})
    second = event_from_request(request, "part2").model_copy(update={"samples": request.samples[middle:]})
    assert service.ingest(first, actors["operator"])["reason"] == "not_sustained"
    restarted = DiagnosisService(Store(service.store.root), service.model_dir)
    result = restarted.ingest(second, actors["operator"])
    assert result["status"] == "triggered"
    assert result["diagnosis"]["prediction"]["class_id"] == "charging_failure"


def test_unsorted_telemetry_rejected_and_sparse_never_triggered(setup):
    service, actors, _ = setup
    request = make_request("battery_depleted", entry=Entry.TELEMETRY)
    event = event_from_request(request)
    with pytest.raises(DomainError) as error:
        service.ingest(event.model_copy(update={"samples": list(reversed(event.samples))}), actors["operator"])
    assert error.value.code == "event_order"
    event.samples = event.samples[:2]
    assert service.ingest(event, actors["operator"])["status"] == "ignored"


def test_feedback_immutable_idempotent_and_exported(setup):
    service, actors, _ = setup
    diagnosis = service.diagnose(make_request("battery_depleted", seed=912), actors["operator"])
    request = FeedbackRequest(feedback_id="feedback", correct_class="battery_depleted", actual_action="合成案例人工复核", repair_confirmed=False)
    with pytest.raises(DomainError) as error:
        service.feedback(diagnosis.diagnosis_id, request, actors["operator"])
    assert error.value.status == 403
    response = service.feedback(diagnosis.diagnosis_id, request, actors["expert"])
    assert response["badcase_type"] == "confirmed"
    assert response == service.feedback(diagnosis.diagnosis_id, request, actors["expert"])
    assert service.get(diagnosis.diagnosis_id, actors["operator"]) == diagnosis
    exported = service.export_feedback(actors["admin"])
    assert exported[0]["event_id"] == diagnosis.diagnosis_id
    assert exported[0]["confirmed_class"] == "battery_depleted"
    service.feedback(diagnosis.diagnosis_id, request.model_copy(update={"feedback_id": "conflict", "correct_class": "charging_failure"}), actors["expert"])
    with pytest.raises(DomainError) as error:
        service.export_feedback(actors["admin"])
    assert error.value.code == "label_conflict"


def test_assignment_atomic_cancel_restores_capacity_and_stock(setup):
    service, actors, _ = setup
    diagnosis = service.diagnose(make_request("battery_depleted", seed=912), actors["operator"])
    candidate = diagnosis.dispatch_candidates[0]
    with service.store.connect() as db:
        before = json.loads(db.execute("SELECT payload FROM services WHERE service_id=?", (candidate["service_id"],)).fetchone()[0])
    request = DispatchRequest(service_id=candidate["service_id"], action_id="assign")
    assigned = service.assign(diagnosis.diagnosis_id, request, actors["expert"])
    assert assigned["repair_completed"] is False
    assert service.assign(diagnosis.diagnosis_id, request, actors["expert"]) == assigned
    close = AssignmentClose(action_id="close", outcome="cancelled", note="合成订单取消")
    cancelled = service.close_assignment(diagnosis.diagnosis_id, close, actors["expert"])
    assert cancelled["status"] == "synthetic_cancelled"
    assert service.close_assignment(diagnosis.diagnosis_id, close, actors["expert"]) == cancelled
    assert service.assign(diagnosis.diagnosis_id, request, actors["expert"]) == assigned
    with pytest.raises(DomainError) as conflict:
        service.close_assignment(diagnosis.diagnosis_id, close.model_copy(update={"action_id": "assign"}), actors["expert"])
    assert conflict.value.code == "action_conflict"
    with service.store.connect() as db:
        after = json.loads(db.execute("SELECT payload FROM services WHERE service_id=?", (candidate["service_id"],)).fetchone()[0])
    assert before == after
    with pytest.raises(DomainError) as error:
        service.close_assignment(diagnosis.diagnosis_id, close.model_copy(update={"action_id": "double-close"}), actors["admin"])
    assert error.value.code == "assignment_terminal"


def test_model_report_cannot_be_forged_or_activated_by_operator(setup):
    service, actors, meta = setup
    forged = deepcopy(meta)
    forged["domains"]["safety"]["test_macro_f1"] = 0.1234
    with pytest.raises(DomainError) as error:
        service.register_model(forged, actors["admin"])
    assert error.value.code == "metadata_mismatch"
    with pytest.raises(DomainError) as error:
        service.activate_model(ModelActivation(version=meta["version"], action_id="forbidden"), actors["operator"])
    assert error.value.status == 403
    with pytest.raises(DomainError) as error:
        service.rollback_model(ModelRollback(version="sim-lgbm-" + "0" * 20, action_id="rollback", reason="合成故障注入"), actors["admin"])
    assert error.value.code == "rollback_unapproved"


def test_knowledge_independent_review_and_regression(setup):
    service, actors, _ = setup
    candidate = KnowledgeBase(version="knowledge-v2").snapshot()
    service.propose_knowledge(candidate, actors["admin"])
    with pytest.raises(DomainError) as error:
        service.publish_knowledge("knowledge-v2", actors["admin"])
    assert error.value.code == "independent_review"
    candidate = KnowledgeBase(version="knowledge-v3").snapshot()
    service.propose_knowledge(candidate, actors["expert"])
    assert service.publish_knowledge("knowledge-v3", actors["admin"])["status"] == "published"
    diagnosis = service.diagnose(make_request("battery_depleted", seed=912), actors["operator"])
    assert diagnosis.knowledge_version == "knowledge-v3"
