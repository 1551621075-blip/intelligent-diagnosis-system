"""Real feedback, LightGBM release/rollback, and authenticated lifecycle routes."""

import json
import shutil
from dataclasses import dataclass
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from intelligent_diagnosis.api import create_app
from intelligent_diagnosis.ml import DiagnosisModels, train_models
from intelligent_diagnosis.schemas import Actor, Domain
from intelligent_diagnosis.service import DiagnosisService
from intelligent_diagnosis.synthetic import make_request

TRAIN_SEED = 17
TRAIN_DEVICES = 96


@dataclass(repr=False)
class LifecycleAPI:
    client: TestClient
    service: DiagnosisService
    headers: dict[str, dict[str, str]]
    admin: Actor
    original: dict

    def request(self, method, path, payload=None, role="admin"):
        return self.client.request(
            method, path, json=payload, headers=self.headers[role] if role else {}
        )


def ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def error(response, status, code):
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    if status == 401:
        assert response.headers["www-authenticate"] == "Bearer"


def database_state(service):
    # Deliberately exclude users: tests only use freshly provisioned local identities.
    tables = (
        "settings", "models", "model_approvals", "actions", "audit", "diagnoses",
        "feedback", "assignments", "services",
    )
    with service.store.connect() as db:
        return {
            table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in tables
        }


def active_version(service):
    with service.store.connect() as db:
        return db.execute("SELECT value FROM settings WHERE key='active_model'").fetchone()[0]


def artifact_snapshot(directory):
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.iterdir()
        if path.is_file()
    }


def diagnose(api, request_id="lifecycle-original"):
    payload = make_request(
        "battery_depleted", seed=912, device_id="new-field-synthetic-device",
        request_id=request_id,
    ).model_dump(mode="json")
    result = ok(api.request("POST", "/api/diagnoses", payload, role="operator"))
    assert result["prediction"]["class_id"] == "battery_depleted"
    assert result["prediction"]["accepted"] is True
    assert result["outcome"] == "resolved"
    assert result["evidence"][0]["status"] == "supported"
    assert result["attribution"]["required_part"] is not None
    assert result["dispatch_candidates"]
    return payload, result


def add_feedback(api, diagnosis_id, feedback_id="lifecycle-feedback", **changes):
    payload = {
        "feedback_id": feedback_id,
        "correct_class": "battery_depleted",
        "actual_action": "Expert reviewed the original synthetic measurements; no real repair.",
        "repair_confirmed": False,
        **changes,
    }
    return ok(api.request(
        "POST", f"/api/diagnoses/{diagnosis_id}/feedback", payload, role="expert"
    ))


def resource(service, service_id):
    with service.store.connect() as db:
        return json.loads(db.execute(
            "SELECT payload FROM services WHERE service_id=?", (service_id,)
        ).fetchone()[0])


@pytest.fixture(scope="module")
def lifecycle_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lifecycle-models")
    metadata = train_models(directory, seed=TRAIN_SEED, n_devices=TRAIN_DEVICES)
    return directory, metadata


@pytest.fixture
def lifecycle_api(tmp_path, lifecycle_models):
    directory, metadata = lifecycle_models
    models = tmp_path / "models"
    shutil.copytree(directory / metadata["version"], models / metadata["version"])
    app = create_app(tmp_path / "runtime", models)
    headers, actors = {}, {}
    for name, role in (
        ("operator", "operator"), ("expert", "expert"),
        ("admin", "admin"), ("reviewer", "admin"),
    ):
        # These credentials are generated for this isolated test database only.
        token = app.state.store.provision_user(name, role)
        headers[name] = {"Authorization": f"Bearer {token}"}
        actors[name] = app.state.store.authenticate(token)
    with TestClient(app) as client:
        api = LifecycleAPI(client, app.state.service, headers, actors["admin"], metadata)
        assert api.service.register_model(metadata, api.admin) == metadata
        assert ok(api.request("POST", "/api/models/activate", {
            "version": metadata["version"], "action_id": "initial-activation",
        })) == {"status": "activated", "version": metadata["version"], "previous": None}
        yield api


def test_export_train_register_activate_and_rollback_preserves_history(lifecycle_api):
    api = lifecycle_api
    first = api.original
    first_path = api.service.model_dir / first["version"]
    original_artifacts = artifact_snapshot(first_path)
    request, old_diagnosis = diagnose(api)
    diagnosis_id = old_diagnosis["diagnosis_id"]
    diagnosis_path = f"/api/diagnoses/{diagnosis_id}"
    old_artifacts = artifact_snapshot(api.service.store.artifacts)
    with api.service.store.connect() as db:
        original_row = tuple(db.execute(
            "SELECT * FROM diagnoses WHERE diagnosis_id=?", (diagnosis_id,)
        ).fetchone())

    feedback = add_feedback(api, diagnosis_id)
    assert feedback["badcase_type"] == "confirmed"
    assert feedback["repair_confirmed"] is False
    exported = ok(api.request("GET", "/api/feedback/export", role="expert"))
    assert exported == [{
        "event_id": diagnosis_id,
        "device_id": request["device_id"],
        "request": request,
        "confirmed_class": "battery_depleted",
        "feedback_id": feedback["feedback_id"],
    }]
    assert ok(api.request("GET", "/api/feedback/export")) == exported

    # Feed the real API export straight into training, without editing any reports.
    second = train_models(
        api.service.model_dir, seed=TRAIN_SEED, n_devices=TRAIN_DEVICES,
        feedback_records=exported,
    )
    assert first["feedback"]["accepted_count"] == 0
    assert second["feedback"]["received_records"] == 1
    assert second["feedback"]["accepted_count"] == 1
    assert second["feedback"]["rejected_count"] == second["feedback"]["duplicate_count"] == 0
    assert second["feedback"]["partition"] == "train_only"
    assert second["feedback"]["accepted"][0]["event_id"] == diagnosis_id
    assert second["feedback"]["accepted"][0]["feedback_id"] == feedback["feedback_id"]
    assert second["config"] == first["config"]
    assert second["version"] != first["version"]
    assert len(first["evaluation_fingerprint"]) == 64
    assert second["evaluation_fingerprint"] == first["evaluation_fingerprint"]
    for partition in ("validation", "test"):
        assert second["splits"][partition] == first["splits"][partition]
        assert request["device_id"] not in second["splits"][partition]["device_ids"]
    assert second["splits"]["train"]["n_events"] == first["splits"]["train"]["n_events"] + 1
    assert second["splits"]["train"]["n_devices"] == first["splits"]["train"]["n_devices"] + 1
    assert set(second["splits"]["train"]["device_ids"]) == (
        set(first["splits"]["train"]["device_ids"]) | {request["device_id"]}
    )
    for domain in Domain:
        previous, current = first["domains"][domain.value], second["domains"][domain.value]
        added = int(domain == Domain.LOW_VOLTAGE)
        assert current["feedback_train_events"] == added
        assert current["train_events"] == previous["train_events"] + added
        assert current["test_macro_f1"] == second["metrics"][domain.value]["test"]["macro_f1"]
        assert current["test_macro_f1"] >= previous["test_macro_f1"] >= 0.80
        before = (first_path / previous["model_file"]).read_bytes()
        after = (api.service.model_dir / second["version"] / current["model_file"]).read_bytes()
        assert sha256(before).hexdigest() == previous["model_sha256"]
        assert sha256(after).hexdigest() == current["model_sha256"]
        assert (before != after) == bool(added)
    assert DiagnosisModels(api.service.model_dir, second["version"]).metadata == second

    activation = {"version": second["version"], "action_id": "feedback-activation"}
    before_registration = database_state(api.service)
    error(api.request("POST", "/api/models/activate", activation), 404, "model_unregistered")
    assert database_state(api.service) == before_registration
    assert active_version(api.service) == first["version"]
    assert api.service.register_model(second, api.admin) == second
    assert active_version(api.service) == first["version"]
    activated = ok(api.request("POST", "/api/models/activate", activation))
    assert activated == {
        "status": "activated", "version": second["version"], "previous": first["version"],
    }
    after_activation = database_state(api.service)
    assert ok(api.request("POST", "/api/models/activate", activation)) == activated
    assert database_state(api.service) == after_activation
    assert active_version(api.service) == second["version"]
    _, new_diagnosis = diagnose(api, "after-activation")
    assert new_diagnosis["model_version"] == second["version"]
    assert new_diagnosis["prediction"]["model_version"] == second["version"]
    assert ok(api.request("GET", diagnosis_path, role="operator")) == old_diagnosis
    assert ok(api.request("POST", "/api/diagnoses", request, role="operator")) == old_diagnosis

    rollback = {
        "version": first["version"], "action_id": "feedback-rollback",
        "reason": "Synthetic release exercise: restore the first approved model.",
    }
    rolled_back = ok(api.request("POST", "/api/models/rollback", rollback))
    assert rolled_back == {
        "status": "rolled_back", "version": first["version"],
        "previous": second["version"], "reason": rollback["reason"],
    }
    after_rollback = database_state(api.service)
    assert ok(api.request("POST", "/api/models/rollback", rollback)) == rolled_back
    assert database_state(api.service) == after_rollback
    assert active_version(api.service) == first["version"]
    for changed, role, code in (
        ({**rollback, "reason": "A different reason"}, "admin", "action_conflict"),
        (rollback, "reviewer", "action_conflict"),
        ({**rollback, "action_id": "another-rollback"}, "admin", "rollback_no_change"),
    ):
        error(api.request("POST", "/api/models/rollback", changed, role), 409, code)
        assert database_state(api.service) == after_rollback
    _, restored_diagnosis = diagnose(api, "after-rollback")
    assert restored_diagnosis["model_version"] == first["version"]
    assert restored_diagnosis["prediction"] == old_diagnosis["prediction"]
    assert ok(api.request("GET", diagnosis_path, role="operator")) == old_diagnosis
    assert ok(api.request("POST", "/api/diagnoses", request, role="operator")) == old_diagnosis
    assert ok(api.request(
        "GET", f"/api/diagnoses/{new_diagnosis['diagnosis_id']}", role="operator"
    )) == new_diagnosis
    assert artifact_snapshot(first_path) == original_artifacts
    current_artifacts = artifact_snapshot(api.service.store.artifacts)
    assert all(current_artifacts[name] == content for name, content in old_artifacts.items())
    with api.service.store.connect() as db:
        assert tuple(db.execute(
            "SELECT * FROM diagnoses WHERE diagnosis_id=?", (diagnosis_id,)
        ).fetchone()) == original_row
        assert db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
        assert {row[0] for row in db.execute("SELECT version FROM model_approvals")} == {
            first["version"], second["version"],
        }
    audit = ok(api.request("GET", "/api/audit"))
    rollback_events = [row for row in audit if row["event"] == "model_rolled_back"]
    assert len(rollback_events) == 1
    assert rollback_events[0]["actor"] == "admin"
    assert json.loads(rollback_events[0]["payload"]) == rolled_back
    assert sum(row["event"] == "model_activated" for row in audit) == 2
    assert sum(row["event"] == "expert_feedback" for row in audit) == 1


def test_incorrect_synthetic_human_feedback_triggers_real_regression_gate(lifecycle_api):
    """Deliberately incorrect synthetic human labels, never forged model reports."""
    api = lifecycle_api
    original = api.original
    original_artifacts = artifact_snapshot(api.service.model_dir / original["version"])
    # One predetermined corruption amount; do not search/tune against held-out metrics.
    count = 256
    for index in range(count):
        request = make_request(
            "battery_depleted", seed=40_000 + index,
            device_id=f"incorrect-feedback-device-{index}",
            request_id=f"incorrect-feedback-request-{index}",
        ).model_dump(mode="json")
        diagnosis = ok(api.request("POST", "/api/diagnoses", request, role="operator"))
        assert diagnosis["model_version"] == original["version"]
        add_feedback(
            api, diagnosis["diagnosis_id"], feedback_id=f"incorrect-feedback-{index}",
            correct_class="charging_failure",
            actual_action="Deliberately incorrect synthetic human feedback for a release-gate test.",
            note="The simulator generated battery_depleted; this label is intentionally wrong.",
        )
    exported = ok(api.request("GET", "/api/feedback/export", role="expert"))
    assert len(exported) == count
    assert len({row["device_id"] for row in exported}) == count
    assert len({row["event_id"] for row in exported}) == count
    assert {row["confirmed_class"] for row in exported} == {"charging_failure"}
    held_out_devices = {
        device for partition in ("validation", "test")
        for device in original["splits"][partition]["device_ids"]
    }
    assert not held_out_devices & {row["device_id"] for row in exported}

    degraded = train_models(
        api.service.model_dir, seed=TRAIN_SEED, n_devices=TRAIN_DEVICES,
        feedback_records=exported,
    )
    assert degraded["config"] == original["config"]
    assert degraded["feedback"]["accepted_count"] == count
    assert degraded["feedback"]["rejected_count"] == 0
    assert degraded["feedback"]["duplicate_count"] == 0
    assert degraded["feedback"]["partition"] == "train_only"
    assert degraded["evaluation_fingerprint"] == original["evaluation_fingerprint"]
    for partition in ("validation", "test"):
        assert degraded["splits"][partition] == original["splits"][partition]
    previous = original["domains"]["low_voltage"]
    current = degraded["domains"]["low_voltage"]
    assert current["feedback_train_events"] == count
    assert current["train_events"] == previous["train_events"] + count
    assert current["test_macro_f1"] == degraded["metrics"]["low_voltage"]["test"]["macro_f1"]
    assert current["test_macro_f1"] < previous["test_macro_f1"]
    assert current["model_sha256"] != previous["model_sha256"]
    assert degraded["domains"]["safety"] == original["domains"]["safety"]
    assert degraded["version"] != original["version"]
    assert DiagnosisModels(api.service.model_dir, degraded["version"]).metadata == degraded
    assert api.service.register_model(degraded, api.admin) == degraded
    before = database_state(api.service)
    error(api.request("POST", "/api/models/activate", {
        "version": degraded["version"], "action_id": "reject-incorrect-feedback",
    }), 409, "regression_gate")
    assert database_state(api.service) == before
    assert active_version(api.service) == original["version"]
    error(api.request("POST", "/api/models/rollback", {
        "version": degraded["version"], "action_id": "reject-regression-rollback",
        "reason": "A rejected candidate must not bypass release approval through rollback.",
    }), 409, "rollback_unapproved")
    assert database_state(api.service) == before
    with api.service.store.connect() as db:
        assert [row[0] for row in db.execute("SELECT version FROM model_approvals")] == [
            original["version"],
        ]
    assert artifact_snapshot(api.service.model_dir / original["version"]) == original_artifacts


def test_different_real_evaluation_cannot_activate_or_become_rollback_target(lifecycle_api):
    api = lifecycle_api
    different = train_models(
        api.service.model_dir, seed=TRAIN_SEED + 1, n_devices=TRAIN_DEVICES
    )
    assert different["evaluation_fingerprint"] != api.original["evaluation_fingerprint"]
    assert different["splits"]["test"] != api.original["splits"]["test"]
    assert all(info["test_macro_f1"] >= 0.80 for info in different["domains"].values())
    assert DiagnosisModels(api.service.model_dir, different["version"]).metadata == different
    assert api.service.register_model(different, api.admin) == different
    before = database_state(api.service)
    error(api.request("POST", "/api/models/activate", {
        "version": different["version"], "action_id": "different-evaluation",
    }), 409, "evaluation_mismatch")
    assert database_state(api.service) == before
    error(api.request("POST", "/api/models/rollback", {
        "version": different["version"], "action_id": "unapproved-rollback",
        "reason": "A real registered bundle that never passed this release gate.",
    }), 409, "rollback_unapproved")
    assert database_state(api.service) == before
    assert active_version(api.service) == api.original["version"]
    _, result = diagnose(api)
    assert result["model_version"] == api.original["version"]


@pytest.mark.parametrize(("method", "path", "payload", "role"), [
    ("POST", "/api/models/rollback", {
        "version": "absent", "action_id": "denied", "reason": "Synthetic exercise",
    }, role) for role in (None, "operator", "expert")
] + [
    ("GET", "/api/feedback/export", None, role) for role in (None, "operator")
] + [
    ("POST", "/api/diagnoses/absent/assignment/close", {
        "action_id": "denied", "outcome": "cancelled", "note": "Synthetic exercise",
    }, role) for role in (None, "operator")
])
def test_new_routes_enforce_authentication_and_roles(lifecycle_api, method, path, payload, role):
    api = lifecycle_api
    before = database_state(api.service)
    response = api.request(method, path, payload, role)
    error(response, 401 if role is None else 403, "unauthenticated" if role is None else "forbidden")
    assert database_state(api.service) == before


@pytest.mark.parametrize(("path", "payload", "field"), [
    ("/api/models/rollback", {"version": "absent", "action_id": "invalid"}, "reason"),
    ("/api/models/rollback", {
        "version": "absent", "action_id": "invalid", "reason": "",
    }, "reason"),
    ("/api/models/rollback", {
        "version": "absent", "action_id": "invalid", "reason": "Review", "role": "admin",
    }, "role"),
    ("/api/diagnoses/absent/assignment/close", {
        "action_id": "invalid", "outcome": "reopened", "note": "Review",
    }, "outcome"),
    ("/api/diagnoses/absent/assignment/close", {
        "action_id": "invalid", "outcome": "completed", "note": "",
    }, "note"),
    ("/api/diagnoses/absent/assignment/close", {
        "action_id": "invalid", "outcome": "cancelled", "note": "Review", "role": "admin",
    }, "role"),
])
def test_lifecycle_route_validation_has_no_business_side_effects(lifecycle_api, path, payload, field):
    api = lifecycle_api
    before = database_state(api.service)
    response = api.request("POST", path, payload)
    assert response.status_code == 422, response.text
    assert any(item["loc"] == ["body", field] for item in response.json()["detail"])
    assert database_state(api.service) == before


def test_rollback_rejects_missing_target_no_change_and_reused_activation_action(lifecycle_api):
    api = lifecycle_api
    before = database_state(api.service)
    for version, action_id, code in (
        ("absent", "missing-target", "rollback_unapproved"),
        (api.original["version"], "no-change", "rollback_no_change"),
        (api.original["version"], "initial-activation", "action_conflict"),
    ):
        error(api.request("POST", "/api/models/rollback", {
            "version": version, "action_id": action_id, "reason": "Synthetic exercise",
        }), 409, code)
        assert database_state(api.service) == before


@pytest.mark.parametrize("role", ["expert", "admin"])
def test_feedback_export_deduplicates_agreement_but_rejects_real_label_conflict(lifecycle_api, role):
    api = lifecycle_api
    assert ok(api.request("GET", "/api/feedback/export", role=role)) == []
    request, result = diagnose(api)
    diagnosis_id = result["diagnosis_id"]
    first = add_feedback(api, diagnosis_id)
    assert add_feedback(api, diagnosis_id) == first
    add_feedback(api, diagnosis_id, feedback_id="second-review")
    exported = ok(api.request("GET", "/api/feedback/export", role=role))
    assert exported == [{
        "event_id": diagnosis_id, "device_id": request["device_id"], "request": request,
        "confirmed_class": "battery_depleted", "feedback_id": "second-review",
    }]
    # A distinct expert submission creates the conflict through the public API.
    conflicting = add_feedback(
        api, diagnosis_id, feedback_id="conflicting-review", correct_class="charging_failure"
    )
    assert conflicting["badcase_type"] == "model_disagreement"
    before = database_state(api.service)
    error(api.request("GET", "/api/feedback/export", role=role), 409, "label_conflict")
    assert database_state(api.service) == before
    assert ok(api.request("GET", f"/api/diagnoses/{diagnosis_id}", role="operator")) == result
    with api.service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 3


@pytest.mark.parametrize("role", ["expert", "admin"])
@pytest.mark.parametrize("outcome", ["cancelled", "completed"])
def test_assignment_close_releases_real_resources_once(lifecycle_api, role, outcome):
    api = lifecycle_api
    _, diagnosis = diagnose(api)
    diagnosis_id = diagnosis["diagnosis_id"]
    service_id = diagnosis["dispatch_candidates"][0]["service_id"]
    part = diagnosis["attribution"]["required_part"]
    before = resource(api.service, service_id)
    assigned = ok(api.request("POST", f"/api/diagnoses/{diagnosis_id}/assign", {
        "service_id": service_id, "action_id": "assignment",
    }, role="expert"))
    assert assigned["status"] == "synthetic_assignment_confirmed"
    assert assigned["repair_completed"] is False
    reserved = resource(api.service, service_id)
    assert reserved["active_jobs"] == before["active_jobs"] + 1
    assert reserved["parts"][part] == before["parts"][part] - 1
    path = f"/api/diagnoses/{diagnosis_id}/assignment/close"
    payload = {"action_id": "close", "outcome": outcome, "note": "Local synthetic job only."}
    before_close = database_state(api.service)
    error(api.request("POST", path, payload, role="operator"), 403, "forbidden")
    assert database_state(api.service) == before_close
    closed = ok(api.request("POST", path, payload, role=role))
    assert closed == {
        "diagnosis_id": diagnosis_id, "service_id": service_id,
        "status": f"synthetic_{outcome}", "repair_completed": outcome == "completed",
        "note": payload["note"],
    }
    expected = json.loads(json.dumps(before))
    if outcome == "completed":
        expected["parts"][part] -= 1
    assert resource(api.service, service_id) == expected
    after = database_state(api.service)
    assert ok(api.request("POST", path, payload, role=role)) == closed
    assert database_state(api.service) == after
    other_actor = "admin" if role == "expert" else "expert"
    for changed, actor, code in (
        ({**payload, "note": "Changed content"}, role, "action_conflict"),
        (payload, other_actor, "action_conflict"),
        ({**payload, "action_id": "second-close"}, role, "assignment_terminal"),
    ):
        error(api.request("POST", path, changed, role=actor), 409, code)
        assert database_state(api.service) == after
    assert ok(api.request("GET", f"/api/diagnoses/{diagnosis_id}", role="operator")) == diagnosis
    audit = ok(api.request("GET", "/api/audit", role="expert"))
    closed_events = [row for row in audit if row["event"] == "synthetic_assignment_closed"]
    assert len(closed_events) == 1
    assert closed_events[0]["actor"] == role
    assert json.loads(closed_events[0]["payload"]) == closed


@pytest.mark.parametrize("role", ["expert", "admin"])
def test_assignment_close_distinguishes_missing_diagnosis_and_missing_assignment(lifecycle_api, role):
    api = lifecycle_api
    _, diagnosis = diagnose(api)
    before = database_state(api.service)
    for diagnosis_id, code in (
        ("absent", "not_found"), (diagnosis["diagnosis_id"], "assignment_missing"),
    ):
        error(api.request("POST", f"/api/diagnoses/{diagnosis_id}/assignment/close", {
            "action_id": "missing-assignment", "outcome": "cancelled", "note": "No job exists.",
        }, role=role), 404, code)
        assert database_state(api.service) == before
