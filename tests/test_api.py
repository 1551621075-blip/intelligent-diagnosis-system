"""真实 Store/DiagnosisService 集成，不使用替代业务结果的 mock。"""

import asyncio
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intelligent_diagnosis.api import MAX_BODY_BYTES, RequestBoundary, create_app
from intelligent_diagnosis.service import DiagnosisService
from intelligent_diagnosis.storage import Store
from intelligent_diagnosis.synthetic import make_request


class TestTokens(dict):
    __test__ = False

    def __repr__(self):
        return "<test bearer tokens redacted>"


@pytest.fixture
def api_context(tmp_path):
    app = create_app(tmp_path / "runtime", tmp_path / "models")
    tokens = TestTokens({
        "owner": app.state.store.provision_user("owner", "operator"),
        "other": app.state.store.provision_user("other", "operator"),
        "expert": app.state.store.provision_user("expert", "expert"),
        "admin": app.state.store.provision_user("admin", "admin"),
        "reviewer": app.state.store.provision_user("reviewer", "admin"),
    })
    with TestClient(app) as client:
        yield client, app, tokens


def auth(tokens, name="owner"):
    return {"Authorization": f"Bearer {tokens[name]}"}


def request_payload(request_id="api-test"):
    return make_request("circuit_open", seed=7, request_id=request_id).model_dump(mode="json")


def create_diagnosis(client, tokens, request_id="api-test", owner="owner"):
    response = client.post("/api/diagnoses", json=request_payload(request_id), headers=auth(tokens, owner))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(("method", "path", "body"), [
    ("GET", "/api/me", None),
    ("GET", "/api/examples/circuit_open", None),
    ("GET", "/api/diagnoses", None),
    ("GET", "/api/diagnoses/not-yours", None),
    ("POST", "/api/diagnoses", {}),
    ("POST", "/api/telemetry", {}),
    ("POST", "/api/diagnoses/not-yours/feedback", {}),
    ("POST", "/api/diagnoses/not-yours/assign", {}),
    ("GET", "/api/audit", None),
    ("POST", "/api/models/activate", {}),
    ("POST", "/api/models/rollback", {}),
    ("GET", "/api/feedback/export", None),
    ("POST", "/api/diagnoses/not-yours/assignment/close", {}),
    ("GET", "/api/knowledge", None),
    ("POST", "/api/knowledge/proposals", {}),
    ("POST", "/api/knowledge/unknown/publish", None),
])
def test_all_business_routes_require_authorization(api_context, method, path, body):
    client, _, _ = api_context
    response = client.request(method, path, json=body)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"error": {"code": "unauthenticated", "message": "需要有效本地令牌"}}


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer invalid-private-token"},
    {"Authorization": "Basic invalid-private-token"},
    {"Authorization": "Bearer"},
    {"X-Role": "admin", "X-Actor-ID": "admin"},
])
def test_invalid_auth_and_headers_do_not_supply_identity(api_context, headers):
    client, _, _ = api_context
    response = client.get("/api/me", headers=headers)
    assert response.status_code == 401
    assert "invalid-private-token" not in response.text


def test_tokens_in_url_or_cookie_are_not_authentication(api_context):
    client, _, tokens = api_context
    assert client.get("/api/me", params={"token": tokens["owner"]}).status_code == 401
    assert client.get("/api/me", headers={"Cookie": f"token={tokens['owner']}"}).status_code == 401


def test_me_only_returns_server_identity(api_context):
    client, _, tokens = api_context
    response = client.get("/api/me", headers={**auth(tokens), "X-Role": "admin"})
    assert response.json() == {"actor_id": "owner", "role": "operator"}
    assert all(token not in response.text for token in tokens.values())


def test_real_service_no_active_model_persists_unknown_not_fake_prediction(api_context):
    client, app, tokens = api_context
    assert isinstance(app.state.service, DiagnosisService)
    assert isinstance(app.state.store, Store)
    result = create_diagnosis(client, tokens)
    assert result["outcome"] == "unknown"
    assert result["prediction"] is None
    assert result["model_version"] is None
    assert result["attribution"] is None
    assert result["dispatch_candidates"] == []
    assert "无已批准模型" in result["reason"]
    assert result["stages"] == ["perception", "routing", "execution", "gating", "feedback_pending"]
    fetched = client.get(f"/api/diagnoses/{result['diagnosis_id']}", headers=auth(tokens))
    assert fetched.json() == result
    assert client.get("/api/diagnoses", headers=auth(tokens)).json() == [result]
    with app.state.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0] == 1
        assert db.execute("SELECT value FROM settings WHERE key='active_model'").fetchone() is None
    assert len(list(app.state.store.artifacts.glob("*.json"))) == 1


def test_operator_cannot_read_another_operators_diagnosis(api_context):
    client, _, tokens = api_context
    result = create_diagnosis(client, tokens)
    path = f"/api/diagnoses/{result['diagnosis_id']}"
    response = client.get(path, headers=auth(tokens, "other"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert client.get("/api/diagnoses", headers=auth(tokens, "other")).json() == []
    assert client.get(path, headers=auth(tokens, "expert")).json() == result
    assert client.get(path, headers=auth(tokens, "admin")).json() == result


@pytest.mark.parametrize("forged", [
    {"role": "admin"}, {"actor_id": "admin"},
    {"actor": {"actor_id": "admin", "role": "admin"}},
])
def test_request_body_cannot_forge_actor_or_role(api_context, forged):
    client, _, tokens = api_context
    response = client.post("/api/diagnoses", json={**request_payload(), **forged}, headers=auth(tokens))
    assert response.status_code == 422
    assert any(item["type"] == "extra_forbidden" for item in response.json()["detail"])
    assert client.get("/api/diagnoses", headers=auth(tokens)).json() == []


def test_privileged_actions_use_authenticated_role(api_context):
    client, _, tokens = api_context
    result = create_diagnosis(client, tokens)
    operator_headers = {**auth(tokens), "X-Role": "admin", "X-Actor-ID": "admin"}
    routes = [
        (f"/api/diagnoses/{result['diagnosis_id']}/feedback", {
            "feedback_id": "f-1", "correct_class": "circuit_open", "actual_action": "人工检查",
        }),
        (f"/api/diagnoses/{result['diagnosis_id']}/assign", {"service_id": "demo-north", "action_id": "a-1"}),
        ("/api/models/activate", {"version": "missing", "action_id": "m-1"}),
        ("/api/knowledge/proposals", {"role": "admin", "actor_id": "admin"}),
        ("/api/knowledge/missing/publish", {"role": "admin"}),
    ]
    for path, payload in routes:
        response = client.post(path, json=payload, headers=operator_headers)
        assert response.status_code == 403, (path, response.text)
        assert response.json()["error"]["code"] == "forbidden"
    assert client.get("/api/audit", headers=operator_headers).status_code == 403
    assert client.post("/api/models/activate", json=routes[2][1], headers=auth(tokens, "expert")).status_code == 403


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999", '"NaN"', '"Infinity"'])
def test_nonfinite_signal_is_standard_422(api_context, value):
    client, _, tokens = api_context
    payload = request_payload()
    payload["samples"][0]["values"]["resistance_ohm"] = "__NONFINITE__"
    raw = json.dumps(payload).replace('"__NONFINITE__"', value)
    response = client.post("/api/diagnoses", content=raw, headers={**auth(tokens), "Content-Type": "application/json"})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert client.get("/api/diagnoses", headers=auth(tokens)).json() == []


@pytest.mark.parametrize(("path", "raw"), [
    ("/api/diagnoses", '{"request_id":'),
    ("/api/diagnoses", '{"complaint":"private-invalid-input"}'),
    ("/api/diagnoses", '[]'),
    ("/api/diagnoses", '{"request_id":"r","device_id":"d","fault_time_ms":0}'),
    ("/api/knowledge/proposals", '{"version":"v","nested":{"value":NaN}}'),
])
def test_invalid_json_and_schema_return_serializable_422(api_context, path, raw):
    client, _, tokens = api_context
    response = client.post(path, content=raw, headers={**auth(tokens, "expert"), "Content-Type": "application/json"})
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "private-invalid-input" not in response.text


@pytest.mark.parametrize("field", ["fault_time_ms", "sample", "telemetry"])
@pytest.mark.parametrize("value", [2**63, -(2**63), 10**100])
def test_out_of_range_timestamps_return_422_without_database_write(api_context, field, value):
    client, app, tokens = api_context
    payload = request_payload()
    path = "/api/diagnoses"
    if field == "fault_time_ms":
        payload[field] = value
    else:
        payload["samples"][0]["timestamp_ms"] = value
        if field == "telemetry":
            payload = {"event_id": "huge-time", "device_id": payload["device_id"],
                       "domain": "safety", "samples": payload["samples"]}
            path = "/api/telemetry"
    response = client.post(path, json=payload, headers=auth(tokens))
    assert response.status_code == 422
    with app.state.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0] == 0


def test_no_active_model_request_retries_are_real_idempotent_transactions(api_context):
    client, _, tokens = api_context
    first = create_diagnosis(client, tokens)
    second = create_diagnosis(client, tokens)
    assert first == second
    changed = request_payload()
    changed["complaint"] += "补充信息"
    response = client.post("/api/diagnoses", json=changed, headers=auth(tokens))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "idempotency_conflict"


def test_examples_are_fresh_authorized_real_schema_inputs(api_context):
    client, _, tokens = api_context
    one = client.get("/api/examples/circuit_open", headers=auth(tokens)).json()
    two = client.get("/api/examples/circuit_open", headers=auth(tokens)).json()
    assert one["request_id"] != two["request_id"]
    assert len(one["samples"]) == 9
    assert "class_id" not in one
    response = client.post("/api/diagnoses", json=one, headers=auth(tokens))
    assert response.status_code == 200
    assert "无已批准模型" in response.json()["reason"]
    assert client.get("/api/examples/not-a-class", headers=auth(tokens)).status_code == 404


def test_telemetry_calls_real_service_and_deduplicates(api_context):
    client, _, tokens = api_context
    request = request_payload()
    for sample in request["samples"]:
        sample["values"]["warning_active"] = 0
    event = {
        "event_id": "quiet-event", "device_id": request["device_id"],
        "domain": "safety", "samples": request["samples"],
    }
    first = client.post("/api/telemetry", json=event, headers=auth(tokens))
    assert first.status_code == 200
    assert first.json() == {"status": "ignored", "reason": "not_sustained", "diagnosis": None,
                            "diagnoses": [], "trigger_count": 0}
    assert client.post("/api/telemetry", json=event, headers=auth(tokens)).json() == first.json()
    assert client.get("/api/diagnoses", headers=auth(tokens)).json() == []


def test_feedback_audit_and_assignment_rejection_are_real_business_operations(api_context):
    client, app, tokens = api_context
    result = create_diagnosis(client, tokens)
    note = '<img src=x onerror="alert(1)">'
    feedback = {
        "feedback_id": "actual-review", "correct_class": "circuit_open",
        "actual_action": "人工检查合成采样，未宣称完成真实维修", "note": note,
    }
    response = client.post(f"/api/diagnoses/{result['diagnosis_id']}/feedback", json=feedback, headers=auth(tokens, "expert"))
    assert response.status_code == 200
    assert response.json()["repair_confirmed"] is False
    assert response.json()["note"] == note
    audit = client.get("/api/audit", headers=auth(tokens, "expert"))
    assert audit.status_code == 200
    assert any(row["event"] == "expert_feedback" and json.loads(row["payload"])["note"] == note for row in audit.json())
    assert all(token not in audit.text for token in tokens.values())
    rejected = client.post(f"/api/diagnoses/{result['diagnosis_id']}/assign", json={"service_id": "demo-north", "action_id": "no-attribution"}, headers=auth(tokens, "expert"))
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "no_attribution"
    with app.state.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == 0


def test_knowledge_requires_independent_review_and_model_registration(api_context):
    client, _, tokens = api_context
    snapshot = client.get("/api/knowledge", headers=auth(tokens)).json()
    current_version = snapshot["version"]
    snapshot["version"] = "knowledge-v2"
    proposed = client.post("/api/knowledge/proposals", json=snapshot, headers=auth(tokens, "admin"))
    assert proposed.status_code == 200, proposed.text
    assert proposed.json() == {"version": "knowledge-v2", "status": "draft"}
    assert client.get("/api/knowledge", headers=auth(tokens)).json()["version"] == current_version
    path = "/api/knowledge/knowledge-v2/publish"
    same_actor = client.post(path, headers=auth(tokens, "admin"))
    assert same_actor.status_code == 403
    assert same_actor.json()["error"]["code"] == "independent_review"
    approved = client.post(path, headers=auth(tokens, "reviewer"))
    assert approved.status_code == 200, approved.text
    assert client.get("/api/knowledge", headers=auth(tokens)).json()["version"] == "knowledge-v2"
    activation = client.post("/api/models/activate", json={"version": "not-registered", "action_id": "activation-1"}, headers=auth(tokens, "admin"))
    assert activation.status_code == 404
    assert activation.json()["error"]["code"] == "model_unregistered"


def test_public_health_static_workstation_and_openapi_do_not_expose_tokens(api_context):
    client, _, tokens = api_context
    health = client.get("/health")
    assert health.json() == {"status": "ok", "synthetic": True}
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert 'id="token-input"' in page.text
    assert "物理置信度" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["cache-control"] == "no-store"
    script = client.get("/static/app.js")
    css = client.get("/static/style.css")
    assert script.status_code == css.status_code == 200
    assert "textContent" in script.text
    for forbidden in ("localStorage", "sessionStorage", "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert forbidden not in script.text
    assert not any(token in health.text + page.text + script.text for token in tokens.values())
    cross_origin = client.options("/api/me", headers={"Origin": "https://untrusted.invalid", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in cross_origin.headers
    assert client.get("/static/../runtime/diagnosis.sqlite3").status_code == 404


def test_openapi_documents_schemas_and_bearer_security(api_context):
    client, _, _ = api_context
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    request_schema = schema["components"]["schemas"]["DiagnosisRequest"]
    assert request_schema["additionalProperties"] is False
    assert "request_id" in request_schema["required"]
    assert "role" not in request_schema["properties"]
    post = schema["paths"]["/api/diagnoses"]["post"]
    assert post["security"] == [{"HTTPBearer": []}]
    assert post["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("DiagnosisRequest")
    assert post["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("DiagnosisResult")
    assert "security" not in schema["paths"]["/health"]["get"]


def test_request_body_limit(api_context):
    client, _, tokens = api_context
    response = client.post("/api/diagnoses", content=b"x" * (MAX_BODY_BYTES + 1), headers=auth(tokens))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_too_large"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_body_limit_checks_stream_even_without_declared_length():
    async def run():
        messages = iter([
            {"type": "http.request", "body": b"x" * MAX_BODY_BYTES, "more_body": True},
            {"type": "http.request", "body": b"x", "more_body": False},
        ])
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        async def downstream(scope, receive, send):
            pytest.fail("Oversized streamed body must not reach the service")

        await RequestBoundary(downstream)({"type": "http", "path": "/api/diagnoses", "headers": []}, receive, send)
        assert sent[0]["status"] == 413
        assert json.loads(sent[1]["body"])["error"]["code"] == "body_too_large"

    asyncio.run(run())


def test_factory_paths_and_import_have_no_runtime_side_effects(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "env-runtime"
    models = tmp_path / "env-models"
    monkeypatch.setenv("DIAG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("DIAG_MODEL_DIR", str(models))
    environment = {**os.environ, "PYTHONPATH": str(project / "src")}
    process = subprocess.run(
        [sys.executable, "-c", "import intelligent_diagnosis.api; print('import-ok')"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "import-ok" in process.stdout
    assert not runtime.exists()
    assert not models.exists()
    assert not (tmp_path / "runtime").exists()
    app = create_app()
    assert app.state.store.root == runtime.resolve()
    assert app.state.service.model_dir == models.resolve()
    explicit = create_app(tmp_path / "explicit-runtime", tmp_path / "explicit-models")
    assert explicit.state.store.root == (tmp_path / "explicit-runtime").resolve()
    assert explicit.state.service.model_dir == (tmp_path / "explicit-models").resolve()
    module = importlib.import_module("intelligent_diagnosis.api")
    assert not hasattr(module, "app")
