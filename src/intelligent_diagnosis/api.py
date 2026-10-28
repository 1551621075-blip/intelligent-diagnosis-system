"""本地合成诊断工作台；仅调用工厂时初始化存储，不在导入时写文件。"""

import json
import os
from math import isfinite
from pathlib import Path
from secrets import randbelow
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .schemas import (
    Actor,
    AssignmentClose,
    DiagnosisRequest,
    DiagnosisResult,
    DispatchRequest,
    DomainError,
    FeedbackRequest,
    Identifier,
    ModelActivation,
    ModelRollback,
    TelemetryEvent,
)
from .service import DiagnosisService
from .storage import Store
from .synthetic import CLASS_DOMAIN, make_request

MAX_BODY_BYTES = 1024 * 1024


def _reject_constant(value: str) -> None:
    raise ValueError("JSON数值必须有限")


def _finite_float(value: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError("JSON数值必须有限")
    return number


class RequestBoundary:
    """限长时实际读取流，不能只信任 Content-Length；错误也带安全响应头。"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def protected_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", (
                        b"default-src 'self'; script-src 'self'; style-src 'self'; "
                        b"connect-src 'self'; img-src 'self'; object-src 'none'; "
                        b"base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
                    )),
                ])
                if not scope["path"].startswith("/static/"):
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        async def fail(status: int, content: dict) -> None:
            await JSONResponse(content, status_code=status)(scope, receive, protected_send)

        headers = dict(scope.get("headers", []))
        declared_length = headers.get(b"content-length")
        if declared_length is not None:
            try:
                length = int(declared_length)
                if length < 0:
                    raise ValueError
            except ValueError:
                await fail(400, {"error": {
                    "code": "invalid_content_length", "message": "请求长度头无效",
                }})
                return
            if length > MAX_BODY_BYTES:
                await fail(413, {"error": {
                    "code": "body_too_large", "message": "请求体不能超过1 MiB",
                }})
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
                await fail(413, {"error": {
                    "code": "body_too_large", "message": "请求体不能超过1 MiB",
                }})
                return
            if not message.get("more_body", False):
                break

        content_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
        if body and (not content_type or content_type == b"application/json"
                     or content_type.endswith(b"+json")):
            try:
                json.loads(body, parse_constant=_reject_constant, parse_float=_finite_float)
            except (ValueError, UnicodeError, RecursionError):
                # 不回显原始JSON、Authorization或异常内部对象。
                await fail(422, {"detail": [{
                    "type": "json_invalid", "loc": ["body"],
                    "msg": "请求体必须是合法JSON，且所有数值必须有限",
                }]})
                return

        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, protected_send)


def create_app(root: Path | None = None, model_dir: Path | None = None) -> FastAPI:
    """root/model_dir 优先于环境变量；CLI须调用工厂，不要导入模块级app。"""
    runtime = Path(root if root is not None else os.environ.get("DIAG_RUNTIME_DIR", "runtime"))
    models = Path(model_dir if model_dir is not None else os.environ.get("DIAG_MODEL_DIR", "models"))
    store = Store(runtime.expanduser())
    service = DiagnosisService(store, models.expanduser().resolve())
    app = FastAPI(
        title="合成场景智能诊断 API",
        version="0.1.0",
        description="原创合成数据演示；预测概率不是物理置信度，派单不表示维修完成。",
        docs_url=None,
        redoc_url=None,
    )
    app.state.store = store
    app.state.service = service
    app.add_middleware(RequestBoundary)

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message}},
            status_code=exc.status,
            headers={"WWW-Authenticate": "Bearer"} if exc.status == 401 else None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic的ctx可能含ValueError，input可能含NaN或敏感文本；只返回标准定位字段。
        return JSONResponse({"detail": [
            {"type": item["type"], "loc": list(item["loc"]), "msg": item["msg"]}
            for item in exc.errors()
        ]}, status_code=422)

    bearer = HTTPBearer(auto_error=False)

    def current_actor(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> Actor:
        if credentials is None or not credentials.credentials.strip():
            raise DomainError("unauthenticated", "需要有效本地令牌", 401)
        return store.authenticate(credentials.credentials)

    # 静态资源随wheel安装，不能依赖仓库所在的绝对路径。
    assets = Path(__file__).resolve().parent
    app.mount("/static", StaticFiles(directory=assets / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def workstation() -> FileResponse:
        return FileResponse(assets / "templates/index.html", media_type="text/html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "synthetic": True}

    @app.get("/api/me", response_model=Actor)
    def me(actor: Annotated[Actor, Depends(current_actor)]) -> Actor:
        return actor

    @app.get("/api/examples/{class_id}", response_model=DiagnosisRequest)
    def example(class_id: str, actor: Annotated[Actor, Depends(current_actor)]) -> DiagnosisRequest:
        if class_id not in CLASS_DOMAIN:
            raise DomainError("example_not_found", "没有该合成类别样例", 404)
        return make_request(
            class_id, seed=randbelow(2**31), request_id=f"web-{uuid4().hex}",
        )

    @app.get("/api/diagnoses", response_model=list[DiagnosisResult])
    def diagnoses(actor: Annotated[Actor, Depends(current_actor)]) -> list[dict]:
        return service.list_diagnoses(actor)

    @app.post("/api/diagnoses", response_model=DiagnosisResult)
    def diagnose(
        payload: DiagnosisRequest, actor: Annotated[Actor, Depends(current_actor)],
    ) -> DiagnosisResult:
        return service.diagnose(payload, actor)

    @app.get("/api/diagnoses/{diagnosis_id}", response_model=DiagnosisResult)
    def diagnosis(
        diagnosis_id: Identifier, actor: Annotated[Actor, Depends(current_actor)],
    ) -> DiagnosisResult:
        return service.get(diagnosis_id, actor)

    @app.post("/api/telemetry")
    def telemetry(
        payload: TelemetryEvent, actor: Annotated[Actor, Depends(current_actor)],
    ) -> dict:
        return service.ingest(payload, actor)

    @app.post("/api/diagnoses/{diagnosis_id}/feedback")
    def feedback(
        diagnosis_id: Identifier, payload: FeedbackRequest,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> dict:
        return service.feedback(diagnosis_id, payload, actor)

    @app.post("/api/diagnoses/{diagnosis_id}/assign")
    def assign(
        diagnosis_id: Identifier, payload: DispatchRequest,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> dict:
        return service.assign(diagnosis_id, payload, actor)

    @app.get("/api/audit")
    def audit(actor: Annotated[Actor, Depends(current_actor)]) -> list[dict]:
        return service.audit_log(actor)

    @app.post("/api/models/activate")
    def activate(
        payload: ModelActivation, actor: Annotated[Actor, Depends(current_actor)],
    ) -> dict:
        return service.activate_model(payload, actor)

    @app.post("/api/models/rollback")
    def rollback(payload: ModelRollback, actor: Annotated[Actor, Depends(current_actor)]) -> dict:
        return service.rollback_model(payload, actor)

    @app.get("/api/feedback/export")
    def export_feedback(actor: Annotated[Actor, Depends(current_actor)]) -> list[dict]:
        return service.export_feedback(actor)

    @app.post("/api/diagnoses/{diagnosis_id}/assignment/close")
    def close_assignment(
        diagnosis_id: Identifier, payload: AssignmentClose,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> dict:
        return service.close_assignment(diagnosis_id, payload, actor)

    @app.get("/api/knowledge")
    def knowledge(actor: Annotated[Actor, Depends(current_actor)]) -> dict:
        with store.connect() as db:
            return service.knowledge(db).snapshot()

    @app.post("/api/knowledge/proposals")
    def propose(
        snapshot: Annotated[dict[str, Any], Body()],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> dict:
        return service.propose_knowledge(snapshot, actor)

    @app.post("/api/knowledge/{version}/publish")
    def publish(version: Identifier, actor: Annotated[Actor, Depends(current_actor)]) -> dict:
        return service.publish_knowledge(version, actor)

    return app
