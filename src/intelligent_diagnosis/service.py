"""五阶段诊断、主动事件去重、专家反馈与版本发布的事务服务。"""

import json
from pathlib import Path
from uuid import uuid4

from .dispatch import DEFAULT_SERVICES, rank_services
from .features import extract_features
from .knowledge import KnowledgeBase
from .ml import DiagnosisModels
from .routing import classify_complaint
from .schemas import (
    Actor,
    AssignmentClose,
    DiagnosisRequest,
    DiagnosisResult,
    DispatchRequest,
    Domain,
    DomainError,
    Entry,
    FeedbackRequest,
    ModelActivation,
    ModelRollback,
    Outcome,
    SignalSample,
    TelemetryEvent,
)
from .storage import Store, canonical, digest, now_iso


class DiagnosisService:
    """业务顺序由程序控制；模型仅负责分类，不选择下一工具。"""

    def __init__(self, store: Store, model_dir: Path):
        self.store = store
        self.model_dir = Path(model_dir)
        default_knowledge = KnowledgeBase()
        with store.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO knowledge VALUES(?,?,'published','bootstrap','bootstrap',?)",
                (default_knowledge.version, canonical(default_knowledge.snapshot()), now_iso()),
            )
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES('active_knowledge',?)",
                (default_knowledge.version,),
            )
            for service in DEFAULT_SERVICES:
                db.execute(
                    "INSERT OR IGNORE INTO services VALUES(?,?)",
                    (service["service_id"], canonical(service)),
                )

    @staticmethod
    def require(actor: Actor, roles: set[str]) -> None:
        if actor.role not in roles:
            raise DomainError("forbidden", "当前角色没有执行该动作的权限", 403)

    @staticmethod
    def owned(db, diagnosis_id: str, actor: Actor):
        row = db.execute("SELECT * FROM diagnoses WHERE diagnosis_id=?", (diagnosis_id,)).fetchone()
        if row is None or (row["owner"] != actor.actor_id and actor.role == "operator"):
            raise DomainError("not_found", "诊断不存在或不在当前授权范围", 404)
        return row

    @staticmethod
    def knowledge(db) -> KnowledgeBase:
        row = db.execute(
            "SELECT snapshot FROM knowledge WHERE version="
            "(SELECT value FROM settings WHERE key='active_knowledge') AND status='published'"
        ).fetchone()
        if not row:
            raise DomainError("knowledge_unavailable", "没有已发布的知识版本", 503)
        return KnowledgeBase.from_snapshot(json.loads(row["snapshot"]))

    def _diagnose(self, db, request: DiagnosisRequest, actor: Actor) -> DiagnosisResult:
        request_payload = request.model_dump(mode="json")
        request_hash = digest(request_payload)
        prior = db.execute(
            "SELECT * FROM diagnoses WHERE owner=? AND request_id=?",
            (actor.actor_id, request.request_id),
        ).fetchone()
        if prior:
            if prior["request_hash"] != request_hash:
                raise DomainError("idempotency_conflict", "请求ID已用于不同内容", 409)
            return DiagnosisResult.model_validate_json(prior["result_json"])

        knowledge = self.knowledge(db)
        domain = request.domain
        result = DiagnosisResult(
            diagnosis_id=f"diag-{uuid4().hex}",
            request_id=request.request_id,
            device_id=request.device_id,
            entry=request.entry,
            domain=domain,
            outcome=Outcome.UNKNOWN,
            reason="尚未取得有效诊断",
            knowledge_version=knowledge.version,
            created_at=now_iso(),
            stages=["perception"],
        )
        result.stages.append("routing")
        routing_blocked = False
        if request.entry == Entry.USER:
            inferred, score = classify_complaint(request.complaint)
            if inferred is None:
                result.reason = "主诉未命中支持的诊断范围，转人工澄清"
                routing_blocked = True
            elif domain is not None and domain != inferred:
                result.reason = "主诉与指定故障域冲突，不能直接继续"
                routing_blocked = True
            else:
                domain = inferred
                result.domain = domain
        if domain is None:
            result.reason = "未明确故障域"
            routing_blocked = True
        if not routing_blocked and domain is not None:
            fault_id = request.fault_id or knowledge.fault_id(domain)
            if not knowledge.validate_fault(domain, fault_id):
                result.reason = "故障标识与领域配置不匹配"
                routing_blocked = True
        if not routing_blocked and domain is not None:
            result.stages.append("execution")
            features, issues = extract_features(request, domain)
            if issues:
                result.reason = "信号不满足诊断要求：" + "；".join(issues)
            else:
                active = db.execute(
                    "SELECT value FROM settings WHERE key='active_model'"
                ).fetchone()
                if not active:
                    result.reason = "无已批准模型，请先训练并激活版本"
                else:
                    version = active["value"]
                    try:
                        prediction = DiagnosisModels(self.model_dir, version).predict(
                            domain, features
                        )
                        result.prediction = prediction
                        result.model_version = version
                        result.evidence = knowledge.verify(domain, prediction.class_id, features)
                        if not prediction.accepted:
                            result.reason = "模型可靠性未通过：" + prediction.reliability_reason
                        elif not result.evidence:
                            result.reason = "该类别没有可验证条件，拒绝无证据确认"
                        elif result.evidence[0].status == "missing":
                            result.reason = "必要证据缺失，不能确认模型分类"
                        elif result.evidence[0].status == "refuted":
                            result.reason = "模型分类与当前事件证据冲突，需人工复核"
                            result.outcome = Outcome.REVIEW
                        else:
                            result.attribution = knowledge.resolve(domain, prediction.class_id)
                            result.outcome = (
                                Outcome.REVIEW
                                if result.attribution.safety_critical
                                else Outcome.RESOLVED
                            )
                            result.reason = (
                                "分类及证据一致，安全相关结果必须专家复核"
                                if result.attribution.safety_critical
                                else "分类及证据满足合成场景诊断规则；不表示已完成维修"
                            )
                            services = [
                                json.loads(row[0])
                                for row in db.execute("SELECT payload FROM services")
                            ]
                            result.dispatch_candidates = rank_services(
                                result.attribution, request.latitude, request.longitude, services
                            )
                    except DomainError as exc:
                        result.outcome = Outcome.UNKNOWN
                        result.attribution = None
                        result.dispatch_candidates = []
                        result.reason = f"诊断受控停止[{exc.code}]：{exc.message}"
        result.stages.extend(["gating", "feedback_pending"])
        artifact = self.store.put_artifact(
            {"request": request_payload, "result": result.model_dump(mode="json")}
        )
        db.execute(
            "INSERT INTO diagnoses VALUES(?,?,?,?,?,?,?)",
            (
                result.diagnosis_id,
                actor.actor_id,
                request.request_id,
                request_hash,
                canonical(request_payload),
                result.model_dump_json(),
                result.created_at,
            ),
        )
        self.store.audit(
            db,
            actor,
            "diagnosis_created",
            {
                "diagnosis_id": result.diagnosis_id,
                "outcome": result.outcome,
                "artifact": artifact,
            },
        )
        return result

    def diagnose(self, request: DiagnosisRequest, actor: Actor) -> DiagnosisResult:
        with self.store.transaction() as db:
            return self._diagnose(db, request, actor)

    def ingest(self, event: TelemetryEvent, actor: Actor) -> dict:
        """持续告警、事件时间水位、冷却及派生工单在同一事务中提交。"""
        payload_hash = digest(event.model_dump(mode="json"))
        with self.store.transaction() as db:
            prior = db.execute(
                "SELECT * FROM telemetry WHERE owner=? AND event_id=?",
                (actor.actor_id, event.event_id),
            ).fetchone()
            if prior:
                if prior["payload_hash"] != payload_hash:
                    raise DomainError("event_conflict", "事件ID对应不同数据", 409)
                return json.loads(prior["result_json"])
            timestamps = [s.timestamp_ms for s in event.samples]
            if timestamps != sorted(set(timestamps)):
                raise DomainError("event_order", "事件采样必须严格按时间递增且不可重复")
            latest = max(timestamps)
            state = db.execute(
                "SELECT * FROM device_events WHERE owner=? AND device_id=? AND domain=?",
                (actor.actor_id, event.device_id, event.domain),
            ).fetchone()
            if state and latest <= state["latest_ms"]:
                response = {
                    "status": "ignored",
                    "reason": "late_or_duplicate",
                    "diagnosis": None,
                    "diagnoses": [],
                    "trigger_count": 0,
                }
            else:
                buffer_row = db.execute(
                    "SELECT samples FROM telemetry_buffers WHERE owner=? AND device_id=? AND domain=?",
                    (actor.actor_id, event.device_id, event.domain),
                ).fetchone()
                previous = (
                    [SignalSample.model_validate(s) for s in json.loads(buffer_row[0])]
                    if buffer_row
                    else []
                )
                by_time = {s.timestamp_ms: s for s in previous}
                for sample in event.samples:
                    if sample.timestamp_ms in by_time and sample != by_time[sample.timestamp_ms]:
                        raise DomainError("sample_conflict", "同一设备时刻的采样内容冲突", 409)
                recent = previous
                last_trigger = state["last_trigger_ms"] if state else None
                diagnoses = []
                saw_cooldown = False
                for sample in event.samples:
                    if state and sample.timestamp_ms <= state["latest_ms"]:
                        continue
                    if recent and sample.timestamp_ms - recent[-1].timestamp_ms > 1000:
                        recent = []
                    recent.append(sample)
                    cutoff = sample.timestamp_ms - 2000
                    # 保留窗口左边界的前驱点，不能要求采样时刻恰好落在两秒边界。
                    while len(recent) > 1 and recent[1].timestamp_ms <= cutoff:
                        recent.pop(0)
                    if len(recent) > 1000:
                        raise DomainError("sample_density", "告警取证窗口含边界前驱超过1000点")
                    if (
                        len(recent) < 3
                        or sample.timestamp_ms - recent[0].timestamp_ms < 2000
                        or any(s.values.get("warning_active") != 1 for s in recent)
                    ):
                        continue
                    if last_trigger is not None and sample.timestamp_ms - last_trigger < 60_000:
                        saw_cooldown = True
                        continue
                    request = DiagnosisRequest(
                        request_id="auto-"
                        + digest(
                            {
                                "actor": actor.actor_id,
                                "device": event.device_id,
                                "domain": event.domain,
                                "trigger_ms": sample.timestamp_ms,
                            }
                        )[:24],
                        device_id=event.device_id,
                        entry=Entry.TELEMETRY,
                        domain=event.domain,
                        fault_id=self.knowledge(db).fault_id(event.domain),
                        fault_time_ms=sample.timestamp_ms - 1000,
                        samples=list(recent),
                        latitude=event.latitude,
                        longitude=event.longitude,
                    )
                    result = self._diagnose(db, request, actor)
                    diagnoses.append(result.model_dump(mode="json"))
                    last_trigger = sample.timestamp_ms
                db.execute(
                    "INSERT INTO telemetry_buffers VALUES(?,?,?,?) ON CONFLICT(owner,device_id,domain) "
                    "DO UPDATE SET samples=excluded.samples",
                    (
                        actor.actor_id,
                        event.device_id,
                        event.domain,
                        canonical([s.model_dump(mode="json") for s in recent]),
                    ),
                )
                response = {
                    "status": "triggered" if diagnoses else "ignored",
                    "reason": "sustained_alarm"
                    if diagnoses
                    else ("cooldown" if saw_cooldown else "not_sustained"),
                    "diagnosis": diagnoses[0] if diagnoses else None,
                    "diagnoses": diagnoses,
                    "trigger_count": len(diagnoses),
                }
                db.execute(
                    "INSERT INTO device_events VALUES(?,?,?,?,?) ON CONFLICT(owner,device_id,domain) "
                    "DO UPDATE SET latest_ms=excluded.latest_ms,last_trigger_ms=excluded.last_trigger_ms",
                    (actor.actor_id, event.device_id, event.domain, latest, last_trigger),
                )
            db.execute(
                "INSERT INTO telemetry VALUES(?,?,?,?)",
                (
                    actor.actor_id,
                    event.event_id,
                    payload_hash,
                    canonical(response),
                ),
            )
            self.store.audit(
                db,
                actor,
                "telemetry_ingested",
                {
                    "event_id": event.event_id,
                    "status": response["status"],
                    "reason": response["reason"],
                },
            )
            return response

    def get(self, diagnosis_id: str, actor: Actor) -> DiagnosisResult:
        with self.store.connect() as db:
            row = self.owned(db, diagnosis_id, actor)
            return DiagnosisResult.model_validate_json(row["result_json"])

    def list_diagnoses(self, actor: Actor) -> list[dict]:
        with self.store.connect() as db:
            query = "SELECT result_json FROM diagnoses"
            args = ()
            if actor.role == "operator":
                query += " WHERE owner=?"
                args = (actor.actor_id,)
            query += " ORDER BY created_at DESC LIMIT 100"
            return [json.loads(row[0]) for row in db.execute(query, args)]

    def feedback(self, diagnosis_id: str, request: FeedbackRequest, actor: Actor) -> dict:
        self.require(actor, {"expert", "admin"})
        payload = {"diagnosis_id": diagnosis_id, **request.model_dump()}
        with self.store.transaction() as db:
            row = self.owned(db, diagnosis_id, actor)
            diagnosis = DiagnosisResult.model_validate_json(row["result_json"])
            if diagnosis.domain is None:
                raise DomainError("domain_unresolved", "需先澄清故障域，不能直接写根因标签")
            historical = db.execute(
                "SELECT snapshot FROM knowledge WHERE version=?", (diagnosis.knowledge_version,)
            ).fetchone()
            if not historical:
                raise DomainError("knowledge_missing", "原诊断知识版本缺失，不能校验反馈", 409)
            KnowledgeBase.from_snapshot(json.loads(historical[0])).resolve(
                diagnosis.domain, request.correct_class
            )
            prior = db.execute(
                "SELECT * FROM feedback WHERE feedback_id=?", (request.feedback_id,)
            ).fetchone()
            if prior:
                if prior["payload_hash"] != digest(payload) or prior["actor"] != actor.actor_id:
                    raise DomainError("feedback_conflict", "反馈ID已用于其他内容", 409)
                return json.loads(prior["payload"])
            predicted = diagnosis.prediction.class_id if diagnosis.prediction else None
            payload["badcase_type"] = (
                "model_disagreement" if predicted != request.correct_class else "confirmed"
            )
            db.execute(
                "INSERT INTO feedback VALUES(?,?,?,?,?,?)",
                (
                    request.feedback_id,
                    diagnosis_id,
                    actor.actor_id,
                    digest({"diagnosis_id": diagnosis_id, **request.model_dump()}),
                    canonical(payload),
                    now_iso(),
                ),
            )
            self.store.audit(db, actor, "expert_feedback", payload)
            return payload

    def assign(self, diagnosis_id: str, request: DispatchRequest, actor: Actor) -> dict:
        self.require(actor, {"expert", "admin"})
        payload_hash = digest(
            {"operation": "assign", "diagnosis_id": diagnosis_id, **request.model_dump()}
        )
        with self.store.transaction() as db:
            row = self.owned(db, diagnosis_id, actor)
            prior = db.execute(
                "SELECT * FROM actions WHERE action_id=?", (request.action_id,)
            ).fetchone()
            if prior:
                if prior["actor"] != actor.actor_id or prior["payload_hash"] != payload_hash:
                    raise DomainError("action_conflict", "操作ID已被使用", 409)
                return json.loads(prior["response"])
            result = DiagnosisResult.model_validate_json(row["result_json"])
            if result.attribution is None:
                raise DomainError("no_attribution", "尚无可分派的归因，先补充诊断")
            old = db.execute(
                "SELECT * FROM assignments WHERE diagnosis_id=? OR action_id=?",
                (diagnosis_id, request.action_id),
            ).fetchone()
            if old:
                raise DomainError("assignment_conflict", "重复或冲突的服务确认", 409)
            original = DiagnosisRequest.model_validate_json(row["request_json"])
            services = [json.loads(r[0]) for r in db.execute("SELECT payload FROM services")]
            candidates = rank_services(
                result.attribution, original.latitude, original.longitude, services
            )
            if request.service_id not in {c["service_id"] for c in candidates}:
                raise DomainError("service_unavailable", "当前技能/库存/容量条件不满足", 409)
            service = next(s for s in services if s["service_id"] == request.service_id)
            service["active_jobs"] += 1
            if result.attribution.required_part:
                service["parts"][result.attribution.required_part] -= 1
            db.execute(
                "UPDATE services SET payload=? WHERE service_id=?",
                (canonical(service), request.service_id),
            )
            payload = {
                "diagnosis_id": diagnosis_id,
                "service_id": request.service_id,
                "status": "synthetic_assignment_confirmed",
                "repair_completed": False,
            }
            db.execute(
                "INSERT INTO assignments VALUES(?,?,?,?,?)",
                (
                    diagnosis_id,
                    request.action_id,
                    actor.actor_id,
                    request.service_id,
                    canonical(payload),
                ),
            )
            db.execute(
                "INSERT INTO actions VALUES(?,?,?,?)",
                (request.action_id, actor.actor_id, payload_hash, canonical(payload)),
            )
            self.store.audit(db, actor, "synthetic_assignment", payload)
            return payload

    def export_feedback(self, actor: Actor) -> list[dict]:
        """导出已专家确认的标签和原始请求；冲突标签不静默选择最新值。"""
        self.require(actor, {"expert", "admin"})
        with self.store.connect() as db:
            records = {}
            for row in db.execute(
                "SELECT f.payload,d.diagnosis_id,d.request_json FROM feedback f "
                "JOIN diagnoses d ON f.diagnosis_id=d.diagnosis_id ORDER BY f.created_at"
            ):
                feedback = json.loads(row["payload"])
                request = json.loads(row["request_json"])
                key = row["diagnosis_id"]
                label = feedback["correct_class"]
                if key in records and records[key]["confirmed_class"] != label:
                    raise DomainError(
                        "label_conflict", "同一诊断有冲突的专家标签，需先人工解决", 409
                    )
                records[key] = {
                    "event_id": key,
                    "device_id": request["device_id"],
                    "request": request,
                    "confirmed_class": label,
                    "feedback_id": feedback["feedback_id"],
                }
            return list(records.values())

    def close_assignment(self, diagnosis_id: str, request: AssignmentClose, actor: Actor) -> dict:
        """只推进本地合成工单，取消归还预留件，完成只释放工位。"""
        self.require(actor, {"expert", "admin"})
        payload_hash = digest(
            {"operation": "close_assignment", "diagnosis_id": diagnosis_id, **request.model_dump()}
        )
        with self.store.transaction() as db:
            self.owned(db, diagnosis_id, actor)
            prior = db.execute(
                "SELECT * FROM actions WHERE action_id=?", (request.action_id,)
            ).fetchone()
            if prior:
                if prior["actor"] != actor.actor_id or prior["payload_hash"] != payload_hash:
                    raise DomainError("action_conflict", "操作ID已被使用", 409)
                return json.loads(prior["response"])
            assignment = db.execute(
                "SELECT * FROM assignments WHERE diagnosis_id=?", (diagnosis_id,)
            ).fetchone()
            if not assignment:
                raise DomainError("assignment_missing", "无已确认的合成服务工单", 404)
            payload = json.loads(assignment["payload"])
            if payload["status"] != "synthetic_assignment_confirmed":
                raise DomainError(
                    "assignment_terminal", "合成服务工单已关闭，不能重复释放资源", 409
                )
            service_row = db.execute(
                "SELECT payload FROM services WHERE service_id=?", (assignment["service_id"],)
            ).fetchone()
            resource = json.loads(service_row[0])
            if resource["active_jobs"] < 1:
                raise DomainError("resource_state", "工位计数不一致，需对账", 409)
            resource["active_jobs"] -= 1
            result = DiagnosisResult.model_validate_json(
                self.owned(db, diagnosis_id, actor)["result_json"]
            )
            if request.outcome == "cancelled" and result.attribution.required_part:
                resource["parts"][result.attribution.required_part] += 1
            payload.update(
                status=f"synthetic_{request.outcome}",
                repair_completed=request.outcome == "completed",
                note=request.note,
            )
            db.execute(
                "UPDATE services SET payload=? WHERE service_id=?",
                (canonical(resource), assignment["service_id"]),
            )
            db.execute(
                "UPDATE assignments SET payload=? WHERE diagnosis_id=?",
                (canonical(payload), diagnosis_id),
            )
            db.execute(
                "INSERT INTO actions VALUES(?,?,?,?)",
                (request.action_id, actor.actor_id, payload_hash, canonical(payload)),
            )
            self.store.audit(db, actor, "synthetic_assignment_closed", payload)
            return payload

    def register_model(self, metadata: dict, actor: Actor) -> dict:
        self.require(actor, {"admin"})
        version = metadata.get("version", "")
        actual = DiagnosisModels(self.model_dir, version).metadata
        if metadata != actual:
            raise DomainError(
                "metadata_mismatch", "注册指标必须与已校验模型制品的评测报告一致", 409
            )
        with self.store.transaction() as db:
            existing = db.execute(
                "SELECT metadata FROM models WHERE version=?", (version,)
            ).fetchone()
            if existing and json.loads(existing[0]) != metadata:
                raise DomainError("version_conflict", "同一版本的模型报告不同", 409)
            db.execute(
                "INSERT OR IGNORE INTO models VALUES(?,?,?)",
                (version, canonical(metadata), now_iso()),
            )
            self.store.audit(db, actor, "model_registered", {"version": version})
        return metadata

    def activate_model(self, request: ModelActivation, actor: Actor) -> dict:
        self.require(actor, {"admin"})
        payload_hash = digest({"operation": "activate_model", **request.model_dump()})
        with self.store.transaction() as db:
            prior = db.execute(
                "SELECT * FROM actions WHERE action_id=?", (request.action_id,)
            ).fetchone()
            if prior:
                if prior["actor"] != actor.actor_id or prior["payload_hash"] != payload_hash:
                    raise DomainError("action_conflict", "操作ID已被使用", 409)
                return json.loads(prior["response"])
            row = db.execute(
                "SELECT metadata FROM models WHERE version=?", (request.version,)
            ).fetchone()
            if not row:
                raise DomainError("model_unregistered", "模型尚未注册", 404)
            meta = json.loads(row[0])
            actual = DiagnosisModels(self.model_dir, request.version).metadata
            if actual != meta:
                raise DomainError("metadata_mismatch", "注册报告与模型制品不同，拒绝激活", 409)
            self.check_knowledge(self.knowledge(db))
            for domain in Domain:
                metric = meta.get("domains", {}).get(domain.value, {})
                if metric.get("test_macro_f1", 0) < 0.80:
                    raise DomainError("regression_gate", "未达到合成回归最低宏F1示例门限0.80", 409)
            active = db.execute("SELECT value FROM settings WHERE key='active_model'").fetchone()
            if active and active[0] != request.version:
                old = json.loads(
                    db.execute(
                        "SELECT metadata FROM models WHERE version=?", (active[0],)
                    ).fetchone()[0]
                )
                if meta.get("evaluation_fingerprint") != old.get(
                    "evaluation_fingerprint"
                ) or not meta.get("evaluation_fingerprint"):
                    raise DomainError(
                        "evaluation_mismatch", "新旧模型未使用同一冻结评测集，不允许直接比较", 409
                    )
                for domain in Domain:
                    if (
                        meta["domains"][domain.value]["test_macro_f1"]
                        < old["domains"][domain.value]["test_macro_f1"]
                    ):
                        raise DomainError("regression_gate", "当前领域指标低于已启用模型", 409)
            db.execute(
                "INSERT INTO settings VALUES('active_model',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (request.version,),
            )
            db.execute(
                "INSERT OR IGNORE INTO model_approvals VALUES(?,?,?)",
                (request.version, now_iso(), actor.actor_id),
            )
            response = {
                "status": "activated",
                "version": request.version,
                "previous": active[0] if active else None,
            }
            db.execute(
                "INSERT INTO actions VALUES(?,?,?,?)",
                (request.action_id, actor.actor_id, payload_hash, canonical(response)),
            )
            self.store.audit(db, actor, "model_activated", response)
            return response

    def rollback_model(self, request: ModelRollback, actor: Actor) -> dict:
        """只允许回滚到历史真实通过发布门的制品，原因与指针变更原子入账。"""
        self.require(actor, {"admin"})
        payload_hash = digest({"operation": "rollback_model", **request.model_dump()})
        with self.store.transaction() as db:
            prior = db.execute(
                "SELECT * FROM actions WHERE action_id=?", (request.action_id,)
            ).fetchone()
            if prior:
                if prior["actor"] != actor.actor_id or prior["payload_hash"] != payload_hash:
                    raise DomainError("action_conflict", "操作ID已被使用", 409)
                return json.loads(prior["response"])
            approved = db.execute(
                "SELECT 1 FROM model_approvals WHERE version=?", (request.version,)
            ).fetchone()
            if not approved:
                raise DomainError("rollback_unapproved", "不能回滚到从未通过发布门的模型", 409)
            DiagnosisModels(self.model_dir, request.version)
            self.check_knowledge(self.knowledge(db))
            active = db.execute("SELECT value FROM settings WHERE key='active_model'").fetchone()
            if not active or active[0] == request.version:
                raise DomainError("rollback_no_change", "没有可回滚的版本变更", 409)
            db.execute("UPDATE settings SET value=? WHERE key='active_model'", (request.version,))
            response = {
                "status": "rolled_back",
                "version": request.version,
                "previous": active[0],
                "reason": request.reason,
            }
            db.execute(
                "INSERT INTO actions VALUES(?,?,?,?)",
                (request.action_id, actor.actor_id, payload_hash, canonical(response)),
            )
            self.store.audit(db, actor, "model_rolled_back", response)
            return response

    @staticmethod
    def check_knowledge(candidate: KnowledgeBase) -> None:
        """固定合成回归集检查映射、正反例和安全门；不替代企业专家验证。"""
        from .synthetic import make_request

        baseline = KnowledgeBase()
        classes = {
            Domain.SAFETY: (
                "circuit_open",
                "circuit_short",
                "safety_sensor_invalid",
                "safety_normal",
            ),
            Domain.LOW_VOLTAGE: (
                "battery_depleted",
                "charging_failure",
                "terminal_resistance",
                "voltage_normal",
            ),
        }
        for domain, labels in classes.items():
            for label in labels:
                attribution = candidate.resolve(domain, label)
                if (
                    baseline.resolve(domain, label).safety_critical
                    and not attribution.safety_critical
                ):
                    raise DomainError("knowledge_safety", "知识变更不能取消已有安全复核门", 409)
                for seed in (303, 509, 701):
                    features, issues = extract_features(
                        make_request(label, seed=seed, entry=Entry.TELEMETRY), domain
                    )
                    if issues:
                        raise DomainError(
                            "regression_fixture", "固定合成回归样本不满足信号要求", 409
                        )
                    checks = candidate.verify(domain, label, features)
                    if not checks or checks[0].status != "supported":
                        raise DomainError(
                            "knowledge_regression", f"知识未通过固定合成正例回归：{label}", 409
                        )
                    missing = candidate.verify(domain, label, {})
                    if not missing or missing[0].status == "supported":
                        raise DomainError("knowledge_regression", "缺失全部信号时不能确认归因", 409)
                    for other in labels:
                        if other == label:
                            continue
                        wrong = candidate.verify(domain, other, features)
                        if wrong and wrong[0].status == "supported":
                            raise DomainError(
                                "knowledge_regression", "互斥类别在固定回归样本上同时成立", 409
                            )

    def propose_knowledge(self, snapshot: dict, actor: Actor) -> dict:
        self.require(actor, {"expert", "admin"})
        knowledge = KnowledgeBase.from_snapshot(snapshot)
        knowledge.validate()
        with self.store.transaction() as db:
            old = db.execute(
                "SELECT snapshot FROM knowledge WHERE version=?", (knowledge.version,)
            ).fetchone()
            if old:
                raise DomainError("knowledge_version_exists", "知识版本已存在，不能覆盖", 409)
            db.execute(
                "INSERT INTO knowledge VALUES(?,?,'draft',?,NULL,?)",
                (knowledge.version, canonical(snapshot), actor.actor_id, now_iso()),
            )
            self.store.audit(db, actor, "knowledge_proposed", {"version": knowledge.version})
        return {"version": knowledge.version, "status": "draft"}

    def publish_knowledge(self, version: str, actor: Actor) -> dict:
        self.require(actor, {"admin"})
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM knowledge WHERE version=?", (version,)).fetchone()
            if not row:
                raise DomainError("knowledge_missing", "知识草稿不存在", 404)
            if row["proposer"] == actor.actor_id:
                raise DomainError("independent_review", "提案人不能批准自己的知识变更", 403)
            if row["status"] != "draft":
                raise DomainError("knowledge_state", "只有草稿可以审批发布", 409)
            candidate = KnowledgeBase.from_snapshot(json.loads(row["snapshot"]))
            candidate.validate()
            self.check_knowledge(candidate)
            db.execute(
                "UPDATE knowledge SET status='published',approver=? WHERE version=?",
                (actor.actor_id, version),
            )
            db.execute(
                "INSERT INTO settings VALUES('active_knowledge',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (version,),
            )
            self.store.audit(db, actor, "knowledge_published", {"version": version})
            return {"version": version, "status": "published"}

    def audit_log(self, actor: Actor) -> list[dict]:
        self.require(actor, {"expert", "admin"})
        with self.store.connect() as db:
            return [
                dict(row) for row in db.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT 200")
            ]
