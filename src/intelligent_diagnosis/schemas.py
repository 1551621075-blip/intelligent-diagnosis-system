"""输入、预测、诊断与人工处置的类型；缺失、失败和业务动作分开记录。"""

from enum import StrEnum
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")]
Timestamp = Annotated[int, Field(ge=0, le=2**63 - 1)]


class Domain(StrEnum):
    SAFETY = "safety"
    LOW_VOLTAGE = "low_voltage"


class Entry(StrEnum):
    USER = "user"
    TELEMETRY = "telemetry"


class Outcome(StrEnum):
    RESOLVED = "resolved"
    REVIEW = "review_required"
    UNKNOWN = "unknown"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SignalSample(StrictModel):
    timestamp_ms: Timestamp
    values: dict[str, float | None]

    @field_validator("values")
    @classmethod
    def finite_values(cls, values: dict[str, float | None]) -> dict[str, float | None]:
        if len(values) > 32 or any(v is not None and not isfinite(v) for v in values.values()):
            raise ValueError("信号必须为有限数值，单点最多32个字段")
        return values


class DiagnosisRequest(StrictModel):
    request_id: Identifier
    device_id: Identifier
    entry: Entry = Entry.USER
    complaint: str = Field(default="", max_length=2000)
    domain: Domain | None = None
    fault_id: Identifier | None = None
    fault_time_ms: Timestamp
    samples: list[SignalSample] = Field(default_factory=list, max_length=1000)
    latitude: float = Field(default=31.0, ge=-90, le=90)
    longitude: float = Field(default=121.0, ge=-180, le=180)

    @model_validator(mode="after")
    def entry_required(self) -> "DiagnosisRequest":
        if self.entry == Entry.USER and not self.complaint.strip():
            raise ValueError("用户入口必须提供主诉")
        if self.entry == Entry.TELEMETRY and (self.domain is None or self.fault_id is None):
            raise ValueError("主动入口必须绑定故障域及故障标识")
        return self


class TelemetryEvent(StrictModel):
    event_id: Identifier
    device_id: Identifier
    domain: Domain
    samples: list[SignalSample] = Field(min_length=1, max_length=1000)
    latitude: float = Field(default=31.0, ge=-90, le=90)
    longitude: float = Field(default=121.0, ge=-180, le=180)


class Prediction(StrictModel):
    domain: Domain
    class_id: str
    scores: dict[str, float]
    model_version: str
    accepted: bool
    reliability_reason: str


class EvidenceCheck(StrictModel):
    condition: str
    status: Literal["supported", "refuted", "missing"]
    observed: float | None = None
    description: str


class Attribution(StrictModel):
    attribution_id: str
    name: str
    solution_id: str
    solution: str
    required_skill: str
    required_part: str | None = None
    safety_critical: bool = False


class DiagnosisResult(StrictModel):
    diagnosis_id: str
    request_id: str
    device_id: str
    entry: Entry
    domain: Domain | None
    outcome: Outcome
    reason: str
    prediction: Prediction | None = None
    attribution: Attribution | None = None
    evidence: list[EvidenceCheck] = Field(default_factory=list)
    knowledge_version: str
    model_version: str | None = None
    dispatch_candidates: list[dict] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    created_at: str


class FeedbackRequest(StrictModel):
    feedback_id: Identifier
    correct_class: str
    actual_action: str = Field(min_length=1, max_length=1000)
    repair_confirmed: bool = False
    note: str = Field(default="", max_length=2000)


class DispatchRequest(StrictModel):
    service_id: Identifier
    action_id: Identifier


class ModelActivation(StrictModel):
    version: Identifier
    action_id: Identifier


class AssignmentClose(StrictModel):
    action_id: Identifier
    outcome: Literal["cancelled", "completed"]
    note: str = Field(min_length=1, max_length=1000)


class ModelRollback(ModelActivation):
    reason: str = Field(min_length=1, max_length=1000)


class Actor(StrictModel):
    actor_id: str
    role: Literal["operator", "expert", "admin"]


class DomainError(Exception):
    """携带稳定错误码；不能把异常当成无故障。"""

    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)
