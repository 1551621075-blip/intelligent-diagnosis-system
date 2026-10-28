"""本地命令行：训练、初始化身份、运行诊断、自检和启动工作台。"""

import json
import os
from pathlib import Path
from uuid import uuid4

import typer

from .schemas import Actor, DiagnosisRequest, DomainError, ModelActivation, ModelRollback
from .storage import Store

app = typer.Typer(no_args_is_help=True, help="原创合成诊断工程，不连接任何生产接口。")


def paths() -> tuple[Path, Path]:
    return Path(os.environ.get("DIAG_RUNTIME_DIR", "runtime")), Path(
        os.environ.get("DIAG_MODEL_DIR", "models")
    )


def print_json(value: dict | list) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def actor_from_env(store: Store) -> Actor:
    token = os.environ.get("DIAG_TOKEN", "")
    if not token:
        raise typer.BadParameter("先设置DIAG_TOKEN；不要把令牌写入源码或Git")
    return store.authenticate(token)


@app.command()
def user(actor_id: str, role: str = "operator"):
    """首次创建本地身份，只显示一次新生成的访问令牌。"""
    root, _ = paths()
    token = Store(root).provision_user(actor_id, role)
    typer.echo(f"已创建本地身份 {actor_id}（{role}）。以下新令牌只在本次显示，请妥善保管：")
    typer.echo(token)


@app.command()
def train(seed: int = 17, devices: int = 240, include_feedback: bool = False):
    """真实生成合成数据、分组训练与留出评测；不自动激活。"""
    from .ml import train_models
    from .service import DiagnosisService

    root, models = paths()
    store = Store(root)
    actor = actor_from_env(store)
    DiagnosisService.require(actor, {"admin"})
    service = DiagnosisService(store, models)
    records = service.export_feedback(actor) if include_feedback else []
    metadata = train_models(models, seed=seed, n_devices=devices, feedback_records=records)
    service.register_model(metadata, actor)
    print_json(metadata)


@app.command()
def activate(version: str):
    """对注册模型执行分域回归闸门后激活。"""
    from .service import DiagnosisService

    root, models = paths()
    store = Store(root)
    result = DiagnosisService(store, models).activate_model(
        ModelActivation(version=version, action_id=f"activate-{uuid4().hex}"), actor_from_env(store)
    )
    print_json(result)


@app.command()
def rollback(version: str, reason: str):
    """回滚到历史通过门控的已批准版本；保留原因与操作审计。"""
    from .service import DiagnosisService

    root, models = paths()
    store = Store(root)
    response = DiagnosisService(store, models).rollback_model(
        ModelRollback(version=version, action_id=f"rollback-{uuid4().hex}", reason=reason),
        actor_from_env(store),
    )
    print_json(response)


@app.command()
def export_feedback():
    """输出本地专家标签及来源请求，回流仍须避开冻结评测设备。"""
    from .service import DiagnosisService

    root, models = paths()
    store = Store(root)
    print_json(DiagnosisService(store, models).export_feedback(actor_from_env(store)))


@app.command()
def diagnose(input_file: Path):
    """读取本仓库用户准备的请求JSON并执行完整固定诊断流程。"""
    from .service import DiagnosisService

    root, models = paths()
    store = Store(root)
    if input_file.stat().st_size > 2_000_000:
        raise typer.BadParameter("请求文件超过2MB限制")
    request = DiagnosisRequest.model_validate_json(input_file.read_text())
    result = DiagnosisService(store, models).diagnose(request, actor_from_env(store))
    print_json(result.model_dump(mode="json"))


@app.command()
def example(class_id: str = "battery_depleted", seed: int = 12):
    """输出原创合成请求，所有设备与信号均非真实业务数据。"""
    from .synthetic import make_request

    print_json(
        make_request(class_id, seed=seed, request_id=f"example-{uuid4().hex}").model_dump(
            mode="json"
        )
    )


@app.command()
def serve(port: int = 8000):
    """仅监听127.0.0.1；不自动连接云端模型或部署到互联网。"""
    import uvicorn

    from .api import create_app

    root, models = paths()
    uvicorn.run(create_app(root=root, model_dir=models), host="127.0.0.1", port=port)


@app.command()
def demo(directory: Path = Path("runtime/demo"), seed: int = 17):
    """独立合成环境真实训练并跑两域案例，令牌不输出或提交。"""
    from .ml import train_models
    from .schemas import Entry, FeedbackRequest, TelemetryEvent
    from .service import DiagnosisService
    from .synthetic import make_request

    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise typer.BadParameter("演示目录非空，请使用新的目录，避免覆盖已有结果")
    store = Store(directory / "state")
    admin = store.authenticate(store.provision_user("demo-admin", "admin"))
    operator = store.authenticate(store.provision_user("demo-operator", "operator"))
    expert = store.authenticate(store.provision_user("demo-expert", "expert"))
    models = directory / "models"
    metadata = train_models(models, seed=seed, n_devices=240)
    service = DiagnosisService(store, models)
    service.register_model(metadata, admin)
    service.activate_model(
        ModelActivation(version=metadata["version"], action_id="initial-activation"), admin
    )
    outcomes = []
    for idx, class_id in enumerate(
        (
            "circuit_open",
            "circuit_short",
            "safety_sensor_invalid",
            "safety_normal",
            "battery_depleted",
            "charging_failure",
            "terminal_resistance",
            "voltage_normal",
        )
    ):
        request = make_request(class_id, seed=seed + idx + 100, request_id=f"demo-{class_id}")
        result = service.diagnose(request, operator)
        outcomes.append({"expected_class": class_id, "result": result.model_dump(mode="json")})
    automatic = make_request("battery_depleted", seed=93, entry=Entry.TELEMETRY)
    event = TelemetryEvent(
        event_id="demo-event",
        device_id=automatic.device_id,
        domain=automatic.domain,
        samples=automatic.samples,
    )
    first = service.ingest(event, operator)
    repeated = service.ingest(event, operator)
    assert first == repeated, "主动事件幂等性失败"
    diagnosis_id = outcomes[4]["result"]["diagnosis_id"]
    feedback = service.feedback(
        diagnosis_id,
        FeedbackRequest(
            feedback_id="demo-feedback",
            correct_class="battery_depleted",
            actual_action="合成环境：复核电池状态",
            repair_confirmed=False,
        ),
        expert,
    )
    report = {
        "synthetic_only": True,
        "model": metadata,
        "cases": outcomes,
        "telemetry": first,
        "duplicate_event_same_result": first == repeated,
        "feedback": feedback,
    }
    output = directory / "demo-report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print_json(
        {
            "report": str(output),
            "model_version": metadata["version"],
            "outcomes": [
                {
                    "expected": r["expected_class"],
                    "predicted": r["result"]["prediction"]["class_id"]
                    if r["result"]["prediction"]
                    else None,
                    "outcome": r["result"]["outcome"],
                    "reason": r["result"]["reason"],
                }
                for r in outcomes
            ],
        }
    )


@app.command()
def mine(input_file: Path):
    """从本仓库合成反馈统计关联候选；不自动发布因果知识。"""
    from .mining import mine_patterns

    if input_file.stat().st_size > 5_000_000:
        raise typer.BadParameter("模式文件超过5MB限制")
    result = mine_patterns(json.loads(input_file.read_text()))
    print_json({"status": "draft_requires_expert_review", "patterns": result})


def main() -> None:
    try:
        app()
    except DomainError as exc:
        typer.echo(f"{exc.code}: {exc.message}", err=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
