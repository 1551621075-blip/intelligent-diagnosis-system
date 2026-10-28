"""命令行入口真实调用，不把领域错误显示成未处理的Python异常。"""

import json
import os
import subprocess
import sys

from intelligent_diagnosis.storage import Store


def invoke(tmp_path, *args, token=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DIAG_")}
    env.update(DIAG_RUNTIME_DIR=str(tmp_path / "state"), DIAG_MODEL_DIR=str(tmp_path / "models"))
    if token:
        env["DIAG_TOKEN"] = token
    return subprocess.run(
        [sys.executable, "-m", "intelligent_diagnosis.cli", *args],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )


def test_cli_example_and_help_do_not_require_identity(tmp_path):
    response = invoke(tmp_path, "--help")
    assert response.returncode == 0
    assert "export-feedback" in response.stdout
    response = invoke(tmp_path, "example", "--class-id", "battery_depleted")
    assert response.returncode == 0
    payload = json.loads(response.stdout)
    assert payload["entry"] == "user"
    assert payload["complaint"]
    assert "voltage" in payload["samples"][0]["values"]
    assert not (tmp_path / "state").exists()


def test_cli_domain_error_is_concise_and_secret_free(tmp_path):
    store = Store(tmp_path / "state")
    token = store.provision_user("admin", "admin")
    response = invoke(tmp_path, "activate", "missing-version", token=token)
    assert response.returncode == 1
    assert "model_unregistered" in response.stderr
    assert "Traceback" not in response.stderr
    assert token not in response.stderr + response.stdout


def test_cli_role_denial_and_occupied_demo_are_non_destructive(tmp_path):
    store = Store(tmp_path / "state")
    token = store.provision_user("operator", "operator")
    response = invoke(tmp_path, "train", "--devices", "24", token=token)
    assert response.returncode == 1
    assert "forbidden" in response.stderr
    assert not (tmp_path / "models").exists()
    occupied = tmp_path / "existing-demo"
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("original content")
    response = invoke(tmp_path, "demo", "--directory", str(occupied))
    assert response.returncode != 0
    assert marker.read_text() == "original content"
    assert list(occupied.iterdir()) == [marker]


def test_installed_entrypoint_uses_error_wrapper():
    from importlib.metadata import distribution

    entry = next(e for e in distribution("intelligent-diagnosis-system").entry_points if e.name == "diag")
    assert entry.value == "intelligent_diagnosis.cli:main"
