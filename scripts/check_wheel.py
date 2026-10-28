"""从wheel在全新虚拟环境安装，并在源码目录之外验证CLI、网页和真实API。

用法：uv build && uv run python scripts/check_wheel.py dist/<wheel>.whl
默认离线；缺缓存明确失败。显式--allow-download才允许从公开PyPI下载依赖。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SMOKE = r'''
import pathlib
from fastapi.testclient import TestClient
import intelligent_diagnosis
from intelligent_diagnosis.api import create_app
from intelligent_diagnosis.synthetic import make_request

assert pathlib.Path(intelligent_diagnosis.__file__).resolve().is_relative_to(pathlib.Path("venv").resolve())
app = create_app(pathlib.Path("state"), pathlib.Path("models"))
token = app.state.store.provision_user("wheel-operator", "operator")
with TestClient(app) as client:
    assert client.get("/health").status_code == 200
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="close-form"' in page.text
    assert 'id="rollback-form"' in page.text
    for path in ("/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200
    assert client.get("/api/diagnoses").status_code == 401
    response = client.post("/api/diagnoses", headers={"Authorization": "Bearer " + token},
                           json=make_request("battery_depleted").model_dump(mode="json"))
    assert response.status_code == 200
    assert response.json()["outcome"] == "unknown"
    assert response.json()["prediction"] is None
    assert response.json()["attribution"] is None
print("PASS: installed CLI/assets/authenticated API; missing model correctly remains unknown")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for required in ("templates/index.html", "static/app.js", "static/style.css"):
            assert "intelligent_diagnosis/" + required in names, required
        for name in names:
            path = Path(name)
            assert not {"runtime", "models", ".env", ".venv"}.intersection(path.parts), name
    env = {k: v for k, v in os.environ.items()
           if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} and not k.startswith("DIAG_")}
    with tempfile.TemporaryDirectory(prefix="diagnosis-wheel-") as tmp:
        root = Path(tmp)
        python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([uv, "venv", "--python", sys.executable, str(root / "venv")], check=True, env=env)
        network = [] if args.allow_download else ["--offline"]
        subprocess.run([uv, "pip", "install", *network, "--python", str(python),
                        "--default-index", "https://pypi.org/simple", str(wheel), "httpx"],
                       check=True, env=env)
        subprocess.run([str(python), "-I", "-m", "intelligent_diagnosis.cli", "--help"],
                       cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([str(python), "-I", "-c", SMOKE], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
