"""发布前检查Git可见文件、暂存区与历史；仅输出文件名及类别，不输出匹配内容。

这是项目特定的最小防泄漏检查，不是通用秘密检测器，也不替代人工内容审核。
工作目录中Git忽略的模型/运行数据不读取；但若它们被强行纳入索引则阻断。
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_PARTS = {".venv", "runtime", "models", "__pycache__", ".claude", ".workflow"}
BLOCKED_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".pickle", ".pkl", ".joblib", ".onnx", ".gguf", ".safetensors", ".log"}
PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "cloud_token": re.compile(rb"\bsk-[A-Za-z0-9_-]{24,}"),
    "home_absolute_path": re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private_network_url": re.compile(rb"https?://(?:127\.0\.0\.[2-9]|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)(?=[:/])"),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def check(path: str, content: bytes) -> list[str]:
    parsed = Path(path)
    problems = []
    if BLOCKED_PARTS.intersection(parsed.parts) or parsed.suffix in BLOCKED_SUFFIXES:
        problems.append("blocked_runtime_or_model")
    if parsed.name.startswith(".env") and parsed.name != ".env.example":
        problems.append("environment_file")
    if len(content) > 2_000_000:
        problems.append("large_file_requires_review")
    # The checker necessarily includes the names it detects; its source is manually reviewed.
    if path != "scripts/check_release.py":
        for name, pattern in PATTERNS.items():
            if pattern.search(content):
                problems.append(name)
    return problems


def main() -> None:
    failures = []
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    visible = sorted(set(filter(None, names)))
    for name in visible:
        path = ROOT / name
        if path.is_symlink():
            failures.append(("worktree", name, "symlink"))
        elif path.is_file():
            failures.extend(("worktree", name, item) for item in check(name, path.read_bytes()))
    staged = git("ls-files", "--stage", "-z").decode().split("\0")
    for item in filter(None, staged):
        header, name = item.split("\t", 1)
        mode, oid, _ = header.split()
        if mode in {"120000", "160000"}:
            failures.append(("index", name, "symlink_or_submodule"))
        content = git("cat-file", "blob", oid)
        failures.extend(("index", name, category) for category in check(name, content))
    history = git("rev-list", "--objects", "--all").decode().splitlines()
    blobs_checked = 0
    for item in history:
        parts = item.split(" ", 1)
        if len(parts) != 2:
            continue
        oid, name = parts
        if git("cat-file", "-t", oid).strip() != b"blob":
            continue
        blobs_checked += 1
        failures.extend(("history", name, category) for category in check(name, git("cat-file", "blob", oid)))
    if failures:
        for source, name, reason in failures:
            print(f"BLOCKED {source} {name}: {reason}")
        raise SystemExit(1)
    print(f"PASS: {len(visible)} Git-visible files, {blobs_checked} historical blobs; manual review still required")


if __name__ == "__main__":
    main()
