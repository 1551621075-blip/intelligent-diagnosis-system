"""事务持久化、内容哈希和审计；运行数据库永不进入Git。"""

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import Actor, DomainError


def canonical(value: Any) -> str:
    """稳定序列化用于幂等比较；不接受NaN。"""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ClosingConnection(sqlite3.Connection):
    """SQLite默认上下文不关闭连接，这里在提交/回滚后显式关闭。"""

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class Store:
    """SQLite事务是本地权威记录，不依赖进程内字典保证一致性。"""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "diagnosis.sqlite3"
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS users(
                    actor_id TEXT PRIMARY KEY, role TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE);
                CREATE TABLE IF NOT EXISTS diagnoses(
                    diagnosis_id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, UNIQUE(owner, request_id));
                CREATE TABLE IF NOT EXISTS telemetry(
                    owner TEXT NOT NULL, event_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL, PRIMARY KEY(owner,event_id));
                CREATE TABLE IF NOT EXISTS device_events(
                    owner TEXT NOT NULL, device_id TEXT NOT NULL, domain TEXT NOT NULL,
                    latest_ms INTEGER NOT NULL, last_trigger_ms INTEGER,
                    PRIMARY KEY(owner,device_id,domain));
                CREATE TABLE IF NOT EXISTS telemetry_buffers(
                    owner TEXT NOT NULL, device_id TEXT NOT NULL, domain TEXT NOT NULL,
                    samples TEXT NOT NULL, PRIMARY KEY(owner,device_id,domain));
                CREATE TABLE IF NOT EXISTS model_approvals(
                    version TEXT PRIMARY KEY, approved_at TEXT NOT NULL, actor TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
                    actor TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback(
                    feedback_id TEXT PRIMARY KEY, diagnosis_id TEXT NOT NULL,
                    actor TEXT NOT NULL, payload_hash TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS services(
                    service_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS assignments(
                    diagnosis_id TEXT PRIMARY KEY, action_id TEXT UNIQUE NOT NULL,
                    actor TEXT NOT NULL, service_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS models(
                    version TEXT PRIMARY KEY, metadata TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS actions(
                    action_id TEXT PRIMARY KEY, actor TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS knowledge(
                    version TEXT PRIMARY KEY, snapshot TEXT NOT NULL,
                    status TEXT NOT NULL, proposer TEXT NOT NULL, approver TEXT,
                    created_at TEXT NOT NULL);
                """
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None, factory=ClosingConnection)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    @contextmanager
    def transaction(self):
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def audit(db: sqlite3.Connection, actor: Actor, event: str, payload: dict) -> None:
        db.execute(
            "INSERT INTO audit(at,actor,event,payload) VALUES(?,?,?,?)",
            (now_iso(), actor.actor_id, event, canonical(payload)),
        )

    def provision_user(self, actor_id: str, role: str) -> str:
        """创建一次性返回的本地令牌；不覆盖已有身份或读取已有秘密。"""
        if role not in {"operator", "expert", "admin"}:
            raise DomainError("invalid_role", "角色无效")
        token = secrets.token_urlsafe(32)
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM users WHERE actor_id=?", (actor_id,)).fetchone():
                raise DomainError("user_exists", "身份已存在，不覆盖令牌", 409)
            db.execute(
                "INSERT INTO users VALUES(?,?,?)",
                (actor_id, role, hashlib.sha256(token.encode()).hexdigest()),
            )
        return token

    def authenticate(self, token: str) -> Actor:
        with self.connect() as db:
            row = db.execute(
                "SELECT actor_id,role FROM users WHERE token_hash=?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            ).fetchone()
        if not row:
            raise DomainError("unauthenticated", "需要有效本地令牌", 401)
        return Actor(actor_id=row["actor_id"], role=row["role"])

    def put_artifact(self, payload: dict) -> str:
        content = canonical(payload).encode()
        key = hashlib.sha256(content).hexdigest()
        target = self.artifacts / f"{key}.json"
        try:
            with target.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            if target.read_bytes() != content:
                raise DomainError("artifact_conflict", "制品哈希冲突", 409) from None
        return key

    def get_artifact(self, key: str) -> dict:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise DomainError("invalid_artifact", "制品引用无效")
        path = self.artifacts / f"{key}.json"
        if not path.is_file():
            raise DomainError("artifact_missing", "制品不存在", 404)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != key:
            raise DomainError("artifact_corrupt", "制品内容校验失败", 409)
        return json.loads(content)
