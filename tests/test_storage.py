"""本地事务、身份与内容寻址制品的独立测试。"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from intelligent_diagnosis.schemas import DomainError
from intelligent_diagnosis.storage import Store, digest


def test_context_closes_connection(tmp_path):
    store = Store(tmp_path)
    with store.connect() as db:
        assert db.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute("SELECT 1")


def test_exception_rolls_back_transaction(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(RuntimeError), store.transaction() as db:
        db.execute("INSERT INTO settings VALUES('temporary','value')")
        raise RuntimeError("injected rollback test")
    with store.connect() as db:
        assert db.execute("SELECT 1 FROM settings").fetchone() is None


def test_tokens_hashed_and_existing_user_never_overwritten(tmp_path):
    store = Store(tmp_path)
    token = store.provision_user("tester", "expert")
    assert store.authenticate(token).role == "expert"
    with store.connect() as db:
        stored = db.execute("SELECT token_hash FROM users").fetchone()[0]
        assert stored != token
        assert len(stored) == 64
    with pytest.raises(DomainError, match="身份已存在"):
        store.provision_user("tester", "admin")
    with pytest.raises(DomainError) as error:
        store.authenticate("invalid")
    assert error.value.status == 401


def test_content_artifact_integrity(tmp_path):
    store = Store(tmp_path)
    payload = {"synthetic": True, "value": 3}
    key = store.put_artifact(payload)
    assert key == store.put_artifact(payload) == digest(payload)
    assert store.get_artifact(key) == payload
    (store.artifacts / f"{key}.json").write_text("{}")
    with pytest.raises(DomainError, match="校验失败"):
        store.get_artifact(key)
    with pytest.raises(DomainError):
        store.get_artifact("../../outside")


def test_concurrent_transactions_preserve_all_updates(tmp_path):
    store = Store(tmp_path)
    with store.transaction() as db:
        db.execute("INSERT INTO settings VALUES('counter','0')")

    def increment(_):
        with store.transaction() as db:
            old = int(db.execute("SELECT value FROM settings WHERE key='counter'").fetchone()[0])
            db.execute("UPDATE settings SET value=? WHERE key='counter'", (str(old + 1),))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(increment, range(20)))
    with store.connect() as db:
        assert db.execute("SELECT value FROM settings WHERE key='counter'").fetchone()[0] == "20"
