"""db/models.py:schema 完整性、WAL、幂等、级联删除。"""

from codeatlas import config as config_mod
from codeatlas.db.models import SCHEMA_VERSION, connect, init_db

EXPECTED_TABLES = {
    "meta", "repos", "files", "symbols", "edges", "chunks",
    "vector_refs", "embed_cache",
    "ddl_tables", "ddl_columns", "ddl_indexes", "query_column_map",
    "audit_findings", "usage_log",
}


def test_init_creates_all_tables(db):
    names = {
        r["name"] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert EXPECTED_TABLES <= names, EXPECTED_TABLES - names


def test_fts5_virtual_table_exists(db):
    row = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    assert row is not None
    # 可写入性由 M1 的显式同步负责;这里只验证 FTS5 模块可用
    db.execute("INSERT INTO chunks_fts(rowid, content) VALUES(1, 'hello world')")
    hits = db.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'hello'"
    ).fetchall()
    assert [h["rowid"] for h in hits] == [1]


def test_wal_mode(db):
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_init_idempotent(db):
    init_db(db)  # 二次建库不抛错
    init_db(db)
    version = db.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert version == str(SCHEMA_VERSION)


def test_foreign_keys_on(db):
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_cascade_delete_repo_to_symbols(db):
    db.execute(
        "INSERT INTO repos(name, path, languages) VALUES('r', '/tmp/r', '[]')"
    )
    repo_id = db.execute("SELECT id FROM repos WHERE name='r'").fetchone()["id"]
    db.execute(
        "INSERT INTO files(repo_id, path, hash) VALUES(?, 'a.java', 'h')", (repo_id,)
    )
    file_id = db.execute("SELECT id FROM files").fetchone()["id"]
    db.execute(
        "INSERT INTO symbols(repo_id, file_id, kind, name, qualified_name, line_start, line_end) "
        "VALUES(?, ?, 'method', 'foo', 'r#foo', 1, 2)",
        (repo_id, file_id),
    )
    sym_id = db.execute("SELECT id FROM symbols").fetchone()["id"]
    db.execute(
        "INSERT INTO edges(repo_id, src_id, dst_id, kind) VALUES(?, ?, ?, 'CONTAINS')",
        (repo_id, file_id, sym_id),
    )
    db.execute(
        "INSERT INTO chunks(repo_id, file_id, symbol_id, kind, content, content_hash) "
        "VALUES(?, ?, ?, 'code', 'x', 'hx')",
        (repo_id, file_id, sym_id),
    )
    db.commit()

    db.execute("DELETE FROM repos WHERE id=?", (repo_id,))
    db.commit()
    for table in ("files", "symbols", "edges", "chunks"):
        assert db.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"] == 0


def test_edges_unique_dedup(db):
    """(repo,src,dst,kind,line,col) 幂等去重:重索引安全(PLAN §7)。

    注意 SQLite UNIQUE 语义:含 NULL 的组合不冲突,所以去重靠非 NULL 的调用点坐标
    (CALLS 边的 line/col 总有值);CONTAINS/IMPORTS 边的幂等由写入方保证。
    """
    db.execute("INSERT INTO repos(name, path, languages) VALUES('r', '/tmp/r', '[]')")
    repo_id = db.execute("SELECT id FROM repos WHERE name='r'").fetchone()["id"]
    db.execute("INSERT INTO files(repo_id, path) VALUES(?, 'A.java')", (repo_id,))
    file_id = db.execute("SELECT id FROM files").fetchone()["id"]
    for name in ("caller", "callee"):
        db.execute(
            "INSERT INTO symbols(repo_id, file_id, kind, name, qualified_name) "
            "VALUES(?, ?, 'function', ?, ?)",
            (repo_id, file_id, name, f"r#{name}"),
        )
    src = db.execute("SELECT id FROM symbols WHERE name='caller'").fetchone()["id"]
    dst = db.execute("SELECT id FROM symbols WHERE name='callee'").fetchone()["id"]

    rows = [
        (repo_id, src, dst, "CALLS", 10, 5),
        (repo_id, src, dst, "CALLS", 10, 5),   # 完全相同 → 去重
        (repo_id, src, dst, "CALLS", 11, 5),   # 不同调用点 → 保留
        (repo_id, src, dst, "CALLS", 10, 6),   # 不同列 → 保留
    ]
    db.executemany(
        "INSERT OR IGNORE INTO edges(repo_id, src_id, dst_id, kind, line, col) "
        "VALUES(?,?,?,?,?,?)",
        rows,
    )
    db.commit()
    assert db.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"] == 3


def test_connect_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "kb.sqlite"
    conn = connect(target)
    try:
        init_db(conn)
        assert target.exists()
    finally:
        conn.close()


def test_connect_honors_monkeypatched_default(tmp_path, monkeypatch):
    """cli 层依赖 connect() 无参调用走 config.DB_PATH,测试可整体重定向。"""
    monkeypatch.setattr(config_mod, "DB_PATH", tmp_path / "redirected.sqlite")
    conn = connect()
    try:
        init_db(conn)
        assert (tmp_path / "redirected.sqlite").exists()
    finally:
        conn.close()
