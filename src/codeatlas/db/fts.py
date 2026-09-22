"""FTS5 显式同步(PLAN §7:chunks ↔ chunks_fts,不用触发器)。

外部内容表(content='chunks')语义:写入用普通 INSERT(rowid=chunk id);
删除必须用 'delete' 命令并提供**与写入时完全一致**的 content——
所以删除前先从 chunks 行取 content,再发 delete。
"""

from __future__ import annotations

import sqlite3


def fts_insert(conn: sqlite3.Connection, chunk_id: int, content: str) -> None:
    conn.execute(
        "INSERT INTO chunks_fts(rowid, content) VALUES(?, ?)", (chunk_id, content)
    )


def fts_delete(conn: sqlite3.Connection, chunk_id: int, content: str) -> None:
    """外部内容表删除:content 必须与当初写入的完全一致。"""
    conn.execute(
        "INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', ?, ?)",
        (chunk_id, content),
    )


def fts_delete_file(conn: sqlite3.Connection, file_id: int) -> None:
    """删除一个文件全部 chunk 的 FTS 行(须在 chunks 行删除之前调用)。"""
    rows = conn.execute(
        "SELECT id, content FROM chunks WHERE file_id=?", (file_id,)
    ).fetchall()
    for r in rows:
        fts_delete(conn, r["id"], r["content"])


def fts_delete_repo(conn: sqlite3.Connection, repo_id: int) -> None:
    rows = conn.execute(
        "SELECT c.id, c.content FROM chunks c WHERE c.repo_id=?", (repo_id,)
    ).fetchall()
    for r in rows:
        fts_delete(conn, r["id"], r["content"])


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """MATCH 查询,返回 chunk 行(M2 检索用;此处供测试与调试)。"""
    rows = conn.execute(
        "SELECT c.* FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    return [dict(r) for r in rows]
