"""非代码 chunk(wiki / report / doc)的整批替换:三处入库共用。

chunks 行、FTS 行、Lance 向量三者必须同进同出:只删 chunks 不删向量会留孤儿
(SQLite 复用 id 后指向别的 chunk);只写 chunks 不写 FTS 则关键词检索不到,
且下次替换时对未入索引的行发 FTS 'delete' 会写坏索引。
"""

from __future__ import annotations

import sqlite3

from codeatlas.db.fts import fts_delete, fts_insert


def purge_chunks(conn: sqlite3.Connection, lance, *, repo_id: int, kind: str) -> int:
    """删除某仓库某类全部 chunk(FTS 行 → 向量 → chunks 行,vector_refs 级联)。"""
    old = conn.execute(
        "SELECT id, content FROM chunks WHERE repo_id=? AND kind=?", (repo_id, kind)
    ).fetchall()
    for r in old:
        fts_delete(conn, r["id"], r["content"])
    if lance is not None and old:
        lance.delete_by_chunk_ids([r["id"] for r in old])
    conn.execute("DELETE FROM chunks WHERE repo_id=? AND kind=?", (repo_id, kind))
    return len(old)


def insert_text_chunk(
    conn: sqlite3.Connection, *, repo_id: int, kind: str, title: str, chunk
) -> int:
    """写一条无源文件归属的 chunk(chunker.ChunkOut)并同步 FTS,返回 chunk id。"""
    cur = conn.execute(
        "INSERT INTO chunks(repo_id, kind, title, content, content_hash, "
        "line_start, line_end) VALUES(?,?,?,?,?,?,?)",
        (repo_id, kind, title, chunk.content, chunk.content_hash,
         chunk.line_start, chunk.line_end),
    )
    fts_insert(conn, cur.lastrowid, chunk.content)
    return cur.lastrowid


async def embed_chunks(
    conn: sqlite3.Connection, lance, embedder, pending: list[tuple[int, str]],
    *, repo_id: int, kind: str, stage: str,
) -> int:
    """pending = [(chunk_id, content)] → embedding → Lance + vector_refs;返回写入向量数。"""
    if embedder is None or not pending:
        return 0
    vectors = await embedder.embed(
        [c for _, c in pending], stage=stage, repo_id=repo_id
    )
    rows = [
        (cid, repo_id, kind, blob)
        for (cid, _), blob in zip(pending, vectors, strict=True)
    ]
    for lance_id, chunk_id in lance.add_vectors(rows):
        conn.execute(
            "INSERT INTO vector_refs(chunk_id, lance_id, model) VALUES(?,?,?)",
            (chunk_id, lance_id, embedder.s.embed_model),
        )
    conn.commit()
    return len(rows)
