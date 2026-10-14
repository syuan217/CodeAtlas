"""FTS5 显式同步(PLAN §7:chunks ↔ chunks_fts,不用触发器)。

无内容表(content='')语义:写入用普通 INSERT(rowid=chunk id),索引的是
fts_text() 预处理后的文本(原文仍只存在 chunks);删除必须用 'delete' 命令并
提供**与写入时完全一致**的文本——所以删除前先从 chunks 行取 content,
再按同一预处理生成文本发 delete。
"""

from __future__ import annotations

import re
import sqlite3

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _cjk_bigrams(m: re.Match) -> str:
    s = m.group(0)
    if len(s) == 1:
        return f" {s} "
    return " " + " ".join(s[i:i + 2] for i in range(len(s) - 1)) + " "


def fts_text(content: str) -> str:
    """入索引前的分词增强(unicode61 不切中文、不拆驼峰,两者在本库都很常见):

    - 连续汉字 → 重叠二元组(订单创建 → 订单 单创 创建),与查询侧 bigram 对齐;
    - 驼峰/帕斯卡标识符末尾追加拆分词(createOrder → create order),原词保留。
    必须是确定性函数:'delete' 要按同样的词序重放。
    """
    text = _CJK_RUN.sub(_cjk_bigrams, content)
    extra: list[str] = []
    for tok in dict.fromkeys(_IDENT.findall(content)):  # 去重且保序(确定性)
        parts = _CAMEL.findall(tok)
        if len(parts) > 1:
            extra.extend(parts)
    return text + ("\n" + " ".join(extra) if extra else "")


def fts_insert(conn: sqlite3.Connection, chunk_id: int, content: str) -> None:
    conn.execute(
        "INSERT INTO chunks_fts(rowid, content) VALUES(?, ?)", (chunk_id, fts_text(content))
    )


def fts_delete(conn: sqlite3.Connection, chunk_id: int, content: str) -> None:
    """删除一行索引:content 为 chunks 原文,按 fts_text 重放出与写入时一致的词序。

    对从未写入索引的 rowid 发 'delete' 会写坏倒排表(后续 MATCH 报
    database disk image is malformed),故先查 docsize 确认该行确实入过索引。
    """
    if conn.execute(
        "SELECT 1 FROM chunks_fts_docsize WHERE id=?", (chunk_id,)
    ).fetchone() is None:
        return
    conn.execute(
        "INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', ?, ?)",
        (chunk_id, fts_text(content)),
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


def fts_rebuild(conn: sqlite3.Connection) -> int:
    """按 chunks 全量重建索引(迁移/修复用);返回写入行数。"""
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")  # 无内容表不支持 DELETE
    n = 0
    for r in conn.execute("SELECT id, content FROM chunks"):
        fts_insert(conn, r[0], r[1])
        n += 1
    return n


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """MATCH 查询,返回 chunk 行(M2 检索用;此处供测试与调试)。"""
    rows = conn.execute(
        "SELECT c.* FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    return [dict(r) for r in rows]
