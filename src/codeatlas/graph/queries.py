"""确定性结构查询(PLAN §9.5):纯 SQL/图遍历,LLM 不参与。

go-to-definition / find-callers 这类结构问题让 LLM 生成查询是错误的形状
(code-graph-rag 结论)——这里全部走固定算法。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class CallEdge:
    src_qname: str
    dst_qname: str
    resolution: str | None
    line: int | None
    col: int | None
    src_file: str | None
    dst_file: str | None


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def find_symbols(
    conn: sqlite3.Connection, pattern: str, repo_id: int | None = None
) -> list[dict]:
    """符号查找,逐级放宽、首个有结果的层级即返回:

    1. qualified_name 精确;2. name 精确;
    3. 限定名后缀('Order#name' 命中 com.x.Order#name 及其重载 @行号);
    4. 兜底:尾段同名或限定名包含。
    返回行带 fpath(文件)与 repo(仓库名)。
    """
    base = (
        "SELECT s.*, f.path AS fpath, r.name AS repo FROM symbols s "
        "JOIN files f ON s.file_id=f.id JOIN repos r ON s.repo_id=r.id "
        "WHERE s.kind != 'module'"
        + (" AND s.repo_id = ?" if repo_id is not None else "")
    )
    scope: list = [repo_id] if repo_id is not None else []

    def q(cond: str, params: list, tail: str = " ORDER BY s.qualified_name") -> list[dict]:
        return [dict(r) for r in conn.execute(f"{base} AND {cond}{tail}", [*scope, *params])]

    rows = q("s.qualified_name = ?", [pattern])
    if rows:
        return rows
    rows = q("s.name = ?", [pattern])
    if rows:
        return rows
    tail = pattern.rsplit(".", 1)[-1].rsplit("#", 1)[-1]
    if tail == pattern:
        return []
    esc = _like_escape(pattern)
    # LIKE 对 ASCII 不区分大小写:SQL 粗筛 + Python 按段边界('.' 或 TS 模块路径 '/')精确复核
    rows = [
        r for r in q("s.qualified_name LIKE ? ESCAPE '\\'", [f"%{esc}%"])
        if any(
            r["qualified_name"].endswith(sep + pattern) or f"{sep}{pattern}@" in r["qualified_name"]
            for sep in (".", "/")
        )
    ]
    if rows:
        return rows
    return q(
        "(s.name = ? OR s.qualified_name LIKE ? ESCAPE '\\')",
        [tail, f"%{esc}%"],
        " ORDER BY s.qualified_name LIMIT 50",
    )


def _call_rows(conn, where: str, params: list) -> list[CallEdge]:
    rows = conn.execute(
        "SELECT a.qualified_name AS sq, b.qualified_name AS dq, e.resolution, "
        "       e.line, e.col, fa.path AS sf, fb.path AS df "
        "FROM edges e "
        "JOIN symbols a ON e.src_id = a.id JOIN files fa ON a.file_id = fa.id "
        "JOIN symbols b ON e.dst_id = b.id JOIN files fb ON b.file_id = fb.id "
        f"WHERE e.kind = 'CALLS' AND {where} "
        "ORDER BY e.line",
        params,
    ).fetchall()
    return [
        CallEdge(
            src_qname=r["sq"], dst_qname=r["dq"], resolution=r["resolution"],
            line=r["line"], col=r["col"], src_file=r["sf"], dst_file=r["df"],
        )
        for r in rows
    ]


def callers_of(conn: sqlite3.Connection, sym: dict) -> list[CallEdge]:
    """谁调用它(直接;exact 优先展示由 ORDER 保证不了,按解析分组由 CLI 展示)。"""
    return _call_rows(conn, "e.dst_id = ?", [sym["id"]])


def callees_of(conn: sqlite3.Connection, sym: dict) -> list[CallEdge]:
    return _call_rows(conn, "e.src_id = ?", [sym["id"]])


def impact_of(
    conn: sqlite3.Connection, sym: dict, max_depth: int = 10
) -> list[dict]:
    """影响面:沿 CALLS 入边向上传播到不动点(改它会破坏哪些方法/文件)。

    返回 [{depth, qname, file, via}],深度 1 = 直接调用方。
    """
    results: dict[int, dict] = {}
    frontier = [(sym["id"], 0)]
    seen = {sym["id"]}
    while frontier:
        nid, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        rows = conn.execute(
            "SELECT a.id AS aid, a.qualified_name AS q, f.path AS p "
            "FROM edges e JOIN symbols a ON e.src_id = a.id "
            "JOIN files f ON a.file_id = f.id "
            "WHERE e.kind = 'CALLS' AND e.dst_id = ?",
            (nid,),
        ).fetchall()
        for r in rows:
            if r["aid"] in seen:
                continue
            seen.add(r["aid"])
            entry = {
                "depth": depth + 1,
                "symbol_id": r["aid"],
                "qname": r["q"],
                "file": r["p"],
            }
            results[r["aid"]] = entry
            frontier.append((r["aid"], depth + 1))
    return sorted(results.values(), key=lambda x: (x["depth"], x["qname"]))


def call_edges_stats(conn: sqlite3.Connection, repo_id: int | None = None) -> dict:
    where = " AND e.repo_id = ?" if repo_id is not None else ""
    params = [repo_id] if repo_id is not None else []
    rows = conn.execute(
        f"SELECT e.resolution, COUNT(*) AS c FROM edges e "
        f"WHERE e.kind='CALLS'{where} GROUP BY e.resolution",
        params,
    ).fetchall()
    return {r["resolution"] or "unknown": r["c"] for r in rows}


def sample_call_edges(
    conn: sqlite3.Connection, n: int = 30, resolution: str | None = None,
    repo_id: int | None = None,
) -> list[CallEdge]:
    """验收抽样:随机取调用边供人工核对。"""
    where = ["e.kind = 'CALLS'"]
    params: list = []
    if resolution:
        where.append("e.resolution = ?")
        params.append(resolution)
    if repo_id is not None:
        where.append("e.repo_id = ?")
        params.append(repo_id)
    rows = conn.execute(
        "SELECT a.qualified_name AS sq, b.qualified_name AS dq, e.resolution, "
        "       e.line, e.col, fa.path AS sf, fb.path AS df "
        "FROM edges e "
        "JOIN symbols a ON e.src_id = a.id JOIN files fa ON a.file_id = fa.id "
        "JOIN symbols b ON e.dst_id = b.id JOIN files fb ON b.file_id = fb.id "
        f"WHERE {' AND '.join(where)} ORDER BY RANDOM() LIMIT ?",
        [*params, n],
    ).fetchall()
    return [
        CallEdge(
            src_qname=r["sq"], dst_qname=r["dq"], resolution=r["resolution"],
            line=r["line"], col=r["col"], src_file=r["sf"], dst_file=r["df"],
        )
        for r in rows
    ]
