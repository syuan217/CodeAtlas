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


def find_symbols(
    conn: sqlite3.Connection, pattern: str, repo_id: int | None = None
) -> list[dict]:
    """符号查找:qualified_name 精确 → name 精确 → 尾段匹配,返回带文件路径的行。"""
    where = "WHERE s.repo_id = ?" if repo_id is not None else ""
    params: list = [repo_id] if repo_id is not None else []
    rows = conn.execute(
        f"SELECT s.*, f.path AS fpath FROM symbols s JOIN files f ON s.file_id=f.id "
        f"{where} AND s.qualified_name = ? AND s.kind != 'module'",
        [*params, pattern],
    ).fetchall()
    if rows:
        return [dict(r) for r in rows]
    rows = conn.execute(
        f"SELECT s.*, f.path AS fpath FROM symbols s JOIN files f ON s.file_id=f.id "
        f"{where} AND s.name = ? AND s.kind != 'module' ORDER BY s.qualified_name",
        [*params, pattern],
    ).fetchall()
    if rows:
        return [dict(r) for r in rows]
    tail = pattern.rsplit(".", 1)[-1].rsplit("#", 1)[-1]
    if tail == pattern:
        return []
    rows = conn.execute(
        f"SELECT s.*, f.path AS fpath FROM symbols s JOIN files f ON s.file_id=f.id "
        f"{where} AND (s.name = ? OR s.qualified_name LIKE ?) AND s.kind != 'module' "
        f"ORDER BY s.qualified_name LIMIT 50",
        [*params, tail, f"%{pattern}%"],
    ).fetchall()
    return [dict(r) for r in rows]


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
    where = "WHERE e.repo_id = ?" if repo_id is not None else ""
    params = [repo_id] if repo_id is not None else []
    rows = conn.execute(
        f"SELECT e.resolution, COUNT(*) AS c FROM edges e "
        f"{where} AND e.kind='CALLS' GROUP BY e.resolution",
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
