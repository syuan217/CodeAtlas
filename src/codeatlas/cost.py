"""费用统计:usage_log 写入与汇总查询。"""

from __future__ import annotations

import sqlite3


def log_usage(
    conn: sqlite3.Connection,
    *,
    stage: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    repo_id: int | None = None,
) -> None:
    """写一条费用流水。cost=0 且单价表查不到该模型时表示 unknown(PLAN §9.9)。"""
    conn.execute(
        "INSERT INTO usage_log(stage, repo_id, model, prompt_tokens, completion_tokens, cost) "
        "VALUES(?,?,?,?,?,?)",
        (stage, repo_id, model, prompt_tokens, completion_tokens, cost),
    )
    conn.commit()


def cost_summary(conn: sqlite3.Connection) -> list[dict]:
    """按 stage 汇总:调用数、tokens、费用。"""
    rows = conn.execute(
        "SELECT stage, COUNT(*) AS calls, "
        "COALESCE(SUM(prompt_tokens),0) AS prompt_tokens, "
        "COALESCE(SUM(completion_tokens),0) AS completion_tokens, "
        "COALESCE(SUM(cost),0) AS cost "
        "FROM usage_log GROUP BY stage ORDER BY cost DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def model_summary(conn: sqlite3.Connection) -> list[dict]:
    """按模型汇总(费用排查用)。"""
    rows = conn.execute(
        "SELECT model, COUNT(*) AS calls, "
        "COALESCE(SUM(prompt_tokens),0) AS prompt_tokens, "
        "COALESCE(SUM(completion_tokens),0) AS completion_tokens, "
        "COALESCE(SUM(cost),0) AS cost "
        "FROM usage_log GROUP BY model ORDER BY cost DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def total_cost(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(SUM(cost),0) AS c FROM usage_log").fetchone()
    return float(row["c"])
