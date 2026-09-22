"""上下文组装(PLAN §9.5):按文件分组、[lines A-B] 真实行号标注、token 预算。

预算 = LLM_CONTEXT_WINDOW - 2048(输出余量);超预算时从融合分最低项开始剔除。
"""

from __future__ import annotations

import sqlite3

from codeatlas.ingest.chunker import estimate_tokens
from codeatlas.retrieve.search import Candidate


def build_context(
    conn: sqlite3.Connection,
    candidates: list[Candidate],
    *,
    top_n: int,
    budget_tokens: int,
) -> tuple[str, list[Candidate]]:
    """组装上下文文本;返回 (context, 实际采用的候选列表)。"""
    picked = candidates[:top_n]
    if not picked:
        return "", []

    rows = {}
    ph = ",".join("?" * len(picked))
    qs = conn.execute(
        f"SELECT c.id AS cid, c.title, c.content, c.line_start, c.line_end, "
        f"       f.path AS fpath "
        f"FROM chunks c LEFT JOIN files f ON c.file_id = f.id "
        f"WHERE c.id IN ({ph})",
        [c.chunk_id for c in picked],
    ).fetchall()
    for r in qs:
        rows[r["cid"]] = r

    # 按文件分组;文件顺序 = 组内最高融合分;文件内按行号排序
    groups: dict[str, list] = {}
    for cand in picked:  # picked 已按分数降序
        r = rows.get(cand.chunk_id)
        if r is None:
            continue
        key = r["fpath"] or f"chunk#{cand.chunk_id}"
        groups.setdefault(key, []).append((cand, r))

    parts: list[str] = []
    used: list[Candidate] = []
    budget = budget_tokens
    for fpath in groups:  # dict 保插入序 = 组内最高分序
        members = sorted(groups[fpath], key=lambda m: (m[1]["line_start"] or 0))
        file_lines: list[str] = [f"### 文件:{fpath}"]
        file_tokens = 0
        file_added = False
        for cand, r in members:
            snippet = "\n".join(
                [
                    f"[lines {r['line_start']}-{r['line_end']}]",
                    r["content"],
                ]
            )
            t = estimate_tokens(snippet)
            if file_tokens + t > budget:
                break  # 预算耗尽,停止装填
            file_lines.append(snippet)
            file_tokens += t
            budget -= t
            used.append(cand)
            file_added = True
        if file_added:
            parts.append("\n\n".join(file_lines))
        if budget <= 0:
            break

    return "\n\n".join(parts), used
