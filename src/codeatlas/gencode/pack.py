"""定向 grounding 打包(PLAN §9.7):按页面 filePaths 直读真实文件。

预算内装填:符号签名优先(类声明/字段/方法签名),入度高的符号带函数体;
超预算按优先级截断。相邻模块接口摘要 = 模块对外符号(facade/接口/公共类)签名。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codeatlas.gencode.segment import Module
from codeatlas.ingest.chunker import estimate_tokens

DEFAULT_BUDGET_TOKENS = 40_000  # 上下文装填预算(模型窗口由调用方扣除)


def _symbol_indegree(conn: sqlite3.Connection) -> dict[int, int]:
    return {
        r["dst_id"]: r["c"]
        for r in conn.execute(
            "SELECT dst_id, COUNT(*) AS c FROM edges WHERE kind='CALLS' "
            "GROUP BY dst_id"
        )
    }


def _module_symbols(conn: sqlite3.Connection, repo_id: int, files: list[str]):
    if not files:
        return []
    ph = ",".join("?" * len(files))
    return [
        dict(r)
        for r in conn.execute(
            f"SELECT s.id, s.file_id, s.kind, s.name, s.qualified_name, "
            f"       s.line_start, s.line_end, s.signature, f.path AS fpath "
            f"FROM symbols s JOIN files f ON s.file_id=f.id "
            f"WHERE s.repo_id=? AND f.path IN ({ph}) AND s.kind!='module'",
            [repo_id, *files],
        )
    ]


def pack_files(
    conn: sqlite3.Connection,
    repo_id: int,
    repo_root: Path,
    file_paths: list[str],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    indegree: dict[int, int] | None = None,
) -> tuple[str, int]:
    """打包成带 [lines A-B] 标注的源码片段文本;返回 (文本, 实际 token 估算)。"""
    indegree = indegree or _symbol_indegree(conn)
    syms_by_file: dict[str, list[dict]] = {}
    for s in _module_symbols(conn, repo_id, file_paths):
        syms_by_file.setdefault(s["fpath"], []).append(s)
    for p in syms_by_file:
        syms_by_file[p].sort(key=lambda s: -indegree.get(s["id"], 0))

    parts: list[str] = []
    used = 0
    for rel in sorted(file_paths):
        path = repo_root / rel
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        header = f"##### FILE: {rel}\n"
        used += estimate_tokens(header)
        parts.append(header)
        budget_left = budget_tokens - used
        if estimate_tokens("\n".join(lines)) <= budget_left:
            body = "\n".join(f"{i+1:>5}| {ln}" for i, ln in enumerate(lines))
            t = estimate_tokens(body)
            parts.append(f"[lines 1-{len(lines)}]\n{body}\n")
            used += t
            continue
        # 超预算:按符号优先级装填(签名段必有;高入度带体)
        emitted: set[int] = set()

        def emit_range(a: int, b: int) -> bool:
            nonlocal used
            seg = "\n".join(f"{i+1:>5}| {lines[i]}" for i in range(a - 1, min(b, len(lines))))
            t = estimate_tokens(seg)
            if used + t > budget_tokens:
                return False
            parts.append(f"[lines {a}-{min(b, len(lines))}]\n{seg}\n")
            used += t
            for i in range(a - 1, min(b, len(lines))):
                emitted.add(i)
            return True

        for s in syms_by_file.get(rel, []):
            a = s["line_start"]
            b = s["line_end"]
            if any(i in emitted for i in range(a - 1, b)):
                continue
            if indegree.get(s["id"], 0) >= 2:
                emit_range(a, b)  # 高入度:完整符号体
            else:
                emit_range(a, a)  # 低入度:仅签名行
        if used >= budget_tokens:
            break
    return "\n".join(parts), used


def interface_summary(
    conn: sqlite3.Connection, repo_id: int, module: Module, limit: int = 40
) -> str:
    """模块对外接口摘要:interface/facade 与 class 的签名行(供相邻页引用)。"""
    syms = _module_symbols(conn, repo_id, module.files)
    picked = [
        s for s in syms if s["kind"] in ("interface", "class") and s["signature"]
    ]
    picked.sort(key=lambda s: (s["kind"] != "interface", s["qualified_name"]))
    lines = [f"模块 {module.id}({len(module.files)} 文件)对外符号:"]
    for s in picked[:limit]:
        lines.append(f"- {s['signature'][:160]}  ({s['fpath']}:{s['line_start']})")
    return "\n".join(lines)
