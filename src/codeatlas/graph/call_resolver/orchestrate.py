"""CALLS 边编排(Pass2):对一批文件(含依赖闭包)解析调用点并写入边。

- 首索引:indexer 在 Pass1 已持有各文件 fs/调用点 → 直接给 CallFile;
- 依赖闭包重解析:文件自身未变更但其 IMPORTS 目标变了 → light_call_file
  重读文件重建 fs 与调用点(符号/块不动,只重算边);
- 每个文件写边前先清其符号的全部旧 CALLS 出边(闭包重算的替换语义);
- 幂等:UNIQUE(repo_id,src,dst,kind,line,col) 对 CALLS 生效(line/col 非 NULL)。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from codeatlas.config import RepoCfg
from codeatlas.graph.call_resolver import generic as generic_resolver
from codeatlas.graph.call_resolver import java as java_resolver
from codeatlas.graph.call_resolver import ts as ts_resolver
from codeatlas.graph.call_resolver.base import (
    CascadeResult,
    ResolveContext,
    build_context,
)
from codeatlas.ingest.symbols import (
    FileSymbols,
    detect_language,
    extract_call_sites,
    extract_symbols,
)


@dataclass
class CallFile:
    """一个待解析调用边的文件。"""

    fs: FileSymbols
    sym_ids: dict[int, int]  # FileSymbols.symbols 局部索引 → 库 symbol id
    module_id: int
    sites: list = field(default_factory=list)


@dataclass
class CallStats:
    sites: int = 0
    exact: int = 0
    heuristic: int = 0
    dropped: int = 0
    edges_written: int = 0


def _resolver_for(lang: str):
    if lang == "java":
        return java_resolver
    if lang in ("javascript", "typescript", "tsx"):
        return ts_resolver
    return generic_resolver


def _qname_to_id(conn: sqlite3.Connection, rel: str) -> dict[str, int] | None:
    rows = conn.execute(
        "SELECT s.id, s.qualified_name FROM symbols s "
        "JOIN files f ON s.file_id = f.id WHERE f.path = ?",
        (rel,),
    ).fetchall()
    if not rows:
        return None
    return {r["qualified_name"]: r["id"] for r in rows}


def _module_id(conn: sqlite3.Connection, rel: str) -> int | None:
    row = conn.execute(
        "SELECT s.id FROM symbols s JOIN files f ON s.file_id=f.id "
        "WHERE f.path=? AND s.kind='module'",
        (rel,),
    ).fetchone()
    return row["id"] if row else None


def light_call_file(conn: sqlite3.Connection, repo: RepoCfg, rel: str) -> CallFile | None:
    """依赖闭包文件重解析:重读文件重建 fs 与调用点,库 symbol id 按 qname 对齐。"""
    try:
        raw = (repo.path / rel).read_bytes()
    except OSError:
        return None
    lang = detect_language(rel)
    if lang is None:
        return None
    fs = extract_symbols(rel, lang, raw)
    qmap = _qname_to_id(conn, rel)
    module_id = _module_id(conn, rel)
    if qmap is None or module_id is None:
        return None
    sym_ids = {
        i: qmap[s.qualified_name]
        for i, s in enumerate(fs.symbols)
        if s.qualified_name in qmap
    }
    return CallFile(
        fs=fs, sym_ids=sym_ids, module_id=module_id,
        sites=extract_call_sites(rel, lang, raw, fs),
    )


def resolve_and_write_calls(
    conn: sqlite3.Connection,
    repo_id: int,
    files: list[CallFile],
    ctx: ResolveContext | None = None,
) -> CallStats:
    ctx = ctx or build_context(conn, repo_id)
    stats = CallStats()
    for cf in files:
        resolver = _resolver_for(cf.fs.lang)
        # 替换语义:清该文件符号的全部旧 CALLS 出边后重写
        conn.execute(
            "DELETE FROM edges WHERE kind='CALLS' AND src_id IN "
            "(SELECT s.id FROM symbols s JOIN files f ON s.file_id=f.id WHERE f.path=?)",
            (cf.fs.rel,),
        )
        for site in cf.sites:
            stats.sites += 1
            if site.caller_local is not None:
                caller_id = cf.sym_ids.get(site.caller_local)
            else:
                caller_id = cf.module_id
            if caller_id is None:
                stats.dropped += 1
                continue
            caller = ctx.by_id.get(caller_id)
            result: CascadeResult = resolver.resolve_site(site, caller, cf.fs, ctx)
            if not result.resolved:
                stats.dropped += 1
                continue
            if result.resolution == "exact":
                stats.exact += 1
            else:
                stats.heuristic += 1
            cur = conn.execute(
                "INSERT OR IGNORE INTO edges(repo_id, src_id, dst_id, kind, "
                "resolution, line, col) VALUES(?,?,?,?,?,?,?)",
                (
                    repo_id,
                    caller_id,
                    result.dst.id,
                    "CALLS",
                    result.resolution,
                    site.line,
                    site.col,
                ),
            )
            stats.edges_written += cur.rowcount
    conn.commit()
    return stats
