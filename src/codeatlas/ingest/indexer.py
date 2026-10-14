"""增量索引编排(PLAN §9.4)。

atlas index 流程:读基线 → 变更集(git diff 主路径 / hash+mtime 兜底)→
deleted 删子树(FTS→Lance→SQL 级联)→ added/modified 幂等跳过后删旧重解析 →
符号/CONTAINS/切块/FTS → 全部 flush 后批量 embedding → LanceDB + vector_refs →
IMPORTS Pass2 → 推进 indexed_commit(最后一步,基线永不提前)。

可中断续跑:按批提交(BATCH_FILES),崩溃保留已完成批次;
重跑时 hash 相同且已完整嵌入的文件直接跳过。M3 后依赖闭包(CALLS 重解析)生效。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from codeatlas.config import RepoCfg, Settings, get_settings, hash_bytes
from codeatlas.db.fts import fts_delete_file, fts_insert
from codeatlas.db.lance import LanceStore
from codeatlas.db.models import connect, in_chunks, init_db
from codeatlas.ingest.chunker import chunk_text
from codeatlas.ingest.gitdiff import compute_change_set
from codeatlas.ingest.imports_resolver import ImportRef, write_import_edges
from codeatlas.graph.call_resolver.orchestrate import (
    CallFile,
    light_call_file,
    resolve_and_write_calls,
)
from codeatlas.ingest.symbols import (
    decode_source,
    detect_language,
    extract_call_sites,
    extract_symbols,
)
from codeatlas.ingest.walker import BINARY_SNIFF_BYTES, MAX_FILE_BYTES, sniff_binary
from codeatlas.providers.embedding import EmbeddingProvider

BATCH_FILES = 50
EMBED_SLICE = 1000  # 每片 chunk 数:embed → 落 Lance/vector_refs → 提交


@dataclass
class IndexStats:
    repo: str
    mode: str = "scan"
    base_commit: str | None = None
    head_commit: str | None = None
    added: int = 0
    modified: int = 0
    deleted: int = 0
    skipped_unchanged: int = 0
    skipped_binary: int = 0
    skipped_large: int = 0
    skipped_unknown_ext: int = 0
    parse_failed: int = 0
    symbols: int = 0
    contains_edges: int = 0
    import_edges: int = 0
    chunks: int = 0
    embedded: int = 0
    orphan_vectors_deleted: int = 0
    call_sites: int = 0
    call_edges_exact: int = 0
    call_edges_heuristic: int = 0
    duration_s: float = 0.0

    def summary_line(self) -> str:
        return (
            f"{self.repo}: mode={self.mode} +{self.added} ~{self.modified} "
            f"-{self.deleted} skip(unchanged/binary/large/ext)="
            f"{self.skipped_unchanged}/{self.skipped_binary}/{self.skipped_large}/"
            f"{self.skipped_unknown_ext} failed={self.parse_failed} | "
            f"symbols={self.symbols} edges(C/I)={self.contains_edges}/{self.import_edges} "
            f"chunks={self.chunks} embedded={self.embedded} "
            f"calls(E+H/D)={self.call_edges_exact}+{self.call_edges_heuristic}/"
            f"{self.call_sites - self.call_edges_exact - self.call_edges_heuristic} "
            f"orphans-{self.orphan_vectors_deleted} "
            f"({self.duration_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# 库操作
# ---------------------------------------------------------------------------

def upsert_repo(conn: sqlite3.Connection, repo: RepoCfg) -> int:
    import json

    conn.execute(
        "INSERT INTO repos(name, path, languages) VALUES(?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET path=excluded.path, languages=excluded.languages",
        (repo.name, str(repo.path), json.dumps(repo.languages)),
    )
    conn.commit()
    return conn.execute("SELECT id FROM repos WHERE name=?", (repo.name,)).fetchone()["id"]


def _known_files(conn: sqlite3.Connection, repo_id: int) -> dict[str, tuple[str, float]]:
    rows = conn.execute(
        "SELECT path, hash, mtime FROM files WHERE repo_id=?", (repo_id,)
    ).fetchall()
    return {r["path"]: (r["hash"] or "", r["mtime"] or 0.0) for r in rows}


def _file_row(conn, repo_id: int, rel: str):
    return conn.execute(
        "SELECT * FROM files WHERE repo_id=? AND path=?", (repo_id, rel)
    ).fetchone()


def _file_fully_embedded(conn: sqlite3.Connection, file_id: int) -> bool:
    """该文件的 chunks 是否全部有向量(中断续跑的完整性判据)。"""
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM chunks c WHERE c.file_id=? "
        "AND NOT EXISTS (SELECT 1 FROM vector_refs v WHERE v.chunk_id=c.id)",
        (file_id,),
    ).fetchone()
    return row["c"] == 0


def remove_file(
    conn: sqlite3.Connection, lance: LanceStore, repo_id: int, rel: str
) -> None:
    """删文件子树:FTS(需原 content)→ Lance 向量 → files 行(级联 symbols/edges/chunks)。

    不 commit,由调用方控制事务边界。
    """
    row = _file_row(conn, repo_id, rel)
    if row is None:
        return
    fts_delete_file(conn, row["id"])
    chunk_ids = [
        r["id"] for r in conn.execute(
            "SELECT id FROM chunks WHERE file_id=?", (row["id"],)
        )
    ]
    lance.delete_by_chunk_ids(chunk_ids)
    conn.execute("DELETE FROM files WHERE id=?", (row["id"],))


def _write_symbols(
    conn: sqlite3.Connection, repo_id: int, file_id: int, fs
) -> tuple[int, dict[int, int], int]:
    """写入 module + symbols + CONTAINS 边;返回 (module_id, {局部索引→id}, 边数)。"""

    def insert(s) -> int:
        cur = conn.execute(
            "INSERT INTO symbols(repo_id, file_id, kind, name, qualified_name, "
            "line_start, line_end, signature) VALUES(?,?,?,?,?,?,?,?)",
            (repo_id, file_id, s.kind, s.name, s.qualified_name,
             s.line_start, s.line_end, s.signature),
        )
        return cur.lastrowid

    module_id = insert(fs.module)
    ids: dict[int, int] = {-1: module_id}  # 局部 parent=None → module
    for i, s in enumerate(fs.symbols):
        ids[i] = insert(s)
    edges = 0
    for parent_idx, child_idx in fs.contains_edges():
        src = ids[parent_idx] if parent_idx is not None else module_id
        conn.execute(
            "INSERT INTO edges(repo_id, src_id, dst_id, kind) VALUES(?,?,?,'CONTAINS')",
            (repo_id, src, ids[child_idx]),
        )
        edges += 1
    return module_id, ids, edges


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def index_repo(
    repo: RepoCfg,
    settings: Settings | None = None,
    embed_provider: EmbeddingProvider | None = None,
    full: bool = False,
    conn: sqlite3.Connection | None = None,
    lance: LanceStore | None = None,
    no_embed: bool = False,
) -> IndexStats:
    """索引单个仓库;embed_provider 供测试注入,生产默认实时构建。

    no_embed=True(--no-embed):纯本地索引——FTS/符号/调用边可用,跳过 embedding;
    向量检索不可用,后续配置好 embedding 后 `index --full` 全量重嵌即可补上。
    """
    t0 = time.time()
    settings = settings or get_settings()
    own_conn = conn is None
    if own_conn:
        conn = connect()
    init_db(conn)
    repo_id = upsert_repo(conn, repo)
    base_commit: str | None = None
    if not full:
        row = conn.execute(
            "SELECT indexed_commit FROM repos WHERE id=?", (repo_id,)
        ).fetchone()
        base_commit = row["indexed_commit"] if row else None

    stats = IndexStats(repo=repo.name)
    known = _known_files(conn, repo_id)
    cs = compute_change_set(repo, None if full else base_commit, known, force_scan=full)
    stats.mode, stats.base_commit, stats.head_commit = cs.mode, cs.base_commit, cs.head_commit

    # repos.yaml.exclude 在 .gitignore 之上再过滤一层,git 与 scan 两种模式都生效
    # (git ls-files 可能包含历史误提交的构建产物)
    if repo.exclude:
        import pathspec

        excl = pathspec.GitIgnoreSpec.from_lines(repo.exclude)
        cs.changes = {
            p: s for p, s in cs.changes.items() if not excl.match_file(p)
        }

    lance = lance or LanceStore(settings)

    # 完整性自愈:库中向量不完整的文件并入处理(scan 模式 mtime 跳过/中断残留),
    # 幂等重处理会删旧子树重建,embed_cache 命中不重复计费(no_embed 模式跳过)
    heal_rows = [] if no_embed else conn.execute(
        "SELECT f.path FROM files f WHERE f.repo_id=? AND f.parse_status='ok' "
        "AND EXISTS (SELECT 1 FROM chunks c WHERE c.file_id=f.id "
        "  AND NOT EXISTS (SELECT 1 FROM vector_refs v WHERE v.chunk_id=c.id))",
        (repo_id,),
    ).fetchall()
    # 中断续跑:上次在图解析(IMPORTS/CALLS)完成前中断的文件仍是 pending
    # (含当时被标记的依赖者),整文件重处理以补齐调用/导入边
    pending_rows = conn.execute(
        "SELECT path FROM files WHERE repo_id=? AND parse_status='pending'", (repo_id,)
    ).fetchall()
    for r in [*heal_rows, *pending_rows]:
        if r["path"] not in cs.changes and (repo.path / r["path"]).exists():
            cs.changes[r["path"]] = "M"

    # 依赖闭包(PLAN §9.4 步骤 4):变更文件的 IMPORTS/CALLS 入边依赖者,
    # 其边随目标子树删除而消失,需重建。必须在删除子树**之前**、并入自愈/续跑
    # 文件**之后**收集(重处理同样会级联删掉指向它的边)。
    changed_rels = set(cs.changes)
    dependent_rels = _dependent_files(conn, repo_id, cs.changes) - changed_rels
    # 依赖者与删除同事务标 pending:若中断在边重建之前,下轮据此续跑
    for part in in_chunks(sorted(dependent_rels)):
        conn.execute(
            f"UPDATE files SET parse_status='pending' WHERE repo_id=? "
            f"AND parse_status='ok' AND path IN ({','.join('?' * len(part))})",
            [repo_id, *part],
        )

    # ---- 删除 ----
    for rel, status in sorted(cs.changes.items()):
        if status == "D":
            remove_file(conn, lance, repo_id, rel)
            stats.deleted += 1
    conn.commit()

    # ---- added / modified ----
    pending = sorted((r, s) for r, s in cs.changes.items() if s != "D")
    import_refs: list[ImportRef] = []
    call_files: list[CallFile] = []
    embed_pending: list[tuple[int, str, str]] = []  # (chunk_id, kind, content)

    for batch_start in range(0, len(pending), BATCH_FILES):
        batch = pending[batch_start : batch_start + BATCH_FILES]
        for rel, status in batch:
            _process_file(
                conn, lance, repo, repo_id, rel, status, cs.head_commit,
                settings, stats, import_refs, embed_pending, call_files,
                force=full, no_embed=no_embed,
            )
        conn.commit()

    # ---- 批量 embedding(全部 flush 后;embed 与 close 同一 event loop)----
    # 按片 embed → 写 Lance/vector_refs → 提交:中断时已完成片保留,
    # 未完成文件下轮由完整性自愈补齐(embed_cache 命中不重复计费)
    if embed_pending and not no_embed:
        provider = embed_provider or EmbeddingProvider(settings, conn)
        own_provider = embed_provider is None

        async def _run_embed() -> int:
            done = 0
            try:
                for i in range(0, len(embed_pending), EMBED_SLICE):
                    part = embed_pending[i:i + EMBED_SLICE]
                    vectors = await provider.embed(
                        [c for _, _, c in part], stage="index", repo_id=repo_id
                    )
                    lance_rows = [
                        (chunk_id, repo_id, kind, blob)
                        for (chunk_id, kind, _), blob in zip(part, vectors, strict=True)
                    ]
                    for (lance_id, chunk_id) in lance.add_vectors(lance_rows):
                        conn.execute(
                            "INSERT INTO vector_refs(chunk_id, lance_id, model) VALUES(?,?,?)",
                            (chunk_id, lance_id, provider.s.embed_model),
                        )
                    conn.commit()
                    done += len(vectors)
            finally:
                if own_provider:
                    await provider.aclose()
            return done

        stats.embedded = asyncio.run(_run_embed())

    # ---- IMPORTS Pass2(含依赖闭包重建)----
    for rel in sorted(dependent_rels):
        fs = _light_imports(repo, rel)
        if fs is None:
            continue
        module_id = conn.execute(
            "SELECT s.id FROM symbols s WHERE s.kind='module' AND s.file_id="
            "(SELECT id FROM files WHERE repo_id=? AND path=?)",
            (repo_id, rel),
        ).fetchone()
        if module_id is not None:
            import_refs.append(ImportRef(module_id["id"], rel, fs[0], fs[1]))
    stats.import_edges = write_import_edges(conn, repo_id, import_refs)

    # ---- CALLS Pass2(PLAN §9.4 步骤 4:M3 生效——依赖闭包重解析调用边)----
    for rel in sorted(dependent_rels):
        cf = light_call_file(conn, repo_id, repo, rel)
        if cf is not None:
            call_files.append(cf)
    call_stats = resolve_and_write_calls(conn, repo_id, call_files)
    stats.call_sites = call_stats.sites
    stats.call_edges_exact = call_stats.exact
    stats.call_edges_heuristic = call_stats.heuristic

    # ---- 收尾:图解析完成,pending → ok;基线推进(git 模式专属)同事务 ----
    conn.execute(
        "UPDATE files SET parse_status='ok' WHERE repo_id=? AND parse_status='pending'",
        (repo_id,),
    )
    if cs.mode == "git" and cs.head_commit:
        conn.execute(
            "UPDATE repos SET indexed_commit=?, last_indexed_at=? WHERE id=?",
            (cs.head_commit, datetime.now(timezone.utc).isoformat(), repo_id),
        )
    conn.commit()

    lance.maybe_create_index()
    # 向量对账:清理 Lance 中不属于任何现存 chunk 的孤儿(中断/重处理残留);
    # 行数一致时零成本跳过
    valid_ids = {
        r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM vector_refs")
    }
    stats.orphan_vectors_deleted, missing = lance.reconcile_orphans(
        valid_ids, expected_rows=len(valid_ids)
    )
    if missing:  # 引用指向已不存在的向量:撤掉引用,下轮自愈重嵌
        for part in in_chunks(sorted(missing)):
            conn.execute(
                f"DELETE FROM vector_refs WHERE chunk_id IN ({','.join('?' * len(part))})",
                part,
            )
        conn.commit()
    stats.duration_s = time.time() - t0
    if own_conn:
        conn.close()
    return stats


def _dependent_files(
    conn: sqlite3.Connection, repo_id: int, changes: dict[str, str]
) -> set[str]:
    """反查 1 层依赖者(PLAN §9.4 步骤 4:IMPORTS/CALLS 入边)。须在删除子树前调用。

    CALLS 入边覆盖 heuristic 兜底边:裸调用他文件方法不产生 IMPORTS 边,
    唯有旧 CALLS 边能暴露这种依赖。
    """
    rels = list(changes)
    if not rels or conn.execute(
        "SELECT 1 FROM edges WHERE repo_id=? LIMIT 1", (repo_id,)
    ).fetchone() is None:
        return set()  # 首次索引:无边可反查
    out: set[str] = set()
    for part in in_chunks(rels):
        ph = ",".join("?" * len(part))
        out.update(r["p"] for r in conn.execute(
            f"SELECT DISTINCT df.path AS p FROM edges e "
            f"JOIN symbols d ON e.dst_id = d.id "
            f"JOIN files dstf ON d.file_id = dstf.id "
            f"JOIN symbols s ON e.src_id = s.id "
            f"JOIN files df ON s.file_id = df.id "
            f"WHERE e.repo_id = ? AND e.kind IN ('IMPORTS', 'CALLS') "
            f"AND dstf.path IN ({ph})",
            [repo_id, *part],
        ))
    return out


def _light_imports(repo: RepoCfg, rel: str) -> tuple[str, list[str]] | None:
    """轻量重解析:只提取语言与 imports,不动符号/块。"""
    path = repo.path / rel
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    lang = detect_language(rel)
    if lang is None:
        return None
    fs = extract_symbols(rel, lang, raw)
    return lang, fs.imports


def _process_file(
    conn, lance, repo, repo_id, rel, status, head_commit, settings, stats,
    import_refs, embed_pending, call_files, force: bool = False,
    no_embed: bool = False,
) -> None:
    abs_path = repo.path / rel
    try:
        raw = abs_path.read_bytes()
        mtime = abs_path.stat().st_mtime
    except OSError:
        stats.parse_failed += 1
        return
    new_hash = hash_bytes(raw)

    existing = _file_row(conn, repo_id, rel)
    # 幂等跳过:内容未变、解析成功且向量完整(scan 兜底/中断续跑);
    # --full 强制重解析(hash 相同也重建,修复性场景;embed_cache 仍命中)
    if (
        not force
        and existing is not None
        and existing["hash"] == new_hash
        and existing["parse_status"] == "ok"
        and (no_embed or _file_fully_embedded(conn, existing["id"]))
    ):
        stats.skipped_unchanged += 1
        return

    # 重解析前清旧子树(modified/状态修复)
    if existing is not None:
        remove_file(conn, lance, repo_id, rel)

    if status == "A":
        stats.added += 1
    else:
        stats.modified += 1

    # 跳过类:binary / 超大 / 未知扩展(files 行仍记录,便于增量比对稳定)
    def persist(status_value: str) -> None:
        conn.execute(
            "INSERT INTO files(repo_id, path, hash, mtime, parse_status, last_seen_commit) "
            "VALUES(?,?,?,?,?,?)",
            (repo_id, rel, new_hash, mtime, status_value, head_commit),
        )

    if sniff_binary(raw[:BINARY_SNIFF_BYTES]):
        persist("skipped")
        stats.skipped_binary += 1
        return
    if len(raw) > MAX_FILE_BYTES:
        persist("skipped")
        stats.skipped_large += 1
        return
    lang = detect_language(rel)
    if lang is None:
        persist("skipped")
        stats.skipped_unknown_ext += 1
        return

    # 单文件隔离:解析/写库出任何异常只记该文件 failed,不中断整次索引
    # (SAVEPOINT 回滚其半截写入;KeyboardInterrupt 等非 Exception 照常上抛)
    marks = (len(embed_pending), len(import_refs), len(call_files))
    if not conn.in_transaction:
        conn.execute("BEGIN")
    conn.execute("SAVEPOINT index_file")
    try:
        _parse_and_store(
            conn, repo_id, rel, lang, raw, new_hash, mtime, head_commit,
            settings, stats, import_refs, embed_pending, call_files,
        )
    except Exception:
        conn.execute("ROLLBACK TO index_file")
        conn.execute("RELEASE index_file")
        del embed_pending[marks[0]:], import_refs[marks[1]:], call_files[marks[2]:]
        persist("failed")
        stats.parse_failed += 1
        return
    conn.execute("RELEASE index_file")


def _parse_and_store(
    conn, repo_id, rel, lang, raw, new_hash, mtime, head_commit,
    settings, stats, import_refs, embed_pending, call_files,
) -> None:
    fs = extract_symbols(rel, lang, raw)
    # pending:图解析(IMPORTS/CALLS Pass2)完成后才置 ok,中断则下轮续跑
    cur = conn.execute(
        "INSERT INTO files(repo_id, path, hash, mtime, parse_status, last_seen_commit) "
        "VALUES(?,?,?,?,?,?)",
        (repo_id, rel, new_hash, mtime, "pending", head_commit),
    )
    file_id = cur.lastrowid

    module_id, sym_ids, edge_count = _write_symbols(conn, repo_id, file_id, fs)
    stats.symbols += len(fs.symbols) + 1
    stats.contains_edges += edge_count

    text = decode_source(raw)
    chunks = chunk_text(text, lang, fs, settings.chunk_max_tokens)
    title_to_local: dict[str, int] = {s.qualified_name: i for i, s in enumerate(fs.symbols)}
    seen_chunk_keys: set = set()
    for ch in chunks:
        local_idx = title_to_local.get(ch.title or "")
        symbol_id = sym_ids.get(local_idx) if local_idx is not None else None
        # 文件内重复内容块去重(如 md 重复段落):否则撞 idx_chunks_dedup 唯一索引
        key = (ch.kind, symbol_id if symbol_id is not None else 0, ch.content_hash)
        if key in seen_chunk_keys:
            continue
        seen_chunk_keys.add(key)
        c = conn.execute(
            "INSERT INTO chunks(repo_id, file_id, symbol_id, kind, title, content, "
            "content_hash, line_start, line_end) VALUES(?,?,?,?,?,?,?,?,?)",
            (repo_id, file_id, symbol_id, ch.kind, ch.title, ch.content,
             ch.content_hash, ch.line_start, ch.line_end),
        )
        chunk_id = c.lastrowid
        fts_insert(conn, chunk_id, ch.content)
        embed_pending.append((chunk_id, ch.kind, ch.content))
    stats.chunks += len(chunks)

    import_refs.append(ImportRef(module_id, rel, lang, fs.imports))
    call_files.append(
        CallFile(
            fs=fs,
            sym_ids=sym_ids,
            module_id=module_id,
            sites=extract_call_sites(rel, lang, raw, fs),
        )
    )
