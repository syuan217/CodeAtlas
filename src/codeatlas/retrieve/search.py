"""融合召回(PLAN §9.5):向量 + FTS5 关键词 + 符号命中 + 图扩展,RRF 融合。

三路并行召回 + 图扩展,多来源命中按 RRF(1/(k+rank))累加;
向量侧做距离→相似度归一化(AnythingLLM 公式)并按阈值过滤。
图扩展 M2 版沿 CONTAINS/IMPORTS(装填优先级:同模块 > 一跳 > 二跳),
M3 上线 CALLS 边后升级为"直接调用 > 同模块 > 二跳,exact 优先"。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from codeatlas.config import Settings
from codeatlas.db.fts import fts_search
from codeatlas.db.lance import LanceStore
from codeatlas.providers.embedding import EmbeddingProvider

RRF_K = 60
VECTOR_SOURCE = "vector"
FTS_SOURCE = "fts"
SYMBOL_SOURCE = "symbol"
GRAPH_SOURCE = "graph"


@dataclass
class Candidate:
    chunk_id: int
    score: float = 0.0  # RRF 融合分
    vec_sim: float | None = None  # 归一化相似度(展示/阈值过滤)
    sources: set[str] = field(default_factory=set)


def distance_to_similarity(d: float) -> float:
    """cosine 距离 → 相似度归一化(AnythingLLM 公式)。"""
    if d >= 1:
        return 0.0
    if d < 0:
        return 1.0 - abs(d)
    return 1.0 - d


# ---------------------------------------------------------------------------
# 查询分析
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "那", "怎么", "如何", "什么", "哪些",
    "the", "a", "an", "is", "are", "how", "what", "where", "which", "of",
    "to", "in", "on", "for", "do", "does", "did", "and", "or", "it",
}


def tokenize_query(q: str) -> list[str]:
    """问题 → FTS5 terms:拆 camelCase、中文按 2-gram,去停用词,保序去重。"""
    terms: list[str] = []
    for tok in re.split(r"[^A-Za-z0-9_\u4e00-\u9fff]+", q):
        if not tok:
            continue
        if _CJK_RE.fullmatch(tok):
            s = tok
            for i in range(len(s) - 1):
                terms.append(s[i : i + 2])  # 中文 bigram(unicode61 对 CJK 分词无效)
        else:
            for w in _CAMEL_RE.findall(tok):
                terms.append(w.lower())
    seen: set[str] = set()
    out = []
    for t in terms:
        if t not in seen and t not in _STOPWORDS and len(t) >= 2:
            seen.add(t)
            out.append(t)
    return out


def extract_identifiers(q: str) -> list[str]:
    """问题中疑似代码标识符(≥2 段的 CamelCase / snake_case 词)。"""
    idents: list[str] = []
    for tok in re.split(r"[^A-Za-z0-9_]+", q):
        if not tok or len(tok) < 4:
            continue
        if "_" in tok and len(tok.split("_")) >= 2:
            idents.append(tok)
            continue
        parts = _CAMEL_RE.findall(tok)
        if len(parts) >= 2 and any(p[0].isupper() for p in parts):
            idents.append(tok)
    return idents


# ---------------------------------------------------------------------------
# 各路召回
# ---------------------------------------------------------------------------

def _vector_recall(
    query_vec: list[float], s: Settings, lance: LanceStore, repo_id: int | None
) -> list[Candidate]:
    q = lance.table.search(query_vec).metric("cosine").limit(s.vector_top_k)
    if repo_id is not None:
        q = q.where(f"repo_id = {int(repo_id)}")
    rows = q.to_list()
    out = []
    for rank, row in enumerate(rows):
        sim = distance_to_similarity(row["_distance"])
        if sim < s.sim_threshold:
            continue
        out.append(
            Candidate(
                chunk_id=row["chunk_id"],
                vec_sim=sim,
                sources={VECTOR_SOURCE},
                score=1.0 / (RRF_K + rank),
            )
        )
    # 阈值过滤后重排名次(过滤掉的 rank 空洞不参与融合)
    for rank, c in enumerate(out):
        c.score = 1.0 / (RRF_K + rank)
    return out


def _fts_recall(conn: sqlite3.Connection, query: str, repo_id: int | None) -> list[int]:
    terms = tokenize_query(query)
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    where = ""
    params: list = []
    if repo_id is not None:
        where = " AND c.repo_id = ?"
        params.append(repo_id)
    rows = conn.execute(
        "SELECT c.id AS cid FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
        f"WHERE chunks_fts MATCH ?{where} ORDER BY rank LIMIT 10",
        [match, *params],
    ).fetchall()
    return [r["cid"] for r in rows]


def _symbol_recall(conn: sqlite3.Connection, query: str, repo_id: int | None):
    """标识符/长词精确匹配 symbols → (命中符号 id 列表, 符号自身 chunk id 列表)。

    候选词 = CamelCase/snake_case 标识符 + 问题中 ≥3 字符的普通词;
    用 symbols.name 精确匹配(IN 查询),库里无此符号名自然不命中,无误报。
    """
    idents = extract_identifiers(query)
    for t in tokenize_query(query):
        if len(t) >= 3 and t not in idents:
            idents.append(t)
    if not idents:
        return [], []
    ph = ",".join("?" * len(idents))
    where = f" AND s.repo_id = ?" if repo_id is not None else ""
    params: list = [*idents] + ([repo_id] if repo_id is not None else [])
    rows = conn.execute(
        f"SELECT s.id AS sid FROM symbols s WHERE s.name IN ({ph}){where}",
        params,
    ).fetchall()
    sym_ids = [r["sid"] for r in rows]
    if not sym_ids:
        return [], []
    ph2 = ",".join("?" * len(sym_ids))
    crows = conn.execute(
        f"SELECT id FROM chunks WHERE symbol_id IN ({ph2})", sym_ids
    ).fetchall()
    return sym_ids, [r["id"] for r in crows]


def _graph_expand(
    conn: sqlite3.Connection, sym_ids: list[int], repo_id: int | None, hops: int
) -> list[int]:
    """命中符号沿 CONTAINS/IMPORTS 扩展的 chunk(M2 版,无 CALLS 边)。

    装填优先级:同文件符号(同模块)> IMPORTS 一跳邻文件 > 二跳。
    """
    if not sym_ids:
        return []
    collected: list[int] = []

    def chunks_of_symbols(sids: list[int]) -> list[int]:
        ph = ",".join("?" * len(sids))
        return [
            r["id"] for r in conn.execute(
                f"SELECT id FROM chunks WHERE symbol_id IN ({ph})", sids
            )
        ]

    def chunks_of_files(fids: list[int]) -> list[int]:
        if not fids:
            return []
        ph = ",".join("?" * len(fids))
        return [
            r["id"] for r in conn.execute(
                f"SELECT id FROM chunks WHERE file_id IN ({ph})", fids
            )
        ]

    # 优先级 1:命中符号所在文件的全部符号 chunk(同模块)
    ph = ",".join("?" * len(sym_ids))
    hit_files = [
        r["file_id"] for r in conn.execute(
            f"SELECT DISTINCT file_id FROM symbols WHERE id IN ({ph})", sym_ids
        )
    ]
    collected += chunks_of_files(hit_files)

    # 沿 IMPORTS 边跳(出边=它依赖谁,入边=谁依赖它),每跳取邻文件 chunk
    frontier_files = hit_files
    seen_files = set(hit_files)
    for _hop in range(max(1, hops)):
        if not frontier_files:
            break
        ph = ",".join("?" * len(frontier_files))
        rows = conn.execute(
            f"SELECT DISTINCT f2.id AS fid FROM edges e "
            f"JOIN symbols a ON e.src_id = a.id "
            f"JOIN symbols b ON e.dst_id = b.id "
            f"JOIN files f2 ON b.file_id = f2.id "
            f"WHERE a.file_id IN ({ph}) AND e.kind IN ('IMPORTS','CONTAINS')",
            frontier_files,
        ).fetchall()
        next_files = [r["fid"] for r in rows if r["fid"] not in seen_files]
        if not next_files:
            break
        collected += chunks_of_files(next_files)
        seen_files.update(next_files)
        frontier_files = next_files

    return collected


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def retrieve(
    query: str,
    s: Settings,
    conn: sqlite3.Connection,
    embedder: EmbeddingProvider,
    lance: LanceStore,
    repo_id: int | None = None,
) -> list[Candidate]:
    """三路召回 + 图扩展,RRF 融合,返回按融合分降序的候选。"""
    by_chunk: dict[int, Candidate] = {}

    def add(chunk_ids: list[int], source: str, offset: int = 0) -> None:
        for rank, cid in enumerate(chunk_ids):
            if cid not in by_chunk:
                by_chunk[cid] = Candidate(chunk_id=cid)
            c = by_chunk[cid]
            c.sources.add(source)
            c.score += 1.0 / (RRF_K + rank + offset)

    # a. 向量召回
    vec = await embedder.embed_query(query)
    for c in _vector_recall(vec, s, lance, repo_id):
        if c.chunk_id not in by_chunk:
            by_chunk[c.chunk_id] = c
        else:
            by_chunk[c.chunk_id].sources.update(c.sources)
            by_chunk[c.chunk_id].vec_sim = by_chunk[c.chunk_id].vec_sim or c.vec_sim
            by_chunk[c.chunk_id].score += c.score

    # b. 关键词召回
    add(_fts_recall(conn, query, repo_id), FTS_SOURCE)

    # c. 符号命中 + d. 图扩展(符号精确匹配是强信号,offset=0 与向量/FTS 同权)
    sym_ids, sym_chunks = _symbol_recall(conn, query, repo_id)
    add(sym_chunks, SYMBOL_SOURCE)
    graph_chunks = _graph_expand(conn, sym_ids, repo_id, s.graph_expand_hops)
    sym_set = set(sym_chunks)
    graph_only = [c for c in graph_chunks if c not in sym_set]
    add(graph_only, GRAPH_SOURCE, offset=2 * RRF_K)  # 图扩展名次靠后,弱信号

    return sorted(by_chunk.values(), key=lambda c: -c.score)
