"""可选 rerank(PLAN §5/§9.5):候选池启发式 + 可插拔接口。

候选池大小 = max(10, min(50, 语料 10%))(AnythingLLM 启发式)。
RERANK_ENABLED 默认关闭:开启后走 rerank API(服务商未定,首版提供
passthrough 实现与接口形状,真实 API 版在确定服务商后补)。
"""

from __future__ import annotations

from codeatlas.config import Settings
from codeatlas.retrieve.search import Candidate


def candidate_pool_size(corpus_size: int) -> int:
    """rerank 候选池:语料 10%,下限 10,上限 50。"""
    return max(10, min(50, corpus_size // 10))


class Reranker:
    """可插拔 rerank。默认 Passthrough(不排序,直接截断到池大小)。"""

    name = "passthrough"

    def rerank(self, query: str, cands: list[Candidate]) -> list[Candidate]:
        return cands


def maybe_rerank(
    query: str, cands: list[Candidate], s: Settings, corpus_size: int
) -> list[Candidate]:
    """按配置应用 rerank;未开启时原样返回。"""
    if not s.rerank_enabled or not cands:
        return cands
    pool = candidate_pool_size(corpus_size)
    reranker = Reranker()
    return reranker.rerank(query, cands)[:pool]
