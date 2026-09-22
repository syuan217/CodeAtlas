"""retrieve 管线单测:分词、标识符、各路召回、RRF 融合、上下文组装。"""

import pytest

from codeatlas.config import RepoCfg
from codeatlas.retrieve.context import build_context
from codeatlas.retrieve.rerank import candidate_pool_size, maybe_rerank
from codeatlas.retrieve.search import (
    distance_to_similarity,
    extract_identifiers,
    retrieve,
    tokenize_query,
)
from conftest import Env, SemanticFakeServer, copy_fixture


def make_env(tmp_path, monkeypatch) -> Env:
    e = Env(tmp_path, monkeypatch, server=SemanticFakeServer())
    copy_fixture("java_mini", tmp_path)
    copy_fixture("ts_mini", tmp_path)
    e.run(RepoCfg(name="jmini", path=tmp_path / "java_mini", languages=["java", "markdown"]))
    e.run(RepoCfg(name="tmini", path=tmp_path / "ts_mini", languages=["typescript"]))
    return e


@pytest.fixture
def senv(tmp_path, monkeypatch) -> Env:
    return make_env(tmp_path, monkeypatch)


async def do_retrieve(env: Env, q: str, **kw):
    return await retrieve(q, env.settings, env.conn, env.provider(), env.lance(), **kw)


def file_of(env: Env, chunk_id: int) -> str:
    return env.one(
        "SELECT f.path AS p FROM chunks c JOIN files f ON c.file_id=f.id WHERE c.id=?",
        chunk_id,
    )["p"]


# ---------------------------------------------------------------------------
# 查询分析
# ---------------------------------------------------------------------------

def test_tokenize_query_camel_and_cjk():
    terms = tokenize_query("UnderwritingRemindDomainService 怎么实现?")
    assert "underwriting" in terms and "remind" in terms
    assert "怎么" not in terms  # 停用词
    zh = tokenize_query("核保提醒")
    assert zh == ["核保", "保提", "提醒"]  # bigram


def test_extract_identifiers():
    assert extract_identifiers("UnderwritingRemindDomainService 如何工作") == [
        "UnderwritingRemindDomainService"
    ]
    assert extract_identifiers("parse_amount 函数在哪") == ["parse_amount"]
    assert extract_identifiers("how to do it") == []  # 无 CamelCase/snake


def test_distance_to_similarity():
    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(0.75) == pytest.approx(0.25)
    assert distance_to_similarity(1.5) == 0.0
    assert distance_to_similarity(-0.2) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 召回与融合
# ---------------------------------------------------------------------------

async def test_vector_recall_ranks_relevant_first(senv):
    cands = await do_retrieve(senv, "如何实现字符串截断 truncate")
    assert cands, "应有候选"
    top_files = {file_of(senv, c.chunk_id) for c in cands[:3]}
    assert "com/example/util/Strings.java" in top_files


async def test_symbol_recall_exact_identifier(senv):
    cands = await do_retrieve(senv, "Strings 类的 truncate 方法")
    sym_chunks = [c for c in cands if "symbol" in c.sources]
    assert sym_chunks, "标识符命中应产生 symbol 来源候选"
    assert any(file_of(senv, c.chunk_id) == "com/example/util/Strings.java"
               for c in sym_chunks)


async def test_graph_expansion_via_imports(senv):
    # validate 符号在 UserService;图扩展应把 IMPORTS 邻居(User/Strings)拉进来
    cands = await do_retrieve(senv, "validate 校验用户 user")
    files = {file_of(senv, c.chunk_id) for c in cands}
    assert "com/example/model/User.java" in files  # UserService import User
    assert any("graph" in c.sources for c in cands)


async def test_multi_source_gets_rrf_boost(senv):
    cands = await do_retrieve(senv, "如何实现字符串截断 truncate")
    top = cands[0]
    assert len(top.sources) >= 2  # 向量+FTS 多路命中
    assert top.score > 0


async def test_repo_filter(senv):
    cands = await do_retrieve(senv, "cart 购物车 total")
    jmini_id = senv.one("SELECT id FROM repos WHERE name='jmini'")["id"]
    for c in cands:
        r = senv.one("SELECT repo_id FROM chunks WHERE id=?", c.chunk_id)
        assert r["repo_id"] != jmini_id


async def test_fts_recall_camel_split(senv):
    # camelCase 拆词后 FTS 应命中 totalCents
    cands = await do_retrieve(senv, "totalCents 计算")
    assert any(file_of(senv, c.chunk_id) == "src/order.ts" for c in cands[:3])


# ---------------------------------------------------------------------------
# 上下文组装
# ---------------------------------------------------------------------------

async def test_build_context_format_and_budget(senv):
    cands = await do_retrieve(senv, "如何实现字符串截断 truncate")
    ctx, used = build_context(senv.conn, cands, top_n=8, budget_tokens=10 ** 6)
    assert ctx
    assert "### 文件:" in ctx
    assert "[lines " in ctx
    assert len(used) <= 8

    # 极小预算:装填受限,不抛错
    ctx2, used2 = build_context(senv.conn, cands, top_n=8, budget_tokens=5)
    assert len(used2) < len(used) or not ctx2


async def test_build_context_file_grouping(senv):
    cands = await do_retrieve(senv, "validate 校验用户 user")
    ctx, _ = build_context(senv.conn, cands, top_n=8, budget_tokens=10 ** 6)
    # 同文件 chunk 应聚合在同一个文件节内
    assert ctx.count("### 文件:com/example/service/UserService.java") <= 1


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

def test_candidate_pool_size():
    assert candidate_pool_size(50) == 10
    assert candidate_pool_size(5000) == 50
    assert candidate_pool_size(300) == 30


def test_maybe_rerank_disabled_passthrough(senv):
    from codeatlas.retrieve.search import Candidate

    cands = [Candidate(chunk_id=i) for i in range(80)]
    out = maybe_rerank("q", cands, senv.settings, corpus_size=1000)
    assert out == cands  # 默认关闭,原样返回
