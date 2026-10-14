"""retrieve 管线单测:分词、标识符、各路召回、RRF 融合、上下文组装。"""

import pytest

from codeatlas.config import RepoCfg
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


def _fts_db(tmp_path, docs: dict[int, str]):
    from codeatlas.db.fts import fts_insert
    from codeatlas.db.models import connect, init_db

    conn = connect(tmp_path / "fts.sqlite")
    init_db(conn)
    conn.execute("INSERT INTO repos(name,path,languages) VALUES('r','','[]')")
    for cid, text in docs.items():
        conn.execute("INSERT INTO chunks(id,repo_id,kind,content,content_hash) "
                     "VALUES(?,1,'code',?,?)", (cid, text, f"h{cid}"))
        fts_insert(conn, cid, text)
    conn.commit()
    return conn


def test_fts_chinese_phrase_inside_longer_run(tmp_path):
    """unicode61 把连续汉字当一个词:'订单创建流程:…' 曾无法被'订单创建'类问题命中。"""
    from codeatlas.retrieve.search import _fts_recall

    conn = _fts_db(tmp_path, {1: "// 订单创建流程:校验库存后落库", 2: "// 用户登录"})
    assert _fts_recall(conn, "订单创建流程是怎样的", None)[0] == 1
    assert _fts_recall(conn, "库存校验在哪", None)[0] == 1
    conn.close()


def test_fts_camel_case_identifier(tmp_path):
    from codeatlas.retrieve.search import _fts_recall

    conn = _fts_db(tmp_path, {
        1: "return orderService.createOrder(req);",
        2: "void cancel() {}",
    })
    assert _fts_recall(conn, "createOrder 在哪", None)[0] == 1
    assert _fts_recall(conn, "create order", None)[0] == 1
    conn.close()


def test_fts_delete_roundtrip_with_preprocessing(tmp_path):
    from codeatlas.db.fts import fts_delete

    conn = _fts_db(tmp_path, {1: "订单 createOrder 流程", 2: "订单 其它"})
    fts_delete(conn, 1, "订单 createOrder 流程")
    conn.execute("DELETE FROM chunks WHERE id=1")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
    from codeatlas.retrieve.search import _fts_recall
    assert _fts_recall(conn, "订单", None) == [2]
    conn.close()


def test_symbol_recall_skips_overly_generic_words(tmp_path):
    """问题里的普通词(如 init)匹配上百个同名符号时不进图扩展,免得结果被噪声淹没、查询变慢。"""
    from codeatlas.db.models import connect, init_db
    from codeatlas.retrieve.search import _symbol_recall

    conn = connect(tmp_path / "s.sqlite")
    init_db(conn)
    conn.execute("INSERT INTO repos(name,path,languages) VALUES('r','','[]')")
    conn.execute("INSERT INTO files(id,repo_id,path) VALUES(1,1,'a.py')")
    for i in range(40):
        conn.execute("INSERT INTO symbols(repo_id,file_id,kind,name,qualified_name) "
                     "VALUES(1,1,'method','init',?)", (f"C{i}#init",))
    conn.execute("INSERT INTO symbols(repo_id,file_id,kind,name,qualified_name) "
                 "VALUES(1,1,'method','settleOrder','Svc#settleOrder')")
    sym_ids, _ = _symbol_recall(conn, "init 之后 settleOrder 怎么走", None)
    names = {r[0] for r in conn.execute(
        f"SELECT name FROM symbols WHERE id IN ({','.join(map(str, sym_ids))})")}
    assert names == {"settleOrder"}
    conn.close()
