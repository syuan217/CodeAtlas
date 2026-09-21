"""embedding.py:前缀、批量切分、缓存命中、维度校验、费用落库。"""

import json
import struct

import httpx
import pytest

from codeatlas.providers._retry import ProviderError
from codeatlas.providers.embedding import EmbeddingProvider, pack_vector, unpack_vector


class EmbedServer:
    """假 embedding 服务:记录请求,返回按文本确定性生成的向量("t3" → 0.4×dim)。"""

    def __init__(self, dim=4, tokens_per_item=3):
        self.requests: list[dict] = []
        self.dim = dim
        self.tokens_per_item = tokens_per_item

    def _vec_for(self, text: str) -> list[float]:
        try:
            n = int(text.split("t")[-1])
        except ValueError:
            n = 0
        return [0.1 * (n + 1)] * self.dim

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        data = [
            {"index": i, "embedding": self._vec_for(t)}
            for i, t in enumerate(body["input"])
        ]
        usage = {"total_tokens": self.tokens_per_item * len(body["input"])}
        return httpx.Response(200, json={"data": data, "usage": usage})


async def test_prefix_applied_and_cached_separately(settings, db):
    s = settings.model_copy(update={"embed_query_prefix": "q: "})
    server = EmbedServer()
    p = EmbeddingProvider(s, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        await p.embed(["hello"], input_type="passage")
        await p.embed(["hello"], input_type="query")
    # passage 无前缀;query 加前缀 → 两次都发请求且内容不同(哈希不同,缓存分开)
    assert [b["input"] for b in server.requests] == [["hello"], ["q: hello"]]


async def test_cache_hit_skips_request(settings, db):
    server = EmbedServer()
    p = EmbeddingProvider(settings, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        v1 = await p.embed(["a", "b"], stage="index")
        v2 = await p.embed(["a", "b"], stage="index")  # 全部命中缓存
    assert len(server.requests) == 1  # 第二次没发请求
    assert v1 == v2
    rows = db.execute("SELECT COUNT(*) AS c FROM embed_cache").fetchone()["c"]
    assert rows == 2


async def test_partial_cache_hit(settings, db):
    server = EmbedServer()
    p = EmbeddingProvider(settings, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        await p.embed(["a"])
        await p.embed(["a", "b"])  # a 命中,只 b 需要请求
    assert len(server.requests) == 2
    assert server.requests[1]["input"] == ["b"]


async def test_batching_and_order(settings, db):
    s = settings.model_copy(update={"embed_batch_size": 2})
    server = EmbedServer()
    p = EmbeddingProvider(s, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        blobs = await p.embed(["t0", "t1", "t2", "t3", "t4"])
    # 5 条、批大小 2 → 3 个请求
    assert len(server.requests) == 3
    assert [len(b["input"]) for b in server.requests] == [2, 2, 1]
    # 返回顺序与输入一致
    for i, blob in enumerate(blobs):
        vec = unpack_vector(blob)
        assert vec == pytest.approx([0.1 * (i + 1)] * 4)


async def test_dim_mismatch_raises(settings, db):
    server = EmbedServer(dim=5)  # EMBED_DIM=4,返回 5 维 → 必须报错(防串库)
    p = EmbeddingProvider(settings, db, transport=httpx.MockTransport(server),
                          backoff_base=0)
    async with p:
        with pytest.raises(ProviderError, match=r"EMBED_DIM"):
            await p.embed(["x"])


async def test_usage_logged(settings, db):
    server = EmbedServer(tokens_per_item=10)
    p = EmbeddingProvider(settings, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        await p.embed(["a", "b"], stage="index")  # 20 tokens,单价 0.5 元/1M
    row = db.execute("SELECT * FROM usage_log").fetchone()
    assert row["stage"] == "index"
    assert row["model"] == "test-embed"
    assert row["prompt_tokens"] == 20
    assert row["completion_tokens"] == 0
    assert row["cost"] == pytest.approx(20 / 1e6 * 0.5)


async def test_cache_hit_does_not_log_usage(settings, db):
    server = EmbedServer()
    p = EmbeddingProvider(settings, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        await p.embed(["a"])
        await p.embed(["a"])  # 命中缓存 → 无新流水
    n = db.execute("SELECT COUNT(*) AS c FROM usage_log").fetchone()["c"]
    assert n == 1


async def test_embed_query_returns_floats(settings, db):
    server = EmbedServer()
    p = EmbeddingProvider(settings, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        vec = await p.embed_query("find callers of foo")
    assert isinstance(vec, list) and len(vec) == 4
    assert all(isinstance(x, float) for x in vec)


async def test_empty_input(settings, db):
    server = EmbedServer()
    p = EmbeddingProvider(settings, db, transport=httpx.MockTransport(server), backoff_base=0)
    async with p:
        assert await p.embed([]) == []
    assert server.requests == []


def test_pack_unpack_roundtrip():
    vec = [0.25, -1.5, 3.14159, 0.0]
    blob = pack_vector(vec)
    assert len(blob) == 4 * 4  # float32
    assert unpack_vector(blob) == pytest.approx(vec)
    # LE 字节序:首元素最低有效字节在前
    assert blob[:4] == struct.pack("<f", 0.25)
