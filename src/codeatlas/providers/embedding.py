"""批量 embedding:embed_cache 命中跳过、query/passage 前缀、维度校验、费用落库。"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import struct

import httpx

from codeatlas.config import Settings, content_hash, get_settings, lookup_embed_price
from codeatlas.cost import log_usage
from codeatlas.providers._retry import CostTracker, ProviderError, post_with_retry


def pack_vector(vec: list[float]) -> bytes:
    """float32 little-endian 打包(embed_cache / LanceDB 统一存储格式)。"""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class EmbeddingProvider:
    """`{EMBED_BASE_URL}/embeddings` 客户端。

    - 先查 embed_cache(content_hash+model 命中直接复用,不发请求);
    - 未命中的按 EMBED_BATCH_SIZE 分批,并发受信号量限制,批间延迟可配;
    - 维度校验:任一返回向量维度 ≠ EMBED_DIM 直接报错(防换模型未改配置串库);
    - query/passage 前缀在此层加(前缀参与内容哈希,两种前缀分开缓存)。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        conn: sqlite3.Connection | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        tracker: CostTracker | None = None,
        backoff_base: float = 0.8,
    ):
        self.s = settings or get_settings()
        self.conn = conn
        self._transport = transport  # 测试注入 MockTransport
        self.tracker = tracker or CostTracker(self.s.cost_limit_per_run)
        self.backoff_base = backoff_base
        self._sem = asyncio.Semaphore(self.s.max_concurrency)
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> EmbeddingProvider:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _require_config(self) -> None:
        if not (self.s.embed_base_url and self.s.embed_api_key and self.s.embed_model):
            raise ProviderError(
                "Embedding 未配置:请在 .env 填 EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL"
                "(可从 .env.example 拷贝模板)"
            )

    async def embed(
        self,
        texts: list[str],
        *,
        input_type: str = "passage",  # "query" | "passage"
        stage: str = "index",
        repo_id: int | None = None,
    ) -> list[bytes]:
        """批量 embed,返回与输入同序的 float32 LE 字节串(全部命中的缓存条目也走同一格式)。"""
        self._require_config()
        if input_type not in ("query", "passage"):
            raise ValueError(f"input_type 必须是 query/passage,得到 {input_type!r}")

        prefix = (
            self.s.embed_query_prefix if input_type == "query"
            else self.s.embed_passage_prefix
        )
        final_texts = [prefix + t for t in texts]
        hashes = [content_hash(t) for t in final_texts]
        vectors: dict[int, bytes] = {}

        # 1) 缓存命中(命中即跳过,不发请求)
        miss_idx = list(range(len(texts)))
        if self.conn is not None and texts:
            ph = ",".join("?" * len(hashes))
            rows = self.conn.execute(
                f"SELECT content_hash, vector FROM embed_cache "
                f"WHERE model=? AND content_hash IN ({ph})",
                [self.s.embed_model, *hashes],
            ).fetchall()
            hit = {r["content_hash"]: r["vector"] for r in rows}
            miss_idx = [i for i, h in enumerate(hashes) if h not in hit]
            vectors = {i: hit[h] for i, h in enumerate(hashes) if h in hit}

        # 2) 未命中分批请求
        total_tokens = 0
        if miss_idx:
            url = self.s.embed_base_url.rstrip("/") + "/embeddings"
            headers = {"Authorization": f"Bearer {self.s.embed_api_key}"}
            batch_size = max(1, self.s.embed_batch_size)
            batches = [
                miss_idx[i:i + batch_size] for i in range(0, len(miss_idx), batch_size)
            ]

            async def run_batch(idxs: list[int]) -> tuple[list[dict], int]:
                payload = {
                    "model": self.s.embed_model,
                    "input": [final_texts[i] for i in idxs],
                }
                async with self._sem:
                    resp = await post_with_retry(
                        self._ensure_client(),
                        url,
                        headers=headers,
                        json_payload=payload,
                        backoff_base=self.backoff_base,
                    )
                    if self.s.embed_batch_delay_ms:
                        await asyncio.sleep(self.s.embed_batch_delay_ms / 1000)
                data = resp.json()
                usage = data.get("usage") or {}
                return data.get("data") or [], int(usage.get("total_tokens", 0))

            results = await asyncio.gather(*[run_batch(b) for b in batches])

            # 3) 对齐校验(维度/有限值)+ 回写缓存;gather 保序,可按批次 zip
            cache_rows: list[tuple] = []
            for idxs, (items, tokens) in zip(batches, results):
                total_tokens += tokens
                if len(items) != len(idxs):
                    raise ProviderError(
                        f"embedding 返回条数 {len(items)} ≠ 请求条数 {len(idxs)}"
                    )
                for item in items:
                    i = idxs[item["index"]]
                    vec = item["embedding"]
                    if len(vec) != self.s.embed_dim:
                        raise ProviderError(
                            f"embedding 维度校验失败:期望 EMBED_DIM={self.s.embed_dim},"
                            f"模型 {self.s.embed_model} 实返 {len(vec)} 维 —— "
                            f"检查 .env 的 EMBED_MODEL 与 EMBED_DIM 是否匹配"
                        )
                    if not all(math.isfinite(x) for x in vec):
                        raise ProviderError("embedding 向量含非有限值(NaN/Inf),拒绝入库")
                    blob = pack_vector(vec)
                    vectors[i] = blob
                    cache_rows.append((hashes[i], self.s.embed_model, self.s.embed_dim, blob))

            if self.conn is not None and cache_rows:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO embed_cache(content_hash, model, dim, vector) "
                    "VALUES(?,?,?,?)",
                    cache_rows,
                )
                self.conn.commit()

            # 4) 费用落库(usage_log)
            price = lookup_embed_price(self.s.embed_model, self.s)
            cost = total_tokens / 1e6 * price if price is not None else 0.0
            if self.conn is not None and total_tokens:
                log_usage(
                    self.conn,
                    stage=stage,
                    repo_id=repo_id,
                    model=self.s.embed_model,
                    prompt_tokens=total_tokens,
                    completion_tokens=0,
                    cost=cost,
                )
            self.tracker.add(cost, f"embed {self.s.embed_model}")

        return [vectors[i] for i in range(len(texts))]

    async def embed_query(self, text: str, *, stage: str = "ask") -> list[float]:
        """单条 query embedding,返回 float 列表(检索用)。"""
        blob = (await self.embed([text], input_type="query", stage=stage))[0]
        return unpack_vector(blob)
