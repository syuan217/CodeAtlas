"""OpenAI 兼容 chat client:并发信号量、指数退避重试、费用落 usage_log。"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass

import httpx

from codeatlas.config import Settings, get_settings, lookup_llm_price
from codeatlas.cost import log_usage
from codeatlas.providers._retry import CostTracker, ProviderError, post_with_retry


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    priced: bool  # False = 单价表查不到,cost 恒 0(unknown)


class LLMProvider:
    """`{LLM_BASE_URL}/chat/completions` 客户端。

    - 并发受 MAX_CONCURRENCY 信号量限制;
    - 429/5xx/网络错误指数退避重试(最多 5 次);
    - 每次调用 usage 记 usage_log,费用 = tokens × 单价表。
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
                timeout=httpx.Timeout(600.0, connect=10.0),
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> LLMProvider:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def chat(
        self,
        messages: list[dict],
        *,
        stage: str = "ask",
        repo_id: int | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        thinking_disabled: bool = False,
    ) -> ChatResult:
        if not (self.s.llm_base_url and self.s.llm_api_key and self.s.llm_model):
            raise ProviderError(
                "LLM 未配置:请在 .env 填 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL"
                "(可从 .env.example 拷贝模板)"
            )
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        payload: dict = {
            "model": self.s.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking_disabled:
            # 思考型模型(如 GLM 系列):思考内容同样计入输出预算,
            # 长文生成场景须关闭思考,否则正文可能被思考耗尽截断
            payload["thinking"] = {"type": "disabled"}
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}"}

        async with self._sem:
            resp = await post_with_retry(
                self._ensure_client(),
                url,
                headers=headers,
                json_payload=payload,
                backoff_base=self.backoff_base,
            )

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"LLM 响应缺 choices/message: {str(data)[:500]}") from e
        usage0 = data.get("usage") or {}
        completion0 = int(usage0.get("completion_tokens", 0))
        if not content.strip() and max_tokens and completion0 >= max_tokens:
            raise ProviderError(
                f"输出被截断:completion_tokens={completion0} 已达 max_tokens={max_tokens},"
                f"content 为空(思考型模型的思考内容耗尽了输出预算)"
            )
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        model = data.get("model", self.s.llm_model)

        price = lookup_llm_price(model, self.s)
        if price is not None:
            p_in, p_out = price
            cost = prompt_tokens / 1e6 * p_in + completion_tokens / 1e6 * p_out
            priced = True
        else:
            cost, priced = 0.0, False  # unknown 单价:记 tokens,cost=0

        if self.conn is not None:
            log_usage(
                self.conn,
                stage=stage,
                repo_id=repo_id,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )
        self.tracker.add(cost, f"llm {model}")
        return ChatResult(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            priced=priced,
        )
