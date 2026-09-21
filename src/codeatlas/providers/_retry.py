"""OpenAI 兼容端点共用:指数退避重试、单次运行费用护栏。"""

from __future__ import annotations

import asyncio
import random

import httpx

# 408 请求超时 / 429 限流 / 5xx 服务端错误 → 可重试;其余 4xx 是调用方问题,直接报错
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """Provider 层基础错误。"""


class ProviderHttpError(ProviderError):
    """HTTP 层错误(重试耗尽或不可重试状态码)。"""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class CostLimitExceeded(ProviderError):
    """单次命令费用超过 COST_LIMIT_PER_RUN,主动中断。"""


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    json_payload: dict,
    max_retries: int = 5,
    backoff_base: float = 0.8,
    max_backoff: float = 30.0,
) -> httpx.Response:
    """POST,429/5xx/网络错误按指数退避重试(最多 max_retries 次)。

    backoff_base 供测试注入 0 以跳过等待。
    """
    attempt = 0
    while True:
        network_error: Exception | None = None
        try:
            resp = await client.post(url, headers=headers, json=json_payload)
        except (httpx.TimeoutException, httpx.TransportError) as e:  # 网络层可重试
            network_error = e
            resp = None

        if resp is not None and resp.status_code < 400:
            return resp

        status = resp.status_code if resp is not None else None
        retryable = network_error is not None or (status in RETRYABLE_STATUS)
        if not retryable:
            assert resp is not None
            raise ProviderHttpError(
                f"POST {url} 返回 {resp.status_code}(不可重试):{resp.text[:500]}",
                status_code=resp.status_code,
                body=resp.text,
            )

        if attempt >= max_retries:
            if network_error is not None:
                raise ProviderHttpError(
                    f"POST {url} 重试 {max_retries} 次后仍网络错误:{network_error}"
                ) from network_error
            assert resp is not None
            raise ProviderHttpError(
                f"POST {url} 重试 {max_retries} 次后仍返回 {resp.status_code}:{resp.text[:500]}",
                status_code=resp.status_code,
                body=resp.text,
            )

        delay = min(backoff_base * (2**attempt), max_backoff)
        delay = delay * (1 + random.uniform(0, 0.25))  # 轻微抖动防同步重试
        await asyncio.sleep(delay)
        attempt += 1


class CostTracker:
    """进程内费用累计,超过 COST_LIMIT_PER_RUN 立即中断(防失控烧钱)。"""

    def __init__(self, limit: float):
        self.limit = limit
        self.total = 0.0

    def add(self, cost: float, label: str = "") -> None:
        self.total += cost
        if self.total > self.limit:
            raise CostLimitExceeded(
                f"本次运行费用 {self.total:.4f} 元已超上限 {self.limit} 元({label});"
                f"如需继续请调大 .env 里的 COST_LIMIT_PER_RUN"
            )
