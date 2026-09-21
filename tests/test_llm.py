"""llm.py:成功路径、重试(429/5xx/网络)、不重试 4xx、费用落库、费用上限。"""

import json

import httpx
import pytest

from codeatlas.providers._retry import CostLimitExceeded, ProviderError, ProviderHttpError
from codeatlas.providers.llm import LLMProvider


def ok_chat_body(content="pong", model="test-model", pt=10, ct=5):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
        "model": model,
    }


async def test_chat_success_and_usage_log(settings, db):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-llm-test"
        return httpx.Response(200, json=ok_chat_body())

    async with LLMProvider(settings, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        r = await p.chat([{"role": "user", "content": "ping"}], stage="doctor", max_tokens=8)

    assert r.content == "pong"
    assert r.model == "test-model"
    assert r.priced is True
    assert r.cost == pytest.approx(10 / 1e6 * 1.0 + 5 / 1e6 * 2.0)
    # 请求体带 model/messages/max_tokens
    body = json.loads(requests[0].content)
    assert body["model"] == "test-model"
    assert body["max_tokens"] == 8
    # usage_log 落库
    row = db.execute("SELECT * FROM usage_log").fetchone()
    assert row["stage"] == "doctor"
    assert row["model"] == "test-model"
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5
    assert row["cost"] == pytest.approx(r.cost)


async def test_retry_on_429_then_success(settings, db):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=ok_chat_body())

    async with LLMProvider(settings, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        r = await p.chat([{"role": "user", "content": "x"}])
    assert r.content == "pong"
    assert len(calls) == 3


async def test_retry_on_503_then_success(settings, db):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=ok_chat_body())

    async with LLMProvider(settings, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        await p.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 2


async def test_retry_exhausted_raises(settings, db):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, text="always limited")

    async with LLMProvider(settings, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        with pytest.raises(ProviderHttpError) as ei:
            await p.chat([{"role": "user", "content": "x"}])
    assert ei.value.status_code == 429
    assert len(calls) == 6  # 首次 + 5 次重试


async def test_client_error_no_retry(settings, db):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    async with LLMProvider(settings, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        with pytest.raises(ProviderHttpError) as ei:
            await p.chat([{"role": "user", "content": "x"}])
    assert ei.value.status_code == 401
    assert len(calls) == 1


async def test_unconfigured_provider_raises_clear_error(db):
    from codeatlas.config import Settings

    s = Settings(_env_file=None)
    async with LLMProvider(s, db, backoff_base=0) as p:
        with pytest.raises(ProviderError, match=r"LLM 未配置"):
            await p.chat([{"role": "user", "content": "x"}])


async def test_unknown_price_costs_zero(settings, db):
    s = settings.model_copy(update={"llm_price_prompt": None, "llm_price_completion": None})
    # test-model 不在内置表 → cost=0 且 priced=False

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_chat_body(model="totally-unknown"))

    async with LLMProvider(s, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        r = await p.chat([{"role": "user", "content": "x"}])
    assert r.priced is False
    assert r.cost == 0.0
    row = db.execute("SELECT cost FROM usage_log").fetchone()
    assert row["cost"] == 0.0


async def test_cost_limit_interrupts(settings, db):
    s = settings.model_copy(update={"cost_limit_per_run": 1e-9})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_chat_body())

    async with LLMProvider(s, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        with pytest.raises(CostLimitExceeded):
            await p.chat([{"role": "user", "content": "x"}])


async def test_malformed_response_raises(settings, db):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    async with LLMProvider(settings, db, transport=httpx.MockTransport(handler),
                           backoff_base=0) as p:
        with pytest.raises(ProviderError, match=r"缺 choices"):
            await p.chat([{"role": "user", "content": "x"}])
