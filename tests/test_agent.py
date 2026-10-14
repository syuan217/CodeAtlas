"""M6 agent 测试:工具循环、tool_calls 协议、轮数上限与强制综答、截断。"""

import asyncio
import json

import httpx
import pytest

from codeatlas.agent.loop import run_agent
from codeatlas.agent.tools import TOOL_SPECS, ToolBox, _truncate
from codeatlas.config import RepoCfg
from codeatlas.providers.llm import LLMProvider
from conftest import Env, FakeEmbedServer, copy_fixture


@pytest.fixture
def jenv(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = copy_fixture("java_mini", tmp_path)
    e.run(RepoCfg(name="jmini", path=root))
    yield e
    e.close()


class AgentMockTransport(httpx.MockTransport):
    """脚本化响应:依次返回 tool_calls(search→callers)再终答。"""

    def __init__(self):
        self.requests: list[dict] = []
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        n = len(self.requests)
        if n == 1:
            msg = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "search",
                             "arguments": json.dumps({"query": "字符串截断"})},
            }]}
        elif n == 2:
            msg = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_2", "type": "function",
                "function": {"name": "callers",
                             "arguments": json.dumps({"symbol": "truncate"})},
            }]}
        else:
            msg = {"role": "assistant",
                   "content": "结论:由 [UserService.java:10-19] 的 validate 调用"}
        return httpx.Response(200, json={
            "choices": [{"message": msg,
                         "finish_reason": "tool_calls" if n < 3 else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
            "model": "test-model",
        })


def test_tool_specs_shape():
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert names == ["search", "definition", "callers", "callees", "impact", "read_file"]
    for t in TOOL_SPECS:
        assert t["function"]["description"]
        assert "required" in t["function"]["parameters"]


def test_toolbox_read_file_whitelist(jenv, tmp_path):
    tb = ToolBox(jenv.conn, None)
    out = asyncio.run(tb.execute(
        "read_file", {"path": "com/example/util/Strings.java",
                      "start_line": 1, "end_line": 5}))
    assert "[lines 1-5" in out and "truncate" in out
    out2 = asyncio.run(tb.execute(
        "read_file", {"path": "/etc/passwd", "start_line": 1, "end_line": 5}))
    assert "不在索引内" in out2  # 白名单防护


def test_toolbox_definition_and_callers(jenv, tmp_path):
    tb = ToolBox(jenv.conn, None)
    out = asyncio.run(tb.execute(
        "definition", {"symbol": "com.example.util.Strings#truncate"}))
    assert "Strings.java" in out and "truncate" in out
    out2 = asyncio.run(tb.execute("callers", {"symbol": "truncate"}))
    assert "UserService#validate" in out2  # exact 调用边


def test_truncate():
    text = "x" * 400_000
    out = _truncate(text)
    assert "截断" in out and len(out) < 400_000


def test_run_agent_loop(jenv, tmp_path):
    mock = AgentMockTransport()
    llm = LLMProvider(jenv.settings, jenv.conn, transport=mock, backoff_base=0)
    tb = ToolBox(jenv.conn, None)
    result = asyncio.run(run_agent("字符串截断被谁使用", llm, jenv.conn, tb))
    assert result.answer.startswith("结论")
    assert result.turns == 3
    assert [c["tool"] for c in result.tool_trace] == ["search", "callers"]
    assert mock.requests[0].get("tools")            # 首轮带工具规格
    roles = [m["role"] for m in mock.requests[2]["messages"]]
    assert "tool" in roles                          # 工具结果已回传
    # messages = system + 本问问答对(供会话记忆复用)
    assert [m["role"] for m in result.messages] == ["system", "user", "assistant"]
    assert result.messages[2]["content"] == result.answer


def test_run_agent_history_memory(jenv, tmp_path):
    """会话记忆:第二问携带第一问问答对,但不携带工具过程消息。"""
    mock = AgentMockTransport()
    llm = LLMProvider(jenv.settings, jenv.conn, transport=mock, backoff_base=0)
    tb = ToolBox(jenv.conn, None)
    r1 = asyncio.run(run_agent("字符串截断被谁使用", llm, jenv.conn, tb))
    r2 = asyncio.run(run_agent(
        "那它的调用方还有谁", llm, jenv.conn, tb, history=r1.messages))
    # 第 4 次请求 = 第二问首轮(mock 脚本:前 3 次 = 第一问)
    msgs = mock.requests[3]["messages"]
    assert "tool" not in [m["role"] for m in msgs]  # 工具过程不入记忆
    assert any(m["role"] == "user" and "字符串截断" in m.get("content", "")
               for m in msgs)                       # 第一问在记忆里
    assert any(m["role"] == "assistant" and m.get("content") == r1.answer
               for m in msgs)                       # 第一答在记忆里
    assert [m["role"] for m in r2.messages] == [
        "system", "user", "assistant", "user", "assistant",
    ]


def test_toolbox_reset(jenv, tmp_path):
    tb = ToolBox(jenv.conn, None)
    asyncio.run(tb.execute("definition", {"symbol": "truncate"}))
    assert tb.calls
    tb.reset()
    assert tb.calls == []


def test_run_agent_turn_limit_forces_answer(jenv, tmp_path):
    """轮数耗尽 → 注入强制综答,不无限循环。"""

    class AlwaysTool(httpx.MockTransport):
        def __init__(self):
            self.n = 0
            super().__init__(self._h)

        def _h(self, request):
            self.n += 1
            body = json.loads(request.content)
            forced = any(
                isinstance(m.get("content"), str) and "上限" in m.get("content", "")
                for m in body["messages"]
            )
            if forced:
                msg = {"role": "assistant", "content": "被迫总结完毕"}
            else:
                msg = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": f"c{self.n}", "type": "function",
                    "function": {"name": "definition",
                                 "arguments": json.dumps({"symbol": "Strings"})},
                }]}
            return httpx.Response(200, json={
                "choices": [{"message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": "test-model",
            })

    llm = LLMProvider(jenv.settings, jenv.conn, transport=AlwaysTool(),
                      backoff_base=0)
    tb = ToolBox(jenv.conn, None)
    result = asyncio.run(run_agent("x", llm, jenv.conn, tb, max_turns=3))
    assert result.answer == "被迫总结完毕"
    assert result.turns == 3
    # 强制综答的合成提示只进工作副本,不进会话记忆
    assert all("上限" not in (m.get("content") or "") for m in result.messages)
    assert result.messages[-1]["role"] == "assistant"


def test_read_file_resolves_repo_root_from_index(jenv, tmp_path, monkeypatch):
    """不带 --repo(ask 默认):按索引里的仓库路径读,与当前工作目录无关。"""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    tb = ToolBox(jenv.conn, None)
    out = asyncio.run(tb.execute("read_file", {"path": "com/example/util/Strings.java"}))
    assert "truncate" in out and "读取失败" not in out


def test_read_file_same_path_in_two_repos(jenv, tmp_path):
    other = tmp_path / "other"
    (other / "com/example/util").mkdir(parents=True)
    (other / "com/example/util/Strings.java").write_text(
        "package com.example.util;\npublic class Strings { /* OTHER_REPO */ }\n"
    , encoding="utf-8")
    jenv.run(RepoCfg(name="other", path=other))
    tb = ToolBox(jenv.conn, None)
    out = asyncio.run(tb.execute("read_file", {"path": "com/example/util/Strings.java"}))
    assert "jmini" in out and "other" in out and "repo" in out   # 歧义:列出仓库让模型指定
    out2 = asyncio.run(tb.execute(
        "read_file", {"path": "com/example/util/Strings.java", "repo": "other"}))
    assert "OTHER_REPO" in out2
    jid = jenv.one("SELECT id FROM repos WHERE name='jmini'")["id"]
    scoped = ToolBox(jenv.conn, jid)                                  # --repo 限定时无歧义
    assert "truncate" in asyncio.run(scoped.execute(
        "read_file", {"path": "com/example/util/Strings.java"}))


def test_search_results_carry_repo_name(jenv):
    from codeatlas.retrieve.search import retrieve

    async def _ret(q):
        return await retrieve(q, jenv.settings, jenv.conn, None, None)

    tb = ToolBox(jenv.conn, None, retriever=_ret)
    out = asyncio.run(tb.execute("search", {"query": "truncate"}))
    assert "repo=jmini" in out


def test_run_agent_keeps_context_within_window(jenv, monkeypatch):
    """多轮大工具结果:发给模型的消息总量不得超出 LLM_CONTEXT_WINDOW,
    超出时最早的工具结果被替换为省略说明(问题与记忆不动)。"""
    from codeatlas.agent.tools import approx_tokens

    s = jenv.settings.model_copy(update={"llm_context_window": 20_000})

    class BigTool(httpx.MockTransport):
        def __init__(self):
            self.bodies = []
            super().__init__(self._h)

        def _h(self, request):
            body = json.loads(request.content)
            self.bodies.append(body)
            n = len(self.bodies)
            if n <= 4:
                msg = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": f"c{n}", "type": "function",
                    "function": {"name": "search", "arguments": json.dumps({"query": f"q{n}"})},
                }]}
            else:
                msg = {"role": "assistant", "content": "done"}
            return httpx.Response(200, json={
                "choices": [{"message": msg}], "model": "test-model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    mock = BigTool()
    llm = LLMProvider(s, jenv.conn, transport=mock, backoff_base=0)
    tb = ToolBox(jenv.conn, None)

    async def huge(name, args):
        return "订单数据" * 2_000  # 约 8000 tokens(中文按字计)

    monkeypatch.setattr(tb, "execute", huge)
    result = asyncio.run(run_agent("问题", llm, jenv.conn, tb, max_turns=6))
    assert result.answer == "done"
    budget = 20_000 - 8192
    for body in mock.bodies:
        total = sum(approx_tokens(m.get("content") or "") for m in body["messages"])
        assert total <= budget
    last = mock.bodies[-1]["messages"]
    assert any("已省略" in (m.get("content") or "") for m in last if m["role"] == "tool")
    assert last[1]["role"] == "user" and last[1]["content"].startswith("问题")
