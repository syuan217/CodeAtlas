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
    tb = ToolBox(jenv.conn, tmp_path / "java_mini", None)
    out = asyncio.run(tb.execute(
        "read_file", {"path": "com/example/util/Strings.java",
                      "start_line": 1, "end_line": 5}))
    assert "[lines 1-5" in out and "truncate" in out
    out2 = asyncio.run(tb.execute(
        "read_file", {"path": "/etc/passwd", "start_line": 1, "end_line": 5}))
    assert "不在索引内" in out2  # 白名单防护


def test_toolbox_definition_and_callers(jenv, tmp_path):
    tb = ToolBox(jenv.conn, tmp_path / "java_mini", None)
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
    tb = ToolBox(jenv.conn, tmp_path / "java_mini", None)
    result = asyncio.run(run_agent("字符串截断被谁使用", llm, jenv.conn, tb))
    assert result.answer.startswith("结论")
    assert result.turns == 3
    assert [c["tool"] for c in result.tool_trace] == ["search", "callers"]
    assert mock.requests[0].get("tools")            # 首轮带工具规格
    roles = [m["role"] for m in mock.requests[2]["messages"]]
    assert "tool" in roles                          # 工具结果已回传


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
    tb = ToolBox(jenv.conn, tmp_path / "java_mini", None)
    result = asyncio.run(run_agent("x", llm, jenv.conn, tb, max_turns=3))
    assert result.answer == "被迫总结完毕"
    assert result.turns == 3
