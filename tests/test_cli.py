"""cli.py:status/cost/doctor 的命令行行为(不触网)。"""

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import codeatlas.cli as cli_mod
from codeatlas import config as config_mod
from codeatlas.config import Settings
from codeatlas.db.models import SCHEMA_VERSION, connect, init_db
from codeatlas.cost import log_usage

runner = CliRunner()


class _MockLLM(cli_mod.LLMProvider):
    def __init__(self, s, conn, handler):
        super().__init__(s, conn, transport=httpx.MockTransport(handler), backoff_base=0)


class _MockEmb(cli_mod.EmbeddingProvider):
    def __init__(self, s, conn, handler):
        super().__init__(s, conn, transport=httpx.MockTransport(handler), backoff_base=0)


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """把 CLI 默认库重定向到临时路径,避免碰项目 data/。"""
    target = tmp_path / "kb.sqlite"
    monkeypatch.setattr(config_mod, "DB_PATH", target)
    conn = connect(target)
    init_db(conn)
    yield conn
    conn.close()


def test_status(isolated_db):
    result = runner.invoke(cli_mod.app, ["status"])
    assert result.exit_code == 0
    assert "schema_version" in result.output
    assert "repos" in result.output
    assert "kb.sqlite" in result.output


def test_cost_empty(isolated_db):
    result = runner.invoke(cli_mod.app, ["cost"])
    assert result.exit_code == 0
    assert "费用统计" in result.output


def test_cost_with_rows(isolated_db):
    log_usage(isolated_db, stage="doctor", model="m", prompt_tokens=10,
              completion_tokens=5, cost=0.00015)
    result = runner.invoke(cli_mod.app, ["cost"])
    assert result.exit_code == 0
    assert "doctor" in result.output

    result2 = runner.invoke(cli_mod.app, ["cost", "--by-model"])
    assert result2.exit_code == 0
    assert "m" in result2.output


def test_doctor_without_config_exits_nonzero(isolated_db, monkeypatch):
    monkeypatch.setattr(cli_mod, "get_settings", lambda: Settings(_env_file=None))
    result = runner.invoke(cli_mod.app, ["doctor"])
    assert result.exit_code == 1
    assert ".env" in result.output


def test_index_without_repos_exits_nonzero(isolated_db, monkeypatch):
    monkeypatch.setattr(cli_mod, "load_repos", lambda: [])
    result = runner.invoke(cli_mod.app, ["index"])
    assert result.exit_code == 1
    assert "repos.yaml" in result.output


def test_index_without_embed_config_exits_nonzero(isolated_db, monkeypatch, tmp_path):
    import shutil as _sh

    from codeatlas.config import RepoCfg

    src = Path(__file__).parent / "fixtures" / "java_mini"
    dst = tmp_path / "java_mini"
    _sh.copytree(src, dst)
    monkeypatch.setattr(
        cli_mod, "load_repos", lambda: [RepoCfg(name="jmini", path=dst)]
    )
    monkeypatch.setattr(cli_mod, "get_settings", lambda: Settings(_env_file=None))
    result = runner.invoke(cli_mod.app, ["index"])
    assert result.exit_code == 1
    assert "Embedding 未配置" in result.output


def test_doctor_with_mock_endpoints(isolated_db, monkeypatch, settings):
    """端到端:doctor 走 MockTransport,两端点 OK、费用落库、exit 0。"""

    def llm_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "pong"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            "model": settings.llm_model,
        })

    def embed_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        data = [{"index": i, "embedding": [0.1, 0.2, 0.3, 0.4]}
                for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 1}})

    monkeypatch.setattr(cli_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_mod, "LLMProvider", lambda s, conn: _MockLLM(s, conn, llm_handler))
    monkeypatch.setattr(cli_mod, "EmbeddingProvider",
                        lambda s, conn: _MockEmb(s, conn, embed_handler))

    result = runner.invoke(cli_mod.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "维度 4" in result.output
    # 两条费用流水落库
    stages = {r["stage"] for r in isolated_db.execute("SELECT stage FROM usage_log")}
    assert stages == {"doctor"}


def test_json_outputs(isolated_db):
    """--json 输出(Agent/Skill 消费):合法 JSON 且无表格装饰。"""
    import json as _json

    r = runner.invoke(cli_mod.app, ["status", "--json"])
    assert r.exit_code == 0
    data = _json.loads(r.output)
    assert data["schema_version"] == str(SCHEMA_VERSION)
    assert "counts" in data and "repos" in data
    assert "┌" not in r.output  # 无表格框线


def test_trim_history():
    """会话记忆瘦身:保留 system + 最近 keep_pairs 组问答对。"""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(4):
        msgs += [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    out = cli_mod._trim_history(msgs, 2)
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_session_reader_non_tty_falls_back_to_input(monkeypatch):
    """非 tty(管道/脚本):回退内置 input(),行为与之前一致。"""
    import asyncio
    import io
    import sys as _sys

    monkeypatch.setattr(_sys, "stdin", io.StringIO())  # isatty() -> False
    monkeypatch.setattr("builtins.input", lambda prompt="": "管道问题")
    read = cli_mod._make_session_reader()
    assert asyncio.run(read()) == "管道问题"


def test_session_reader_tty_uses_prompt_toolkit(monkeypatch):
    """tty:走 prompt_toolkit 异步接口(修 libedit CJK 宽字符退格错位)。"""

    class TtyStdin:
        @staticmethod
        def isatty():
            return True

    def _no_builtin_input(prompt=""):
        raise AssertionError("tty 下不应回退 input()")

    import asyncio

    import prompt_toolkit

    seen = {}

    class FakePS:
        def __init__(self, *a, **kw):
            pass

        async def prompt_async(self, p, **kw):
            seen["prompt"] = p
            return "中文输入"

    monkeypatch.setattr("sys.stdin", TtyStdin())
    monkeypatch.setattr("builtins.input", _no_builtin_input)
    monkeypatch.setattr(prompt_toolkit, "PromptSession", FakePS)
    read = cli_mod._make_session_reader()
    assert asyncio.run(read()) == "中文输入"
    assert seen["prompt"] == "❯ "


def test_trim_history_zero_keeps_no_pairs():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert cli_mod._trim_history(msgs, 0) == [{"role": "system", "content": "sys"}]


def _fake_session(monkeypatch, questions, run_impl):
    """装配假会话环境:脚本化输入 + 假 run_agent,返回控制台输出收集器。"""

    from codeatlas.agent import loop as loop_mod

    class _Closable:
        async def aclose(self):
            pass

    class _Conn:
        def close(self):
            pass

    class _TB:
        calls: list = []

        def reset(self):
            pass

    monkeypatch.setattr(cli_mod, "_agent_env",
                        lambda repo: (_Conn(), _TB(), None, _Closable()))
    it = iter(questions)

    def _reader():
        async def _read():
            try:
                return next(it)
            except StopIteration:
                raise EOFError from None
        return _read

    monkeypatch.setattr(cli_mod, "_make_session_reader", _reader)
    monkeypatch.setattr(loop_mod, "run_agent", run_impl)
    out: list[str] = []
    monkeypatch.setattr(cli_mod.console, "print",
                        lambda *a, **k: out.append(
                            " ".join(str(getattr(x, "renderable", x)) for x in a)))
    return out


def _answer(text):
    from codeatlas.agent.loop import AgentResult

    return AgentResult(answer=text, messages=[
        {"role": "system", "content": "s"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": text}])


def test_agent_session_ctrl_c_interrupts_only_current_question(monkeypatch):
    """思考中 Ctrl-C:只中断本问,会话继续(asyncio 下 SIGINT 表现为 CancelledError)。"""
    import asyncio
    import os
    import signal

    calls = {"n": 0}

    async def fake_run(question, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.sleep(5)
        return _answer(f"答:{question}")

    out = _fake_session(monkeypatch, ["第一问", "第二问"], fake_run)
    asyncio.run(cli_mod._agent_session(None, None, 6, 8))
    joined = "\n".join(out)
    assert "已中断本问" in joined
    assert "答:第二问" in joined
    assert "会话关闭" in joined


def test_agent_session_provider_error_keeps_session(monkeypatch):
    import asyncio

    from codeatlas.providers._retry import ProviderError

    calls = {"n": 0}

    async def fake_run(question, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderError("POST 返回 400")
        return _answer(f"答:{question}")

    out = _fake_session(monkeypatch, ["第一问", "第二问"], fake_run)
    asyncio.run(cli_mod._agent_session(None, None, 6, 8))
    joined = "\n".join(out)
    assert "400" in joined and "答:第二问" in joined


def test_agent_env_without_embedding_uses_local_search(isolated_db, monkeypatch, tmp_path):
    """未配置 embedding:search 工具走 FTS+符号,而不是每次报 Embedding 未配置。"""
    import asyncio

    from codeatlas.config import RepoCfg
    from codeatlas.ingest.indexer import index_repo
    from conftest import copy_fixture

    s = Settings(_env_file=None, llm_base_url="http://llm.test/v1", llm_api_key="k",
                 llm_model="m", embed_dim=4)
    monkeypatch.setattr(cli_mod, "get_settings", lambda: s)
    index_repo(RepoCfg(name="jmini", path=copy_fixture("java_mini", tmp_path)),
               settings=s, conn=isolated_db, no_embed=True)
    conn, toolbox, embedder, llm = cli_mod._agent_env(None)
    try:
        assert embedder is None
        out = asyncio.run(toolbox.execute("search", {"query": "truncate"}))
        assert "Strings.java" in out and "出错" not in out
    finally:
        conn.close()
