"""cli.py:status/cost/doctor 的命令行行为(不触网)。"""

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import codeatlas.cli as cli_mod
from codeatlas import config as config_mod
from codeatlas.config import Settings
from codeatlas.db.models import connect, init_db
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
