"""pytest 公共 fixture。测试一律用临时库与显式 Settings,不碰项目 data/ 与 .env。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from codeatlas.config import RepoCfg, Settings
from codeatlas.db.models import connect, init_db
from codeatlas.providers.embedding import EmbeddingProvider

# 防火墙:任何测试都不得读到项目真实 .env(曾因 repos.yaml 变非空导致
# 某测试意外用真实 API key 索引真实仓库,烧了约 5 元)。显式传 _env_file
# 参数的构造不受影响(init 参数优先级高于 model_config)。
_GUARD_ENV_FILE = "/nonexistent/codeatlas-test-guard.env"


@pytest.fixture(autouse=True)
def _no_real_env(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", _GUARD_ENV_FILE)


FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedServer:
    """确定性 embedding:按请求内序号生成(无语义,索引管线测试用)。"""

    def __init__(self, dim=4):
        self.dim = dim
        self.request_count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        body = json.loads(request.content)
        data = [
            {"index": i, "embedding": [0.01 * ((i % 97) + 1)] * self.dim}
            for i in range(len(body["input"]))
        ]
        return httpx.Response(
            200, json={"data": data, "usage": {"total_tokens": len(body["input"])}}
        )


class SemanticFakeServer:
    """伪语义 embedding:关键词 → 维度映射,共享关键词的文本 cosine 相近。

    检索管线与 golden 回归用:向量路召回在测试里可预测,零网络零费用。
    """

    DIM = 4
    KEYS = {
        "truncate": 0, "字符串": 0, "截断": 0, "strings": 0,
        "validate": 1, "校验": 1, "用户": 1, "user": 1, "name": 1,
        "cart": 2, "购物车": 2, "amount": 2, "金额": 2, "total": 2, "cents": 2,
        "clamp": 3, "区间": 3, "counter": 3, "计数": 3, "bump": 3,
    }

    def __init__(self, dim=4):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        low = text.lower()
        for kw, dim in self.KEYS.items():
            if kw in low:
                vec[dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0:
            vec = [1.0] + [0.0] * (self.dim - 1)  # 无关键词:固定向量
        else:
            vec = [v / norm for v in vec]
        return vec

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        data = [{"index": i, "embedding": self._vec(t)} for i, t in enumerate(body["input"])]
        return httpx.Response(
            200, json={"data": data, "usage": {"total_tokens": len(body["input"])}}
        )


class Env:
    """每个测试一套隔离环境:库 + LanceDB + provider(挂同一 conn)。

    monkeypatch=None 时直接改模块属性(供 module 级 fixture 复用),
    用完必须调用 restore()。
    """

    def __init__(self, tmp_path, monkeypatch=None, server=None):
        from codeatlas import config

        self._mp = monkeypatch
        self._old = (config.DB_PATH, config.LANCEDB_DIR)
        new_db, new_lance = tmp_path / "kb.sqlite", tmp_path / "lancedb"
        if monkeypatch is not None:
            monkeypatch.setattr(config, "DB_PATH", new_db)
            monkeypatch.setattr(config, "LANCEDB_DIR", new_lance)
        else:
            config.DB_PATH = new_db
            config.LANCEDB_DIR = new_lance
        self.settings = Settings(
            _env_file=None,
            llm_base_url="http://llm.test/v1",
            llm_api_key="sk-llm-test",
            llm_model="test-model",
            embed_base_url="http://embed.test/v1",
            embed_api_key="sk-test",
            embed_model="test-embed",
            embed_dim=4,
            embed_batch_size=64,
            chunk_max_tokens=512,
            max_concurrency=4,
            cost_limit_per_run=50.0,
        )
        self.conn = connect()
        init_db(self.conn)
        self.server = server or FakeEmbedServer()

    def restore(self) -> None:
        if self._mp is None:
            from codeatlas import config

            config.DB_PATH, config.LANCEDB_DIR = self._old

    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider(
            self.settings, self.conn,
            transport=httpx.MockTransport(self.server), backoff_base=0,
        )

    def run(self, repo: RepoCfg, **kw):
        from codeatlas.ingest.indexer import index_repo

        return index_repo(
            repo, settings=self.settings, embed_provider=self.provider(),
            conn=self.conn, **kw
        )

    def lance(self):
        from codeatlas.db.lance import LanceStore

        return LanceStore(self.settings)

    def q(self, sql: str, *params):
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, *params):
        return self.conn.execute(sql, params).fetchone()

    def count(self, table: str) -> int:
        return self.one(f"SELECT COUNT(*) AS c FROM {table}")["c"]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch)
    yield e
    e.close()


def copy_fixture(name: str, tmp: Path) -> Path:
    dst = tmp / name
    shutil.copytree(FIXTURES / name, dst)
    return dst


def git_(cwd, *args) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(cwd), *args],
        check=True, capture_output=True,
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    """完全隔离的配置:不读项目 .env,端点指向假地址(配合 MockTransport 使用)。"""
    return Settings(
        _env_file=None,
        llm_base_url="http://llm.test/v1",
        llm_api_key="sk-llm-test",
        llm_model="test-model",
        llm_price_prompt=1.0,
        llm_price_completion=2.0,
        embed_base_url="http://embed.test/v1",
        embed_api_key="sk-embed-test",
        embed_model="test-embed",
        embed_dim=4,
        embed_batch_size=2,
        embed_price=0.5,
        max_concurrency=4,
        cost_limit_per_run=50.0,
    )


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "test_kb.sqlite")
    init_db(conn)
    yield conn
    conn.close()
