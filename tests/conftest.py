"""pytest 公共 fixture。测试一律用临时库与显式 Settings,不碰项目 data/ 与 .env。"""

from __future__ import annotations

import pytest

from codeatlas.config import Settings
from codeatlas.db.models import connect, init_db


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
