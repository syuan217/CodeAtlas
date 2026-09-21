"""config.py:默认值、.env 覆盖、单价查找、repos.yaml 加载。"""

from pathlib import Path

from codeatlas import config
from codeatlas.config import (
    Settings,
    content_hash,
    load_repos,
    lookup_embed_price,
    lookup_llm_price,
)


def test_key_defaults_isolated_from_env_file():
    """不读任何 .env 时的默认值(PLAN §5/§8)。"""
    s = Settings(_env_file=None)
    assert s.llm_context_window == 128000
    assert s.embed_dim == 1024
    assert s.embed_batch_size == 64
    assert s.max_concurrency == 4
    assert s.cost_limit_per_run == 50.0
    assert s.chunk_max_tokens == 512
    assert s.sim_threshold == 0.25
    assert s.vector_top_k == 20
    assert s.context_top_n == 8
    assert s.graph_expand_hops == 2
    assert s.llm_base_url == ""


def test_env_file_override(tmp_path):
    env = tmp_path / ".env"
    # 值带尾随空格时须加引号(dotenv 语义),如 bge 前缀 "q: "
    env.write_text(
        'EMBED_DIM=768\nMAX_CONCURRENCY=8\nEMBED_QUERY_PREFIX="q: "\n',
        encoding="utf-8",
    )
    s = Settings(_env_file=env)
    assert s.embed_dim == 768
    assert s.max_concurrency == 8
    assert s.embed_query_prefix == "q: "


def test_price_lookup_prefix_and_override(settings):
    # 显式覆盖优先
    assert lookup_llm_price("anything", settings) == (1.0, 2.0)
    assert lookup_embed_price("anything", settings) == 0.5
    # 未覆盖时走内置表:最长前缀匹配
    bare = Settings(_env_file=None)
    assert lookup_llm_price("glm-4.6", bare) == (0.6, 2.2)
    # glm-4 前缀不应吞掉 glm-4.6(更长前缀优先)
    assert lookup_llm_price("totally-unknown-model", bare) is None
    assert lookup_embed_price("embedding-3", bare) == 0.5
    assert lookup_embed_price("nope", bare) is None


def test_content_hash_stable_and_32hex():
    h1 = content_hash("hello")
    h2 = content_hash("hello")
    h3 = content_hash("world")
    assert h1 == h2 and h1 != h3
    assert len(h1) == 32
    int(h1, 16)  # 合法 hex


def test_load_repos_missing_file_returns_empty(tmp_path):
    assert load_repos(tmp_path / "nope.yaml") == []


def test_load_repos_parses_entries(tmp_path):
    p = tmp_path / "repos.yaml"
    p.write_text(
        """
repos:
  - name: order-service
    path: /Users/yinn/code/order-service
    languages: [java, xml]
    exclude:
      - "**/generated/**"
  - name: web
    path: /Users/yinn/code/web
""",
        encoding="utf-8",
    )
    repos = load_repos(p)
    assert [r.name for r in repos] == ["order-service", "web"]
    assert repos[0].languages == ["java", "xml"]
    assert repos[0].exclude == ["**/generated/**"]
    assert repos[1].languages == []
    assert repos[1].path == Path("/Users/yinn/code/web")


def test_ensure_dirs_creates_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "LANCEDB_DIR", tmp_path / "lancedb")
    monkeypatch.setattr(config, "WIKI_DIR", tmp_path / "wiki")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    config.ensure_dirs()
    for sub in ("lancedb", "wiki", "docs", "profiles", "reports"):
        assert (tmp_path / sub).is_dir()
    assert (tmp_path / "docs" / "README.md").exists()
