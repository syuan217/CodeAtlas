"""配置加载:.env(服务商与关键参数)+ repos.yaml(仓库清单)+ 内置单价表。

所有关键参数(PLAN §5)在此给默认值,均可被 .env 覆盖。
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"
REPOS_YAML = PROJECT_ROOT / "repos.yaml"


def _resolve_data_dir() -> Path:
    """DATA_DIR 可配置:shell 环境变量 > .env 中 DATA_DIR 行 > 默认 项目内 data/。

    只解析 .env 里的 DATA_DIR 单键,不用 load_dotenv 整体加载——
    避免把服务商 key 灌进 os.environ,破坏测试与进程隔离。
    需在模块常量求值前执行,kb.sqlite / lancedb / wiki 等路径统一从这里派生。
    """
    val = os.environ.get("DATA_DIR")
    if val is None and ENV_FILE.exists():
        try:
            for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                if key.strip() == "DATA_DIR":
                    val = raw.split(" #")[0].strip().strip('"').strip("'")
                    break
        except OSError:
            pass
    return Path(val) if val else PROJECT_ROOT / "data"


# 运行时目录(整体 gitignore,见 PLAN §6/§13)
DATA_DIR = _resolve_data_dir()
DB_PATH = DATA_DIR / "kb.sqlite"
LANCEDB_DIR = DATA_DIR / "lancedb"
WIKI_DIR = DATA_DIR / "wiki"
DOCS_DIR = DATA_DIR / "docs"
PROFILES_DIR = DATA_DIR / "profiles"
REPORTS_DIR = DATA_DIR / "reports"
DDL_DIR = DATA_DIR / "ddl"  # 人工维护:生产 DDL 导出文件(mysqldump --no-data / SHOW CREATE TABLE)


class Settings(BaseSettings):
    """全部可配置项。字段名小写,对应 .env 里的大写键。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM(OpenAI 兼容)----
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_context_window: int = 128000
    # 单价覆盖(元/1M tokens);不设则查内置单价表,再查不到记 cost=0(unknown)
    llm_price_prompt: float | None = None
    llm_price_completion: float | None = None

    # ---- Embedding ----
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    embed_dim: int = 1024  # 维度校验用,换模型必须改,防串库
    embed_batch_size: int = 64
    embed_query_prefix: str = ""  # 非对称模型可选,如 bge 的 query 前缀
    embed_passage_prefix: str = ""
    embed_price: float | None = None  # 元/1M tokens,覆盖内置表

    # ---- 并发与费用护栏 ----
    max_concurrency: int = 4  # API 并发信号量
    cost_limit_per_run: float = 50.0  # 单次命令费用上限(元),超出中断
    embed_batch_delay_ms: int = 0  # embedding 批间延迟,429 高发时调大

    # ---- 检索/切块关键参数(PLAN §5)----
    chunk_max_tokens: int = 512
    sim_threshold: float = 0.25
    vector_top_k: int = 20
    context_top_n: int = 8
    rerank_enabled: bool = False
    graph_expand_hops: int = 2


class RepoCfg(BaseModel):
    """repos.yaml 里单个仓库条目。"""

    name: str
    path: Path
    languages: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 内置单价表(参考价,元/1M tokens;以服务商账单为准,可用 .env 覆盖)
# 查找按"最长前缀匹配";查不到 → cost=0,usage_log 里以 cost=0 标记 unknown。
# ---------------------------------------------------------------------------

LLM_PRICES: dict[str, tuple[float, float]] = {
    # bigmodel(GLM 系列;flash 系列为免费档,如实际收费请用 .env 覆盖)
    "glm-5.3-flash": (0.0, 0.0),
    "glm-5.3": (1.0, 4.0),
    "glm-4.7": (0.6, 2.2),
    "glm-4.6": (0.6, 2.2),
    "glm-4.5": (0.6, 2.2),
    "glm-4-plus": (5.0, 5.0),
    "glm-4-air": (0.2, 0.8),
    "glm-4-flash": (0.0, 0.0),
    "glm-4.5-flash": (0.0, 0.0),
    # deepseek
    "deepseek-chat": (0.5, 2.0),
    "deepseek-reasoner": (1.0, 4.0),
    # openai(美元价按 ≈7.2 折算)
    "gpt-4o": (18.0, 72.0),
    "gpt-4o-mini": (1.05, 4.2),
}

EMBED_PRICES: dict[str, float] = {
    "embedding-3": 0.5,
    "embedding-2": 0.5,
    "bge-m3": 0.5,
    "text-embedding-3-small": 0.14,
    "text-embedding-3-large": 1.4,
}


def lookup_llm_price(model: str, s: Settings | None = None) -> tuple[float, float] | None:
    """返回 (输入单价, 输出单价) 元/1M tokens;.env 显式覆盖优先,否则最长前缀匹配。"""
    s = s or get_settings()
    if s.llm_price_prompt is not None and s.llm_price_completion is not None:
        return (s.llm_price_prompt, s.llm_price_completion)
    best: tuple[float, float] | None = None
    best_len = -1
    for prefix, prices in LLM_PRICES.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = prices, len(prefix)
    return best


def lookup_embed_price(model: str, s: Settings | None = None) -> float | None:
    """返回 embedding 单价 元/1M tokens;.env 显式覆盖优先。"""
    s = s or get_settings()
    if s.embed_price is not None:
        return s.embed_price
    best: float | None = None
    best_len = -1
    for prefix, price in EMBED_PRICES.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = price, len(prefix)
    return best


# ---------------------------------------------------------------------------
# 目录 / 哈希 / repos.yaml
# ---------------------------------------------------------------------------

_DOCS_README = """# data/docs —— 人工维护文档目录

这里存放手写文档(设计笔记、上线手册等),会与生成的 `data/wiki/` 一起入索引。
本目录归人所有,工具只读不写;生成类产物请勿放这里。
"""


def ensure_dirs() -> None:
    """创建运行时目录;首次创建时在 data/docs/ 放说明。"""
    for d in (DATA_DIR, LANCEDB_DIR, WIKI_DIR, DOCS_DIR, PROFILES_DIR, REPORTS_DIR, DDL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    readme = DOCS_DIR / "README.md"
    if not readme.exists():
        readme.write_text(_DOCS_README, encoding="utf-8")


def hash_bytes(data: bytes) -> str:
    """blake2b HEX 前 32 位(bytes 版;files.hash 对原始字节哈希,不依赖编码)。"""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def content_hash(text: str) -> str:
    """blake2b HEX 前 32 位。chunks.content_hash / embed_cache 共用。"""
    return hash_bytes(text.encode("utf-8"))


def load_repos(path: Path | None = None) -> list[RepoCfg]:
    """加载 repos.yaml;文件不存在返回空清单。"""
    p = path or REPOS_YAML
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [RepoCfg(**r) for r in raw.get("repos", [])]


@lru_cache
def get_settings() -> Settings:
    ensure_dirs()
    return Settings()
