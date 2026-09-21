"""SQLite schema 与连接管理(WAL)。

DDL 与 PLAN §7 完全一致,是已确认决策,勿擅自增删列;
变更需求先与用户确认,并升 SCHEMA_VERSION 写迁移。
FTS5 的 chunks→chunks_fts 同步策略为"显式维护"(M1 的 db/fts.py 负责),不用触发器。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codeatlas import config

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- 存 schema_version 等

CREATE TABLE IF NOT EXISTS repos(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  path TEXT NOT NULL,
  languages TEXT NOT NULL,              -- JSON 数组,如 '["java","xml"]'
  indexed_commit TEXT,                  -- git 基线(增量锚点)
  last_indexed_at TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  path TEXT NOT NULL,                   -- 相对仓库根
  hash TEXT,                            -- blake2b(content),HEX 前 32 位
  mtime REAL,
  parse_status TEXT DEFAULT 'pending',  -- pending/ok/failed/skipped(binary|too_large)
  last_seen_commit TEXT,                -- 最后一次确认存在的 commit
  UNIQUE(repo_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id, parse_status);

CREATE TABLE IF NOT EXISTS symbols(
  id INTEGER PRIMARY KEY,
  repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,                   -- module/class/function/method/interface
  name TEXT NOT NULL,
  qualified_name TEXT NOT NULL,         -- 如 com.x.order.OrderService#create
  line_start INTEGER, line_end INTEGER, -- 1-based,含签名行
  signature TEXT                        -- 签名行文本(切块/展示用)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sym_qname ON symbols(repo_id, qualified_name);
CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(repo_id, name);   -- 同名兜底解析用

CREATE TABLE IF NOT EXISTS edges(
  id INTEGER PRIMARY KEY,
  repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  src_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  dst_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,                   -- CONTAINS / IMPORTS / CALLS
  resolution TEXT,                      -- CALLS 边专用:exact / heuristic
  line INTEGER, col INTEGER,            -- 调用点位置(CALLS 边)
  UNIQUE(repo_id, src_id, dst_id, kind, line, col)   -- 幂等去重,重索引安全
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(repo_id, src_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(repo_id, dst_id, kind);

CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
  symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,                   -- code / doc / wiki / report
  title TEXT,                           -- 展示用:符号 qualified_name 或文档标题
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,           -- blake2b(content)
  line_start INTEGER, line_end INTEGER,
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_dedup
  ON chunks(repo_id, kind, file_id, coalesce(symbol_id,0), content_hash);

CREATE TABLE IF NOT EXISTS vector_refs(              -- document_vectors 思路:引用归 SQL 管
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  lance_id TEXT NOT NULL,              -- LanceDB 中的行 id
  model TEXT NOT NULL,
  embedded_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embed_cache(              -- 向量缓存:内容哈希命中跳过重新 embedding
  content_hash TEXT NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,                -- float32 little-endian 定长数组
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY(content_hash, model)
);

-- ===== 体检(M4)=====
CREATE TABLE IF NOT EXISTS ddl_tables(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  ddl_text TEXT NOT NULL,
  -- 画像补充(collect 写入)
  row_count INTEGER, data_length INTEGER, index_length INTEGER,
  read_ratio REAL, write_ratio REAL, auto_inc_value INTEGER
);
CREATE TABLE IF NOT EXISTS ddl_columns(
  id INTEGER PRIMARY KEY,
  table_id INTEGER NOT NULL REFERENCES ddl_tables(id) ON DELETE CASCADE,
  name TEXT NOT NULL, data_type TEXT, nullable INTEGER,
  is_pk INTEGER DEFAULT 0,
  cardinality INTEGER,                 -- 区分度(collect 写入)
  null_ratio REAL, top_values TEXT     -- top_values: JSON [{'v':..,'n':..}]
);
CREATE TABLE IF NOT EXISTS ddl_indexes(
  id INTEGER PRIMARY KEY,
  table_id INTEGER NOT NULL REFERENCES ddl_tables(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  columns TEXT NOT NULL,               -- JSON 数组(有序)
  is_unique INTEGER DEFAULT 0,
  cardinality INTEGER
);
CREATE TABLE IF NOT EXISTS query_column_map(         -- 代码 SQL → 表列 的访问路径
  id INTEGER PRIMARY KEY,
  repo_id INTEGER NOT NULL,
  query_fingerprint TEXT NOT NULL,     -- 归一化 SQL 指纹
  source_file TEXT NOT NULL, source_line INTEGER,
  table_name TEXT NOT NULL, column_name TEXT NOT NULL,
  usage TEXT NOT NULL,                 -- where / join / order / group
  freq INTEGER DEFAULT 1               -- 同指纹出现次数
);

CREATE TABLE IF NOT EXISTS audit_findings(
  id INTEGER PRIMARY KEY,
  table_name TEXT NOT NULL,
  rule_id TEXT NOT NULL,               -- 如 IDX001/IDX002...
  severity TEXT NOT NULL,              -- high/medium/low/info
  evidence TEXT NOT NULL,              -- 证据:代码位置+查询指纹+统计数字(JSON)
  suggestion TEXT,                     -- 建议含完整 DDL
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS usage_log(                -- 费用流水(表名避开 SQLite 关键字 usage)
  id INTEGER PRIMARY KEY,
  ts TEXT DEFAULT (datetime('now')),
  stage TEXT NOT NULL,                 -- index/ask/wiki/audit/doctor
  repo_id INTEGER,
  model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,
  cost REAL NOT NULL
);

-- FTS5(外部内容表;chunks 增删改由 indexer 显式同步,不用触发器)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, content='chunks', content_rowid='id');
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """打开(必要时创建)库文件:WAL + 外键 + busy 超时。"""
    p = Path(path) if path is not None else config.DB_PATH  # 动态读,便于测试重定向
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表(幂等)并记录 schema_version。"""
    conn.executescript(DDL)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
