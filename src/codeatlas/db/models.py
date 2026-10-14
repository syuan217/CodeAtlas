"""SQLite schema 与连接管理(WAL)。

DDL 与 PLAN §7 完全一致,是已确认决策,勿擅自增删列;
变更需求先与用户确认,并升 SCHEMA_VERSION 写迁移。
FTS5 的 chunks→chunks_fts 同步策略为"显式维护"(M1 的 db/fts.py 负责),不用触发器。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codeatlas import config

SCHEMA_VERSION = 2

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
-- 非唯一:多模块 Maven 工程同 FQCN、src/util.ts 与 src/util.js 同模块名都真实存在;
-- 同文件内重名(重载)由 symbols.dedup_qnames 加 @行号 消歧,跨文件重名由解析器按歧义处理
CREATE INDEX IF NOT EXISTS idx_sym_qname ON symbols(repo_id, qualified_name);
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
-- 单列索引:ON DELETE CASCADE 与 callers/impact 按 src_id/dst_id 查找用不上上面
-- 以 repo_id 打头的复合索引,缺它们时每删一个符号就全表扫 edges/chunks
CREATE INDEX IF NOT EXISTS idx_edges_src_id ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst_id ON edges(dst_id);

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
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_id);

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

-- FTS5(无内容表:索引 db/fts.fts_text 预处理后的文本——中文二元组 + 驼峰拆分;
-- 原文只在 chunks。增删由 indexer 等显式同步,不用触发器)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, content='');
"""


# IN (?,?,…) 单条语句的参数个数上限:SQLite 默认 SQLITE_MAX_VARIABLE_NUMBER=32766,
# 大仓库首索引的路径/哈希清单会超,统一按此分片查询
IN_CHUNK = 500


def in_chunks(items: list, size: int = IN_CHUNK):
    """按 size 切片,供 IN 查询分批绑定参数。"""
    for i in range(0, len(items), size):
        yield items[i:i + size]


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


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """按版本顺序升级已有库(新库建表即最新,不走这里)。"""
    if from_version < 2:
        # v2:FTS 改为无内容表 + 预处理文本(中文二元组/驼峰拆分),并修复
        # wiki/report/doc chunk 此前未入 FTS、重跑对未入索引的行发 'delete' 可能写坏索引
        # 的问题 → 删表按新定义重建,再按 chunks 全量重灌
        from codeatlas.db.fts import fts_rebuild

        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content='')")
        fts_rebuild(conn)
        # v2:idx_sym_qname 由 UNIQUE 改为普通索引(跨文件同限定名曾撞约束中断整次索引)
        conn.execute("DROP INDEX IF EXISTS idx_sym_qname")
        conn.execute("CREATE INDEX idx_sym_qname ON symbols(repo_id, qualified_name)")


def init_db(conn: sqlite3.Connection) -> None:
    """建表(幂等)、按需迁移,并记录 schema_version。"""
    conn.executescript(DDL)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is not None and int(row[0]) < SCHEMA_VERSION:
        _migrate(conn, int(row[0]))
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
