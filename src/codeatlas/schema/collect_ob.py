"""画像采集(PLAN §9.8)。实际部署模式:无直连账号 → 人工回填。

- generate_scripts:生成只读 SELECT 脚本(行数/长度/索引基数/低基数列抽样),
  由用户或 DBA 在生产执行,结果按模板回填 JSON;
- manual_template / load_manual / apply_profile:模板生成与画像入库。

在线直连版(pymysql)留待有只读账号时补(用户已确认无法提供)。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from codeatlas.config import PROFILES_DIR
from codeatlas.schema.ddl import ParsedDDL

# 低基数嫌疑列名模式(枚举/状态/类型/标志)
_LOW_CARD_HINTS = ("status", "type", "flag", "state", "level", "kind", "code", "is_")


def _suspect_columns(t) -> list[str]:
    out = []
    for c in t.columns:
        low = c.name.lower()
        if any(h in low for h in _LOW_CARD_HINTS) and not c.is_pk:
            out.append(c.name)
    return out[:5]  # 每表最多抽 5 列,控制脚本规模


def generate_scripts(parsed: dict[str, ParsedDDL], out_dir: Path | None = None) -> list[Path]:
    """为每个库生成只读采集脚本。所有语句均为 SELECT,无任何写操作。"""
    out_dir = out_dir or PROFILES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for schema, p in parsed.items():
        lines = [
            f"-- codeatlas 画像采集脚本(只读 SELECT):{schema}",
            "-- 在生产库执行后将结果按 <schema>_manual.json 模板回填。",
            "",
            "-- 1) 表规模与长度(一次查全库)",
            "SELECT TABLE_NAME, TABLE_ROWS, DATA_LENGTH, INDEX_LENGTH, AUTO_INCREMENT",
            "FROM information_schema.TABLES WHERE TABLE_SCHEMA = '{s}';".format(s=schema),
            "",
            "-- 2) 索引基数(逐表执行 SHOW INDEX)",
        ]
        for t in p.tables:
            lines.append(f"SHOW INDEX FROM `{t.name}`;")
        lines += ["", "-- 3) 低基数嫌疑列抽样(逐条执行;大表请评估成本后执行)"]
        for t in p.tables:
            for col in _suspect_columns(t):
                lines.append(
                    f"SELECT `{col}`, COUNT(*) AS n FROM `{t.name}` "
                    f"GROUP BY `{col}` ORDER BY n DESC LIMIT 10;  -- {t.name}.{col}"
                )
        path = out_dir / f"{schema}_queries.sql"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def manual_template(schema: str, parsed: ParsedDDL, out_dir: Path | None = None) -> Path:
    """生成人工回填 JSON 模板(结构即约定;不确定的项保持 null)。"""
    out_dir = out_dir or PROFILES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl = {
        "schema": schema,
        "tables": {
            t.name: {
                "row_count": None,
                "data_length": None,
                "index_length": None,
                "auto_inc_value": None,
                "index_cardinality": {},  # 索引名 → 基数
                "column_stats": {
                    c.name: {"cardinality": None, "null_ratio": None, "top_values": None}
                    for c in t.columns
                    if c.name in _suspect_columns(t)
                },
            }
            for t in parsed.tables
        },
        "slow_queries": [
            {
                "sql": None, "freq": None, "avg_ms": None,
                "rows_scanned": None, "rows_returned": None,
            }
        ],
    }
    path = out_dir / f"{schema}_manual.json"
    path.write_text(
        json.dumps(tpl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_manual(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_profile(conn: sqlite3.Connection, schema: str, profile: dict) -> int:
    """画像写库(ddl_tables/columns/indexes 的统计字段)。返回更新的表数。"""
    updated = 0
    for tbl, info in (profile.get("tables") or {}).items():
        full = f"{schema}.{tbl}"
        row = conn.execute(
            "SELECT id FROM ddl_tables WHERE name=?", (full,)
        ).fetchone()
        if row is None:
            continue
        tid = row["id"]
        conn.execute(
            "UPDATE ddl_tables SET row_count=?, data_length=?, index_length=?, "
            "auto_inc_value=? WHERE id=?",
            (info.get("row_count"), info.get("data_length"),
             info.get("index_length"), info.get("auto_inc_value"), tid),
        )
        for idx_name, card in (info.get("index_cardinality") or {}).items():
            if card is not None:
                conn.execute(
                    "UPDATE ddl_indexes SET cardinality=? WHERE table_id=? AND name=?",
                    (int(card), tid, idx_name),
                )
        for col, stats in (info.get("column_stats") or {}).items():
            if not stats:
                continue
            tv = stats.get("top_values")
            conn.execute(
                "UPDATE ddl_columns SET cardinality=?, null_ratio=?, top_values=? "
                "WHERE table_id=? AND name=?",
                (
                    stats.get("cardinality"),
                    stats.get("null_ratio"),
                    json.dumps(tv, ensure_ascii=False) if tv else None,
                    tid,
                    col,
                ),
            )
        updated += 1
    conn.commit()
    return updated
