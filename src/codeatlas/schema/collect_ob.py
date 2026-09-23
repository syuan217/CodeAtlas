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

# ---------------------------------------------------------------------------
# 人工填报表(md)与导入
# ---------------------------------------------------------------------------

def generate_fill_sheet(parsed: dict, out_dir: Path | None = None) -> Path:
    """生成单文件汇总填报表:每库一节,表级规模列 + 慢查询段。

    体检规则一致的排除约定(temp/tmp)同样适用于此处。
    """
    out_dir = out_dir or PROFILES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OB 表规模画像填报表",
        "",
        "> 只填数字列,留空 = 无数据(相关统计规则自动跳过该表,不猜)。",
        "> 数据来源:information_schema.TABLES / 监控平台 / DBA 口头数据均可,",
        "> 单位:行数=行;长度=MB。重点表优先,不必全填。填完运行:",
        "> `uv run atlas collect --import <本文件路径>`",
        "",
    ]
    for schema, p_ in parsed.items():
        lines.append(f"## {schema}")
        lines.append("")
        lines.append("| 表名 | 行数 | 数据长度MB | 索引长度MB | 自增值 | 备注 |")
        lines.append("|---|---|---|---|---|---|")
        for tb in p_.tables:
            low = tb.name.lower()
            if "temp" in low or "tmp" in low:
                continue
            lines.append(f"| {tb.name} |  |  |  |  |  |")
        lines.append("")
    lines += [
        "# 慢查询不再填本表:请单独导入文件(CSV/JSON/纯文本 SQL 均可):",
        "# uv run atlas collect --import-slow <文件> [--schema 库名]",
    ]
    out = out_dir / "fill_sheet.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def import_fill_sheet(path: Path, parsed: dict, out_dir: Path | None = None) -> list[Path]:
    """解析填好的 md → 各库 *_manual.json(与人工 JSON 模板同构,audit 直接可用)。"""
    import re as _re

    out_dir = out_dir or PROFILES_DIR
    text = path.read_text(encoding="utf-8")
    written: list[Path] = []
    section: str | None = None
    # 表行:{schema: {table: {row_count, data_length, index_length, auto_inc}}}
    fills: dict[str, dict[str, dict]] = {}
    slow_rows: list[dict] = []
    for line in text.splitlines():
        m = _re.match(r"^##\s+(\S+)\s*$", line.strip())
        if m:
            section = m.group(1)
            continue
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: ") or cells[0] in ("表名", "库"):
            continue
        if section and "慢查询" in section:
            continue  # 已废弃:慢查询走 --import-slow 文件通道
        schema = section
        if schema not in parsed:
            continue
        table = cells[0]
        if table not in {t.name for t in parsed[schema].tables}:
            continue
        entry = fills.setdefault(schema, {}).setdefault(table, {})
        entry["row_count"] = _num(cells[1]) if len(cells) > 1 else None
        entry["data_length"] = _num(cells[2]) if len(cells) > 2 else None
        entry["index_length"] = _num(cells[3]) if len(cells) > 3 else None
        entry["auto_inc_value"] = _num(cells[4]) if len(cells) > 4 else None
    for schema, tables in fills.items():
        if not tables:
            continue
        tpl = json.loads(
            (out_dir / f"{schema}_manual.json").read_text(encoding="utf-8")
        ) if (out_dir / f"{schema}_manual.json").exists() else {
            "schema": schema, "tables": {}, "slow_queries": []
        }
        tpl["tables"].update(tables)
        by_schema_slow = [s for s in slow_rows if s["schema"] == schema]
        if by_schema_slow:
            tpl["slow_queries"] = by_schema_slow
        out = out_dir / f"{schema}_manual.json"
        out.write_text(
            json.dumps(tpl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(out)
    return written


def _num(v: str):
    if v is None:
        return None
    v = v.replace(",", "").strip()
    if not v:
        return None
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 慢查询文件导入(CSV / JSON / 纯文本 SQL)
# ---------------------------------------------------------------------------

def parse_slow_file(path: Path) -> list[dict]:
    """宽容解析:JSON(数组或 {"slow_queries": [...]})/ CSV(含 sql 列)/
    纯文本(按空行或行尾分号分条,无统计字段)。"""
    import csv
    import re as _re_mod

    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    rows: list[dict] = []
    if suffix == ".json" or text.lstrip().startswith(("{", "[")):
        data = json.loads(text)
        items = data.get("slow_queries") if isinstance(data, dict) else data
        for it in items or []:
            rows.append({
                "schema": it.get("schema"), "sql": it.get("sql") or it.get("query") or "",
                "freq": _num(str(it.get("freq") or "")),
                "avg_ms": _num(str(it.get("avg_ms") or it.get("avg_latency_ms") or "")),
                "rows_scanned": _num(str(it.get("rows_scanned") or "")),
                "rows_returned": _num(str(it.get("rows_returned") or "")),
            })
        return [r for r in rows if r["sql"]]
    if suffix == ".csv" or ("," in text.splitlines()[0] and "sql" in text.splitlines()[0].lower()):
        reader = csv.DictReader(text.splitlines())
        for it in reader:
            low = {k.strip().lower(): (v or "").strip() for k, v in it.items() if k}
            if not low.get("sql"):
                continue
            rows.append({
                "schema": low.get("schema") or low.get("库") or low.get("库名"),
                "sql": low.get("sql"),
                "freq": _num(low.get("freq") or low.get("频次") or ""),
                "avg_ms": _num(low.get("avg_ms") or low.get("平均耗时ms") or low.get("耗时ms") or ""),
                "rows_scanned": _num(low.get("rows_scanned") or low.get("扫描行数") or ""),
                "rows_returned": _num(low.get("rows_returned") or low.get("返回行数") or ""),
            })
        return rows
    # 纯文本:按空行分条(行尾分号不强求)
    for block in _re_mod.split(r"\n\s*\n", text):
        sql = block.strip()
        if sql and any(k in sql.upper() for k in ("SELECT", "INSERT", "UPDATE", "DELETE")):
            rows.append({"schema": None, "sql": sql, "freq": None,
                         "avg_ms": None, "rows_scanned": None, "rows_returned": None})
    return rows


def import_slow_queries(
    path: Path, parsed: dict, out_dir: Path | None = None,
    default_schema: str | None = None,
) -> dict[str, int]:
    """慢查询文件 → 各库 *_manual.json 的 slow_queries 段(合并去重)。

    库归属:条目 schema 字段 > --schema 参数 > 按 SQL 中表名唯一命中某库。
    返回 {schema: 导入条数}。
    """
    out_dir = out_dir or PROFILES_DIR
    entries = parse_slow_file(path)
    # 库表索引:表名 → 命中的库集合
    tables_by_schema = {
        s: {t.name for t in p_.tables} for s, p_ in parsed.items()
    }
    assigned: dict[str, list[dict]] = {}
    for e in entries:
        schema = e.get("schema") or default_schema
        if not schema:
            hit = {
                s for s, tbls in tables_by_schema.items()
                if any(t.lower() in tbls for t in _sql_tables(e["sql"]))
            }
            if len(hit) == 1:
                schema = next(iter(hit))
        if not schema:
            continue  # 无法归属,丢弃(可加 --schema 重导)
        assigned.setdefault(schema, []).append(e)
    counts: dict[str, int] = {}
    for schema, items in assigned.items():
        if schema not in parsed:
            continue
        mp = out_dir / f"{schema}_manual.json"
        data = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {
            "schema": schema, "tables": {}, "slow_queries": []
        }
        seen = {json.dumps(x, sort_keys=True) for x in data.get("slow_queries") or []}
        for it in items:
            key = json.dumps(it, sort_keys=True)
            if key not in seen:
                data.setdefault("slow_queries", []).append(it)
                seen.add(key)
        mp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
        counts[schema] = len(items)
    return counts


def _sql_tables(sql: str) -> set[str]:
    import re as _re

    try:
        st = sqlglot.parse_one(sql, dialect="mysql")
        return {t_.name.lower() for t_ in st.find_all(sqlglot.exp.Table) if t_.name}
    except Exception:
        pat = _re.compile(r"(?:FROM|JOIN|INTO|UPDATE)\s+[`]?([a-zA-Z_][a-zA-Z0-9_]*)", _re.I)
        return {m.lower() for m in pat.findall(sql)}
