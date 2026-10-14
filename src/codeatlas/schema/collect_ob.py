"""画像采集(PLAN §9.8)。实际部署模式:无直连账号 → 人工回填。

- generate_scripts:生成只读 SELECT 脚本(行数/长度/索引基数/低基数列抽样),
  由用户或 DBA 在生产执行,结果按模板回填 JSON;
- manual_template / load_manual / apply_profile:模板生成与画像入库。

在线直连版(pymysql)留待有只读账号时补(用户已确认无法提供)。
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlglot

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


def _qi(name: str) -> str:
    """MySQL 反引号标识符(内部反引号双写转义)。"""
    return "`" + name.replace("`", "``") + "`"


def generate_scripts(parsed: dict[str, ParsedDDL], out_dir: Path | None = None) -> list[Path]:
    """为每个库生成只读采集脚本。所有语句均为 SELECT,无任何写操作。"""
    out_dir = out_dir or PROFILES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for schema, p in parsed.items():
        # 库名来自文件名、表/列名来自 DDL:一律转义后再拼,保证脚本确实只读
        lit = schema.replace("\\", "\\\\").replace("'", "''")
        lines = [
            f"-- codeatlas 画像采集脚本(只读 SELECT):{schema}",
            "-- 在生产库执行后将结果按 <schema>_manual.json 模板回填。",
            "",
            f"USE {_qi(schema)};",
            "",
            "-- 1) 表规模与长度(一次查全库)",
            "SELECT TABLE_NAME, TABLE_ROWS, DATA_LENGTH, INDEX_LENGTH, AUTO_INCREMENT",
            f"FROM information_schema.TABLES WHERE TABLE_SCHEMA = '{lit}';",
            "",
            "-- 2) 索引基数(逐表执行 SHOW INDEX)",
        ]
        for t in p.tables:
            lines.append(f"SHOW INDEX FROM {_qi(t.name)};")
        lines += ["", "-- 3) 低基数嫌疑列抽样(逐条执行;大表请评估成本后执行)"]
        for t in p.tables:
            for col in _suspect_columns(t):
                lines.append(
                    f"SELECT {_qi(col)}, COUNT(*) AS n FROM {_qi(t.name)} "
                    f"GROUP BY {_qi(col)} ORDER BY n DESC LIMIT 10;"
                )
        path = out_dir / f"{schema}_queries.sql"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def manual_template(schema: str, parsed: ParsedDDL, out_dir: Path | None = None) -> Path:
    """生成人工回填 JSON 模板(结构即约定;不确定的项保持 null)。

    文件已存在时只补新表/新键,已填的值、导入的慢查询一律保留(重跑 collect 不丢数据)。
    """
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
    if path.exists():
        existing = load_manual(path)
        tables = existing.setdefault("tables", {})
        for name, fresh in tpl["tables"].items():
            cur = tables.setdefault(name, {})
            for k, v in fresh.items():
                cur.setdefault(k, v)
        existing.setdefault("schema", schema)
        existing.setdefault("slow_queries", [])
        tpl = existing
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
    out = out_dir / "fill_sheet.md"
    # 重跑时保留已填的格子(按 库/表 对齐);新增表追加空行
    filled = _read_sheet_cells(out.read_text(encoding="utf-8")) if out.exists() else {}
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
            cells = filled.get((schema, tb.name)) or [""] * 5
            cells = (cells + [""] * 5)[:5]
            lines.append(f"| {tb.name} |" + "".join(f" {c} |" if c else "  |" for c in cells))
        lines.append("")
    lines += [
        "# 慢查询不再填本表:请单独导入文件(CSV/JSON/纯文本 SQL 均可):",
        "# uv run atlas collect --import-slow <文件> [--schema 库名]",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _read_sheet_cells(text: str) -> dict[tuple[str, str], list[str]]:
    """填报表 → {(库, 表): [行数, 数据长度, 索引长度, 自增值, 备注]}(原样字符串)。"""
    out: dict[tuple[str, str], list[str]] = {}
    section: str | None = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(\S+)\s*$", line.strip())
        if m:
            section = m.group(1)
            continue
        if section is None or not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: ") or cells[0] in ("表名", "库"):
            continue
        out[(section, cells[0])] = cells[1:]
    return out


def import_fill_sheet(path: Path, parsed: dict, out_dir: Path | None = None) -> list[Path]:
    """解析填好的 md → 各库 *_manual.json(与人工 JSON 模板同构,audit 直接可用)。

    按字段合并:只写填了数的格子,不动该表的 index_cardinality/column_stats 等其它键。
    """
    out_dir = out_dir or PROFILES_DIR
    written: list[Path] = []
    fills: dict[str, dict[str, dict]] = {}
    keys = ("row_count", "data_length", "index_length", "auto_inc_value")
    for (schema, table), cells in _read_sheet_cells(path.read_text(encoding="utf-8")).items():
        if "慢查询" in schema or schema not in parsed:
            continue  # 慢查询走 --import-slow 文件通道
        if table not in {t.name for t in parsed[schema].tables}:
            continue
        vals = {k: _num(c) for k, c in zip(keys, cells)}
        vals = {k: v for k, v in vals.items() if v is not None}
        if vals:
            fills.setdefault(schema, {})[table] = vals
    for schema, tables in fills.items():
        mp = out_dir / f"{schema}_manual.json"
        tpl = load_manual(mp) if mp.exists() else {
            "schema": schema, "tables": {}, "slow_queries": []
        }
        for table, vals in tables.items():
            tpl.setdefault("tables", {}).setdefault(table, {}).update(vals)
        mp.write_text(json.dumps(tpl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(mp)
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
    """宽容解析:JSON(数组 / {"slow_queries": [...]} / 每行一个对象)/ CSV(含 sql 列,
    支持 Excel 导出的 BOM 与引号内换行)/ 纯文本(按空行或行尾分号分条,无统计字段)。"""
    import csv
    import io

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.strip():
        return []
    suffix = path.suffix.lower()
    if suffix in (".json", ".jsonl", ".ndjson") or text.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(text)
            items = data.get("slow_queries") if isinstance(data, dict) else data
        except json.JSONDecodeError:  # NDJSON:每行一个对象
            items = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        rows = [_slow_row(it.get("schema"), it.get("sql") or it.get("query"), it)
                for it in items or [] if isinstance(it, dict)]
        return [r for r in rows if r["sql"]]
    header = [c.strip().lower() for c in next(csv.reader([text.splitlines()[0]]))]
    if suffix == ".csv" or "sql" in header:
        rows = []
        for it in csv.DictReader(io.StringIO(text)):
            low = {k.strip().lower(): (v or "").strip() for k, v in it.items() if k}
            if not low.get("sql"):
                continue
            rows.append(_slow_row(
                low.get("schema") or low.get("库") or low.get("库名"), low.get("sql"), {
                    "freq": low.get("freq") or low.get("频次"),
                    "avg_ms": low.get("avg_ms") or low.get("平均耗时ms") or low.get("耗时ms"),
                    "rows_scanned": low.get("rows_scanned") or low.get("扫描行数"),
                    "rows_returned": low.get("rows_returned") or low.get("返回行数"),
                }))
        return rows
    # 纯文本:按空行或行尾分号分条
    rows = []
    for block in re.split(r";[ \t]*(?:\n|$)|\n\s*\n", text):
        sql = block.strip()
        if sql and any(k in sql.upper() for k in ("SELECT", "INSERT", "UPDATE", "DELETE")):
            rows.append(_slow_row(None, sql, {}))
    return rows


def _slow_row(schema, sql, it: dict) -> dict:
    def num(k, *alts):
        for key in (k, *alts):
            v = it.get(key)
            if v not in (None, ""):
                return _num(str(v))
        return None

    return {
        "schema": schema, "sql": sql or "",
        "freq": num("freq"), "avg_ms": num("avg_ms", "avg_latency_ms"),
        "rows_scanned": num("rows_scanned"), "rows_returned": num("rows_returned"),
    }


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
    try:
        st = sqlglot.parse_one(sql, dialect="mysql")
    except sqlglot.errors.ParseError:
        st = None
    if st is not None:
        return {t_.name.lower() for t_ in st.find_all(sqlglot.exp.Table) if t_.name}
    # 解析不了才退回正则:库名限定 db.t 取表名段
    pat = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+`?(\w+)`?(?:\.`?(\w+)`?)?", re.I)
    return {(t or s_).lower() for s_, t in pat.findall(sql)}
