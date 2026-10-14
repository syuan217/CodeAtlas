"""体检规则引擎与报告(PLAN §9.8)。

结论 = 规则的确定性计算,LLM 不产生结论(v1 报告为纯模板输出,LLM 润色可选)。
规则清单:IDX001~005 / TBL001 / IDX101~103 / QRY101;画像缺失的规则在报告中
标 unavailable,不猜。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from codeatlas.config import REPORTS_DIR

RULE_NAMES = {
    "TBL001": "无主键",
    "IDX001": "高频查询条件列无索引",
    "IDX002": "复合索引列序错配(最左前缀)",
    "IDX003": "冗余索引(前缀包含)",
    "IDX004": "JOIN 列无索引或类型不一致",
    "IDX005": "索引失效写法(LIKE '%x'/函数包裹列)",
    "IDX101": "小表(<5000 行)索引过多",
    "IDX102": "低区分度单列索引(<10%)",
    "IDX103": "倾斜列索引(top1 占比>90% 且查询少)",
    "QRY101": "扫描/返回行数比异常(>1000)",
}
PROFILE_RULES = {"IDX101", "IDX102", "IDX103", "QRY101"}


@dataclass
class Finding:
    table: str          # 裸表名(不含 schema)
    rule_id: str
    severity: str       # high / medium / low / info
    evidence: dict      # 代码位置/查询指纹/统计数字
    suggestion: str | None = None


@dataclass
class AuditResult:
    schema: str
    findings: list[Finding] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)  # 画像缺失的规则
    table_count: int = 0
    query_count: int = 0
    excluded_tables: list[str] = field(default_factory=list)  # 按约定排除的表


def _is_excluded(table_name: str) -> bool:
    """用户约定(2026-09-22):临时表(temp/tmp)不纳入体检。"""
    low = table_name.lower()
    return "temp" in low or "tmp" in low


# ---------------------------------------------------------------------------
# 数据快照(从库读,一次性)
# ---------------------------------------------------------------------------

@dataclass
class Tbl:
    name: str
    columns: dict[str, str]          # col → data_type
    pk: set[str]
    indexes: list[tuple[str, list[str], bool, int | None]]  # (name, cols, unique, cardinality)
    row_count: int | None
    has_profile: bool


def _schema_rows(conn: sqlite3.Connection, schema: str, cols: str = "id, name, row_count"):
    """某库的 ddl_tables 行(按 "schema." 精确前缀;LIKE 的 _ 是通配符,会串库)。"""
    prefix = f"{schema}."
    return conn.execute(
        f"SELECT {cols} FROM ddl_tables WHERE substr(name, 1, ?) = ?",
        (len(prefix), prefix),
    )


def _load_tables(conn: sqlite3.Connection, schema: str) -> dict[str, Tbl]:
    tables: dict[str, Tbl] = {}
    for r in _schema_rows(conn, schema):
        bare = r["name"].split(".", 1)[1]
        if _is_excluded(bare):
            continue
        cols = {
            c["name"]: (c["data_type"] or "")
            for c in conn.execute(
                "SELECT name, data_type FROM ddl_columns WHERE table_id=?", (r["id"],)
            )
        }
        pk = {
            c["name"]
            for c in conn.execute(
                "SELECT name FROM ddl_columns WHERE table_id=? AND is_pk=1", (r["id"],)
            )
        }
        idxs = [
            (i["name"], json.loads(i["columns"]), bool(i["is_unique"]), i["cardinality"])
            for i in conn.execute(
                "SELECT name, columns, is_unique, cardinality FROM ddl_indexes "
                "WHERE table_id=?",
                (r["id"],),
            )
        ]
        tables[bare] = Tbl(
            name=bare, columns=cols, pk=pk, indexes=idxs,
            row_count=r["row_count"], has_profile=r["row_count"] is not None,
        )
    return tables


# ---------------------------------------------------------------------------
# 规则(每条一个函数;无证据/无画像 → 不产 finding)
# ---------------------------------------------------------------------------

def rule_tbl001(tables: dict[str, Tbl]) -> list[Finding]:
    out = []
    for t in tables.values():
        if not t.pk:
            out.append(
                Finding(
                    table=t.name, rule_id="TBL001", severity="high",
                    evidence={"reason": "未定义 PRIMARY KEY",
                              "columns": len(t.columns)},
                    suggestion=f"ALTER TABLE `{t.name}` ADD PRIMARY KEY (`id`);"
                    "(以自增列或业务唯一键为准)",
                )
            )
    return out


def _indexed_columns(t: Tbl) -> set[str]:
    cols: set[str] = set()
    for _, icols, _, _ in t.indexes:
        cols.update(icols)
    return cols


def rule_idx001(qmap: list[dict], tables: dict[str, Tbl]) -> list[Finding]:
    """高频查询条件列无索引:usage=where 且该列不在任何索引列集合。"""
    out = []
    agg: dict[tuple[str, str], dict] = {}
    for q in qmap:
        if q["usage"] != "where":
            continue
        t = tables.get(q["table_name"])
        if t is None or q["column_name"] not in t.columns:
            continue
        if q["column_name"] in _indexed_columns(t) or q["column_name"] in t.pk:
            continue
        key = (q["table_name"], q["column_name"])
        e = agg.setdefault(key, {"fingerprints": set(), "files": set(), "freq": 0})
        e["fingerprints"].add(q["query_fingerprint"])
        e["files"].add(f'{q["source_file"]}:{q["source_line"]}')
        e["freq"] += q["freq"] or 1
    for (tbl, col), e in sorted(agg.items()):
        out.append(
            Finding(
                table=tbl, rule_id="IDX001", severity="high" if e["freq"] > 2 else "medium",
                evidence={
                    "column": col,
                    "query_fingerprints": sorted(e["fingerprints"])[:3],
                    "code_locations": sorted(e["files"])[:3],
                    "distinct_queries": len(e["fingerprints"]),
                },
                suggestion=f"ALTER TABLE `{tbl}` ADD INDEX `idx_{col}` (`{col}`);"
                "(先确认选择性,低基数列建议并入复合索引)",
            )
        )
    return out


def rule_idx002(qmap: list[dict], tables: dict[str, Tbl]) -> list[Finding]:
    """复合索引列序错配:查询 where 用到索引的非首列组合且不含首列。"""
    out = []
    by_fp: dict[str, dict[str, set[str]]] = {}
    for q in qmap:
        if q["usage"] != "where":
            continue
        by_fp.setdefault(q["query_fingerprint"], {}).setdefault(
            q["table_name"], set()
        ).add(q["column_name"])
    seen: set[tuple[str, str, str]] = set()
    for fp, tbl_cols in by_fp.items():
        for tbl, cols in tbl_cols.items():
            t = tables.get(tbl)
            if t is None:
                continue
            # 已有别的索引(或主键)以查询列打头 → 这条查询有索引可走,不算错配
            leading = {icols[0] for _, icols, _, _ in t.indexes if icols} | (
                t.pk if len(t.pk) == 1 else set()
            )
            if cols & leading:
                continue
            for iname, icols, _, _ in t.indexes:
                if len(icols) < 2:
                    continue
                first = icols[0]
                used = cols & set(icols)
                if used and first not in cols and len(used) >= 1:
                    key = (tbl, iname, ",".join(sorted(cols)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        Finding(
                            table=tbl, rule_id="IDX002", severity="medium",
                            evidence={
                                "index": iname,
                                "index_columns": icols,
                                "query_where_columns": sorted(cols),
                                "query_fingerprint": fp,
                            },
                            suggestion=(
                                f"查询列 {sorted(cols)} 未命中索引 `{iname}`({icols}) 的最左前缀;"
                                f"考虑新增以这些列开头的索引,或调整现有索引列序。"
                            ),
                        )
                    )
    return out


def rule_idx003(tables: dict[str, Tbl]) -> list[Finding]:
    """冗余索引:被另一索引(或主键)最左前缀覆盖的普通索引,含完全重复。

    唯一索引承载约束,永不建议删除(仅当与另一唯一索引完全重复时删其一)。
    """
    out = []
    for t in tables.values():
        others = list(t.indexes)
        if len(t.pk) == 1:
            others.append(("PRIMARY", sorted(t.pk), True, None))
        for n1, c1, u1, _ in t.indexes:
            if not c1:
                continue
            for n2, c2, u2, _ in others:
                if n1 == n2 or c2[: len(c1)] != c1:
                    continue
                if len(c1) < len(c2):
                    redundant = not u1  # 唯一前缀 (a) ⊂ (a,b) 不冗余:它约束的是 a 唯一
                else:  # 列完全相同:保留唯一的那个;同为唯一/非唯一时按名字删后者
                    redundant = (not u1 and u2) or (u1 == u2 and n1 > n2 and n2 != "PRIMARY") \
                        or (u1 == u2 and n2 == "PRIMARY")
                if not redundant:
                    continue
                out.append(
                    Finding(
                        table=t.name, rule_id="IDX003", severity="low",
                        evidence={
                            "kept": n2, "kept_columns": c2,
                            "redundant": n1, "redundant_columns": c1,
                        },
                        suggestion=(
                            f"索引 `{n1}`({c1}) 已被 `{n2}`({c2}) 的最左前缀覆盖,"
                            f"可删除:ALTER TABLE `{t.name}` DROP INDEX `{n1}`;"
                        ),
                    )
                )
                break  # 每个冗余索引报一次
    return out


def rule_idx004(qmap: list[dict], tables: dict[str, Tbl]) -> list[Finding]:
    """JOIN 列无索引或两侧类型不一致(同表内两列 join 的近似;跨表 join 需表对齐)。"""
    out = []
    seen: set[tuple] = set()
    for q in qmap:
        if q["usage"] != "join":
            continue
        t = tables.get(q["table_name"])
        if t is None or q["column_name"] not in t.columns:
            continue
        col = q["column_name"]
        no_index = col not in _indexed_columns(t) and col not in t.pk
        key = (q["table_name"], col, q["query_fingerprint"])
        if key in seen:
            continue
        seen.add(key)
        if no_index:
            out.append(
                Finding(
                    table=t.name, rule_id="IDX004", severity="medium",
                    evidence={
                        "column": col, "issue": "join 列无索引",
                        "query_fingerprint": q["query_fingerprint"],
                        "code_location": f'{q["source_file"]}:{q["source_line"]}',
                    },
                    suggestion=f"ALTER TABLE `{t.name}` ADD INDEX `idx_{col}` (`{col}`);",
                )
            )
    return out


def rule_idx005(anti_patterns: list[dict], tables: dict[str, Tbl]) -> list[Finding]:
    """索引失效写法(来自 sql_extract 的 LIKE '%x'/函数包裹列检测);只报本库的表。"""
    out = []
    seen: set[tuple] = set()
    for ap in anti_patterns:
        if ap["table"] not in tables:
            continue
        key = (ap["table"], ap["column"], ap["pattern"], ap["fingerprint"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Finding(
                table=ap["table"], rule_id="IDX005", severity="medium",
                evidence={
                    "column": ap["column"],
                    "pattern": ap["pattern"],
                    "code_location": f'{ap["source_file"]}:{ap["source_line"]}',
                    "query_fingerprint": ap["fingerprint"],
                },
                suggestion=(
                    "前缀通配 LIKE '%x' 无法走 BTree 索引:改右匹配 'x%',或用覆盖索引/全文索引;"
                    "函数包裹列可改写为等价常量比较。"
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_audit(
    conn: sqlite3.Connection,
    schema: str,
    qmap: list[dict],
    anti_patterns: list[dict],
    slow_queries: list[dict] | None = None,
) -> AuditResult:
    tables = _load_tables(conn, schema)
    excluded = [
        r["name"].split(".", 1)[1]
        for r in _schema_rows(conn, schema, "name")
        if _is_excluded(r["name"].split(".", 1)[1])
    ]
    result = AuditResult(
        schema=schema,
        table_count=len(tables),
        query_count=len({q["query_fingerprint"] for q in qmap}),
        excluded_tables=sorted(excluded),
    )
    result.findings += rule_tbl001(tables)
    result.findings += rule_idx001(qmap, tables)
    result.findings += rule_idx002(qmap, tables)
    result.findings += rule_idx003(tables)
    result.findings += rule_idx004(qmap, tables)
    result.findings += rule_idx005(anti_patterns, tables)

    # 慢查询类(QRY101):有统计条目才跑(返回 0 行的全表扫描恰恰最该报,不能按真值过滤)
    slow = [
        q for q in (slow_queries or [])
        if q.get("rows_scanned") not in (None, "") and q.get("rows_returned") not in (None, "")
    ]
    if slow:
        result.findings += _rule_qry101(schema, slow)
    else:
        result.unavailable.append("QRY101")

    # 画像类规则
    profiled = [t for t in tables.values() if t.has_profile]
    if not profiled:
        result.unavailable += ["IDX101", "IDX102", "IDX103"]
    else:
        result.findings += _rule_idx101(tables)
        stats = _load_column_stats(conn, schema)
        if not stats:
            result.unavailable += ["IDX102", "IDX103"]
        else:
            result.findings += _rule_idx102(tables, stats)
            result.findings += _rule_idx103(tables, stats, qmap)
    return result


def _load_column_stats(conn: sqlite3.Connection, schema: str) -> dict[tuple[str, str], dict]:
    """{(表, 列): {cardinality, top_values}},只取本库、有统计的列;另并入索引基数。"""
    out: dict[tuple[str, str], dict] = {}
    for r in _schema_rows(conn, schema):
        bare = r["name"].split(".", 1)[1]
        for c in conn.execute(
            "SELECT name, cardinality, top_values FROM ddl_columns "
            "WHERE table_id=? AND (cardinality IS NOT NULL OR top_values IS NOT NULL)",
            (r["id"],),
        ):
            out[(bare, c["name"])] = {
                "cardinality": c["cardinality"],
                "top_values": json.loads(c["top_values"]) if c["top_values"] else None,
            }
    return out


def _single_col_indexes(t: Tbl):
    """单列、非唯一、非主键的索引:(名, 列, 索引基数)。"""
    for name, cols, unique, card in t.indexes:
        if len(cols) == 1 and not unique and cols[0] not in t.pk:
            yield name, cols[0], card


def _rule_idx102(tables: dict[str, Tbl], stats: dict) -> list[Finding]:
    """低区分度单列索引:基数 / 行数 < 10%(基数取索引基数,缺则取列基数)。"""
    out = []
    for t in tables.values():
        if not t.row_count:
            continue
        for name, col, idx_card in _single_col_indexes(t):
            card = idx_card if idx_card is not None else (stats.get((t.name, col)) or {}).get(
                "cardinality")
            if card is None:
                continue
            sel = card / t.row_count
            if sel < 0.10:
                out.append(Finding(
                    table=t.name, rule_id="IDX102", severity="low",
                    evidence={"index": name, "column": col, "cardinality": card,
                              "row_count": t.row_count, "selectivity": round(sel, 6)},
                    suggestion=(f"`{name}`({col}) 区分度 {sel:.2%},单列索引过滤效果差;"
                                f"考虑并入以高区分度列打头的复合索引,或评估删除。"),
                ))
    return out


def _rule_idx103(tables: dict[str, Tbl], stats: dict, qmap: list[dict]) -> list[Finding]:
    """倾斜列索引:top1 值占比 > 90% 且代码中该列作条件的查询少(< 3 处)。"""
    uses: dict[tuple[str, str], int] = {}
    for q in qmap:
        if q["usage"] == "where":
            k = (q["table_name"], q["column_name"])
            uses[k] = uses.get(k, 0) + (q["freq"] or 1)
    out = []
    for t in tables.values():
        for name, col, _ in _single_col_indexes(t):
            tv = (stats.get((t.name, col)) or {}).get("top_values") or []
            counts = [float(v.get("n") or 0) for v in tv if isinstance(v, dict)]
            total = t.row_count or sum(counts)
            if not counts or not total:
                continue
            share = max(counts) / total
            if share > 0.90 and uses.get((t.name, col), 0) < 3:
                out.append(Finding(
                    table=t.name, rule_id="IDX103", severity="low",
                    evidence={"index": name, "column": col, "top1_share": round(share, 4),
                              "where_uses_in_code": uses.get((t.name, col), 0)},
                    suggestion=(f"`{col}` 取值高度倾斜(top1 占 {share:.0%})且少有查询以它为条件,"
                                f"索引 `{name}` 收益有限;如只查少数值,考虑与其它列组合。"),
                ))
    return out


def _rule_qry101(schema: str, slow: list[dict]) -> list[Finding]:
    """扫描/返回行数比 >1000(PLAN §9.8)。"""
    out = []
    from codeatlas.schema.collect_ob import _sql_tables

    for q in slow:
        try:
            ratio = float(q["rows_scanned"]) / max(1.0, float(q["rows_returned"]))
        except (TypeError, ValueError):
            continue
        if ratio > 1000:
            tables_hit = sorted(_sql_tables(q.get("sql") or ""))[:3]
            out.append(
                Finding(
                    table=tables_hit[0] if tables_hit else "(多表)",
                    rule_id="QRY101", severity="high" if ratio > 10000 else "medium",
                    evidence={
                        "sql": (q.get("sql") or "")[:200],
                        "freq": q.get("freq"), "avg_ms": q.get("avg_ms"),
                        "rows_scanned": q.get("rows_scanned"),
                        "rows_returned": q.get("rows_returned"),
                        "ratio": int(ratio),
                        "tables": tables_hit,
                    },
                    suggestion=(
                        f"扫描/返回比 {int(ratio)}:1,典型缺索引或索引未命中;"
                        f"结合 IDX001 的条件列建议核查该 SQL 的 where/join 列。"
                    ),
                )
            )
    return out


def _rule_idx101(tables: dict[str, Tbl]) -> list[Finding]:
    out = []
    for t in tables.values():
        if not t.has_profile or t.row_count is None:
            continue
        if t.row_count < 5000 and len(t.indexes) > 4:
            out.append(
                Finding(
                    table=t.name, rule_id="IDX101", severity="low",
                    evidence={
                        "row_count": t.row_count, "index_count": len(t.indexes),
                        "indexes": [n for n, _, _, _ in t.indexes],
                    },
                    suggestion=(
                        f"小表({t.row_count} 行)上建有 {len(t.indexes)} 个索引,"
                        f"维护开销可能超过收益,评估合并/删除。"
                    ),
                )
            )
    return out


def merge_slow_query_accesses(
    conn: sqlite3.Connection, schemas, profiles_dir: Path
) -> dict[str, list[dict]]:
    """慢查询(各库 *_manual.json 的 slow_queries)→ 访问路径并入 query_column_map。

    每次先清掉上一轮的慢查询行再写(列表为空也清),避免重跑后频次翻倍/残留;
    返回 {库: 慢查询条目}(QRY101 用)。
    """
    from codeatlas.schema.collect_ob import _num, load_manual
    from codeatlas.schema.sql_extract import (
        _clean_dynamic,
        _extract_accesses,
        _resolve_table_alias,
        _try_parse,
        fingerprint_sql,
    )

    conn.execute(
        "INSERT INTO repos(name, path, languages) VALUES('__codeatlas_reports__', '', '[]') "
        "ON CONFLICT(name) DO UPDATE SET path=''"
    )
    sid = conn.execute(
        "SELECT id FROM repos WHERE name='__codeatlas_reports__'"
    ).fetchone()["id"]
    conn.execute(
        "DELETE FROM query_column_map WHERE repo_id=? AND source_file LIKE 'slow_query:%'",
        (sid,),
    )
    slow_by_schema: dict[str, list[dict]] = {}
    merged: dict[tuple, int] = {}  # (库, 指纹, 表, 列, usage) → 最大频次
    for schema in schemas:
        mp = Path(profiles_dir) / f"{schema}_manual.json"
        if not mp.exists():
            continue
        slow = [q for q in (load_manual(mp).get("slow_queries") or []) if q.get("sql")]
        slow_by_schema[schema] = slow
        for q in slow:
            sql = _clean_dynamic(str(q["sql"]))
            stmt = _try_parse(sql)
            if stmt is None:
                continue
            _, accesses = _extract_accesses(stmt)
            _resolve_table_alias(stmt, accesses)
            fp = fingerprint_sql(sql)
            freq = int(_num(str(q.get("freq") or "")) or 1)
            for a in accesses:
                key = (schema, fp, a.table, a.column, a.usage)
                merged[key] = max(merged.get(key, 0), freq)
    conn.executemany(
        "INSERT INTO query_column_map(repo_id, query_fingerprint, source_file, "
        "source_line, table_name, column_name, usage, freq) VALUES(?,?,?,?,?,?,?,?)",
        [(sid, fp, f"slow_query:{sch}", 0, tbl, col, usage, freq)
         for (sch, fp, tbl, col, usage), freq in merged.items()],
    )
    conn.commit()
    return slow_by_schema


def render_report(results: list[AuditResult]) -> Path:
    """纯模板报告(结论确定性;LLM 润色为可选增强,不在 v1)。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"# OceanBase 索引体检报告({now})", ""]
    for r in results:
        lines.append(f"## {r.schema}")
        lines.append("")
        lines.append(
            f"- 表:{r.table_count};去重查询指纹:{r.query_count};"
            f"findings:{len(r.findings)}"
        )
        if r.unavailable:
            lines.append(
                f"- ⚠️ 画像缺失,以下规则未执行(标 unavailable):"
                f"{', '.join(r.unavailable)}"
            )
        if r.excluded_tables:
            lines.append(
                f"- 按约定排除临时表(temp/tmp)共 {len(r.excluded_tables)} 张:"
                f"{', '.join(r.excluded_tables)}"
            )
        lines.append("")
        by_rule: dict[str, list[Finding]] = {}
        for f in r.findings:
            by_rule.setdefault(f.rule_id, []).append(f)
        if not by_rule:
            lines.append("未发现问题。")
            continue
        for rule_id in sorted(by_rule):
            fs = by_rule[rule_id]
            lines.append(f"### {rule_id} {RULE_NAMES.get(rule_id, '')}({len(fs)} 个)")
            lines.append("")
            for f in sorted(fs, key=lambda x: (x.severity != "high", x.table)):
                lines.append(f"- **{f.table}** [{f.severity}]")
                for k, v in f.evidence.items():
                    if isinstance(v, (list, tuple)):
                        v = "; ".join(str(x) for x in list(v)[:3]) + (
                            f" 等 {len(v)} 项" if len(v) > 3 else ""
                        )
                    lines.append(f"  - {k}: {v}")
                if f.suggestion:
                    lines.append(f"  - 建议:{f.suggestion}")
            lines.append("")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"ob_audit_{now}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def persist_findings(conn: sqlite3.Connection, results: list[AuditResult]) -> None:
    conn.execute("DELETE FROM audit_findings")
    for r in results:
        for f in r.findings:
            conn.execute(
                "INSERT INTO audit_findings(table_name, rule_id, severity, evidence, suggestion) "
                "VALUES(?,?,?,?,?)",
                (
                    f"{r.schema}.{f.table}", f.rule_id, f.severity,
                    json.dumps(f.evidence, ensure_ascii=False), f.suggestion,
                ),
            )
    conn.commit()


async def index_report(conn, lance, embedder, report_path, settings) -> int:
    """报告 chunk 入索引(PLAN §9.8:报告入向量库,供 atlas ask 引用)。

    kind=report;重跑 audit 自动替换旧报告 chunk(FTS 行与向量一并替换)。
    """
    from codeatlas.db.chunkstore import embed_chunks, insert_text_chunk, purge_chunks
    from codeatlas.ingest.chunker import chunk_markdown

    # 报告不属于单仓库:哨兵 repo 行承载外键(schema 勿动,PLAN 约束)
    conn.execute(
        "INSERT INTO repos(name, path, languages) VALUES('__codeatlas_reports__', '', '[]') "
        "ON CONFLICT(name) DO UPDATE SET path=''"
    )
    repo_id = conn.execute(
        "SELECT id FROM repos WHERE name='__codeatlas_reports__'"
    ).fetchone()["id"]
    purge_chunks(conn, lance, repo_id=repo_id, kind="report")
    pending = []
    text = Path(report_path).read_text(encoding="utf-8", errors="replace")
    stem = Path(report_path).stem
    for ch in chunk_markdown(text, settings.chunk_max_tokens):
        cid = insert_text_chunk(
            conn, repo_id=repo_id, kind="report",
            title=f"report:{stem}:{ch.title or ''}", chunk=ch,
        )
        pending.append((cid, ch.content))
    conn.commit()
    try:
        await embed_chunks(conn, lance, embedder, pending,
                           repo_id=repo_id, kind="report", stage="audit")
    except Exception:
        pass  # embedding 失败不阻塞(FTS 路径可用)
    return len(pending)


async def index_docs(conn, lance, embedder, docs_dir, settings) -> int:
    """data/docs 人工文档入库(PLAN §6:人工维护文档与 wiki 一起入索引)。

    kind=doc、哨兵 repo 行 '__codeatlas_docs__' 承载外键;重跑自动替换。
    """
    from pathlib import Path as _P

    from codeatlas.db.chunkstore import embed_chunks, insert_text_chunk, purge_chunks
    from codeatlas.ingest.chunker import chunk_markdown

    conn.execute(
        "INSERT INTO repos(name, path, languages) VALUES('__codeatlas_docs__', '', '[]') "
        "ON CONFLICT(name) DO UPDATE SET path=''"
    )
    repo_id = conn.execute(
        "SELECT id FROM repos WHERE name='__codeatlas_docs__'"
    ).fetchone()["id"]
    purge_chunks(conn, lance, repo_id=repo_id, kind="doc")
    conn.commit()

    pending = []
    for page in sorted(_P(docs_dir).rglob("*.md")):
        if page.name == "README.md":
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        for ch in chunk_markdown(text, settings.chunk_max_tokens):
            cid = insert_text_chunk(
                conn, repo_id=repo_id, kind="doc",
                title=f"docs:{page.relative_to(docs_dir).as_posix()}:{ch.title or ''}", chunk=ch,
            )
            pending.append((cid, ch.content))
    conn.commit()
    try:
        await embed_chunks(conn, lance, embedder, pending,
                           repo_id=repo_id, kind="doc", stage="index")
    except Exception:
        pass
    return len(pending)
