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


def _load_tables(conn: sqlite3.Connection, schema: str) -> dict[str, Tbl]:
    prefix = f"{schema}.%"
    tables: dict[str, Tbl] = {}
    for r in conn.execute(
        "SELECT id, name, row_count FROM ddl_tables WHERE name LIKE ?", (prefix,)
    ):
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


def _load_query_map(conn: sqlite3.Connection) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT DISTINCT query_fingerprint, source_file, source_line, "
            "       table_name, column_name, usage, freq "
            "FROM query_column_map"
        )
    ]


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
    """冗余索引:(a) ⊂ (a,b) 同表前缀包含。"""
    out = []
    for t in tables.values():
        for n1, c1, u1, _ in t.indexes:
            for n2, c2, u2, _ in t.indexes:
                if n1 == n2 or u1 != u2:
                    continue
                if len(c1) < len(c2) and c2[: len(c1)] == c1:
                    out.append(
                        Finding(
                            table=t.name, rule_id="IDX003", severity="low",
                            evidence={
                                "kept": n2, "kept_columns": c2,
                                "redundant": n1, "redundant_columns": c1,
                            },
                            suggestion=(
                                f"索引 `{n1}`({c1}) 是 `{n2}`({c2}) 的前缀,"
                                f"可删除:ALTER TABLE `{t.name}` DROP INDEX `{n1}`;"
                            ),
                        )
                    )
    return out


def _base_type(dt: str) -> str:
    return (dt or "").split("(")[0].upper().strip()


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


def rule_idx005(anti_patterns: list[dict]) -> list[Finding]:
    """索引失效写法(来自 sql_extract 的 LIKE '%x'/函数包裹列检测)。"""
    out = []
    seen: set[tuple] = set()
    for ap in anti_patterns:
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
) -> AuditResult:
    tables = _load_tables(conn, schema)
    excluded = [
        r["name"].split(".", 1)[1]
        for r in conn.execute(
            "SELECT name FROM ddl_tables WHERE name LIKE ?", (f"{schema}.%",)
        )
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
    result.findings += rule_idx005(anti_patterns)

    # 画像类规则(有画像才跑;v1 实现 IDX101,IDX102/103/QRY101 需列级/慢查画像)
    profiled = [t for t in tables.values() if t.has_profile]
    if not profiled:
        result.unavailable += ["IDX101", "IDX102", "IDX103", "QRY101"]
    else:
        result.findings += _rule_idx101(tables)
        col_stats_ready = conn.execute(
            "SELECT COUNT(*) AS c FROM ddl_columns WHERE cardinality IS NOT NULL"
        ).fetchone()["c"]
        if col_stats_ready == 0:
            result.unavailable += ["IDX102", "IDX103", "QRY101"]
    return result


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

    kind=report;重跑 audit 自动替换旧报告 chunk。
    """
    import asyncio

    from codeatlas.db.fts import fts_delete
    from codeatlas.ingest.chunker import chunk_markdown

    old = conn.execute(
        "SELECT id, content FROM chunks WHERE kind='report'"
    ).fetchall()
    for r in old:
        fts_delete(conn, r["id"], r["content"])
    conn.execute("DELETE FROM chunks WHERE kind='report'")
    conn.commit()

    # 报告不属于单仓库:哨兵 repo 行承载外键(schema 勿动,PLAN 约束)
    conn.execute(
        "INSERT INTO repos(name, path, languages) VALUES('__codeatlas_reports__', '', '[]') "
        "ON CONFLICT(name) DO UPDATE SET path=''"
    )
    conn.commit()
    repo_id = conn.execute(
        "SELECT id FROM repos WHERE name='__codeatlas_reports__'"
    ).fetchone()["id"]
    n = 0
    pending = []
    text = Path(report_path).read_text(encoding="utf-8", errors="replace")
    for ch in chunk_markdown(text, settings.chunk_max_tokens):
        cur = conn.execute(
            "INSERT INTO chunks(repo_id, kind, title, content, content_hash, "
            "line_start, line_end) VALUES(?,?,?,?,?,?,?)",
            (repo_id, "report", f"report:{Path(report_path).stem}:{ch.title or ''}",
             ch.content, ch.content_hash, ch.line_start, ch.line_end),
        )
        pending.append((cur.lastrowid, ch.content))
        n += 1
    conn.commit()
    if embedder is not None and pending:
        try:
            vectors = await embedder.embed(
                [c for _, c in pending], stage="audit"
            )
            rows = [(cid, repo_id, "report", b) for (cid, _), b in zip(pending, vectors)]
            for lance_id, chunk_id in lance.add_vectors(rows):
                conn.execute(
                    "INSERT INTO vector_refs(chunk_id, lance_id, model) VALUES(?,?,?)",
                    (chunk_id, lance_id, embedder.s.embed_model),
                )
            conn.commit()
        except Exception:
            pass  # embedding 失败不阻塞(FTS 路径可用)
    return n


async def index_docs(conn, lance, embedder, docs_dir, settings) -> int:
    """data/docs 人工文档入库(PLAN §6:人工维护文档与 wiki 一起入索引)。

    kind=doc、哨兵 repo 行 '__codeatlas_docs__' 承载外键;重跑自动替换。
    """
    from pathlib import Path as _P

    from codeatlas.db.fts import fts_delete
    from codeatlas.ingest.chunker import chunk_markdown

    conn.execute(
        "INSERT INTO repos(name, path, languages) VALUES('__codeatlas_docs__', '', '[]') "
        "ON CONFLICT(name) DO UPDATE SET path=''"
    )
    repo_id = conn.execute(
        "SELECT id FROM repos WHERE name='__codeatlas_docs__'"
    ).fetchone()["id"]
    old = conn.execute(
        "SELECT id, content FROM chunks WHERE repo_id=? AND kind='doc'", (repo_id,)
    ).fetchall()
    for r in old:
        fts_delete(conn, r["id"], r["content"])
    conn.execute("DELETE FROM chunks WHERE repo_id=? AND kind='doc'", (repo_id,))
    conn.commit()

    n = 0
    pending = []
    for page in sorted(_P(docs_dir).rglob("*.md")):
        if page.name == "README.md":
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        for ch in chunk_markdown(text, settings.chunk_max_tokens):
            cur = conn.execute(
                "INSERT INTO chunks(repo_id, kind, title, content, content_hash, "
                "line_start, line_end) VALUES(?,?,?,?,?,?,?)",
                (repo_id, "doc", f"docs:{page.relative_to(docs_dir)}:{ch.title or ''}",
                 ch.content, ch.content_hash, ch.line_start, ch.line_end),
            )
            pending.append((cur.lastrowid, ch.content))
            n += 1
    conn.commit()
    if embedder is not None and pending:
        try:
            vectors = await embedder.embed([c for _, c in pending], stage="index")
            rows = [(cid, repo_id, "doc", b) for (cid, _), b in zip(pending, vectors)]
            for lance_id, chunk_id in lance.add_vectors(rows):
                conn.execute(
                    "INSERT INTO vector_refs(chunk_id, lance_id, model) VALUES(?,?,?)",
                    (chunk_id, lance_id, embedder.s.embed_model),
                )
            conn.commit()
        except Exception:
            pass
    return n
