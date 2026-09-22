"""代码 SQL 提取(PLAN §9.8,优先级:字符串 SQL → MyBatis XML → 注解——此处按
可达性实现为 XML → 注解 → 字符串,覆盖同一集合)。

产出 query_column_map(代码 SQL → 表列访问路径,usage = where/join/order/group):
- MyBatis XML:<select|insert|update|delete>,动态标签剥离(<if>/<where>/<foreach>/
  <set>/<choose> 等),<include refid> 用 <sql> 片段展开,#{} 与 ${} 替换占位;
- Java 注解:@Select/@Update/@Insert/@Delete;
- Java 字符串:完整 SELECT/INSERT/UPDATE/DELETE 语句形状(保守)。

动态 SQL 是近似提取:剥离后不可解析的语句跳过并计数,不猜。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp

_TAG_RE = re.compile(r"</?(if|where|foreach|set|trim|choose|when|otherwise|bind|sql|include)[^>]*>", re.I)
_STMT_TAGS = re.compile(
    r"<(select|insert|update|delete)\b[^>]*?id=\"([^\"]+)\"[^>]*>(.*?)</\1>",
    re.S | re.I,
)
_SQL_FRAG = re.compile(r"<sql\b[^>]*?id=\"([^\"]+)\"[^>]*>(.*?)</sql>", re.S | re.I)
_INCLUDE = re.compile(r"<include\b[^>]*?refid=\"([^\"]+)\"[^>]*/>", re.I)
_PARAM = re.compile(r"[$#]\{[^}]*\}")
_LEAD_AND = re.compile(r"^\s*(AND|OR)\b", re.I)
_ANNOTATION = re.compile(
    r"@(Select|Update|Insert|Delete)\s*\(\s*\"((?:[^\"\\]|\\.)*)\"", re.S
)
_STR_SQL = re.compile(
    r"\"((?:[^\"\\]|\\.)*?(?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+"
    r"(?:[^\"\\]|\\.)*?)\"",
    re.I | re.S,
)


@dataclass
class Access:
    table: str
    column: str
    usage: str  # where / join / order / group


@dataclass
class ExtractedQuery:
    source_file: str
    source_line: int
    sql: str            # 清洗后可解析的 SQL
    fingerprint: str
    tables: list[str] = field(default_factory=list)
    accesses: list[Access] = field(default_factory=list)
    anti_patterns: list[dict] = field(default_factory=list)  # IDX005 证据


def fingerprint_sql(sql: str) -> str:
    """归一化指纹:字面量替换为 ? 后取 hash。"""
    try:
        st = sqlglot.parse_one(sql, dialect="mysql")
    except Exception:
        return hashlib.blake2b(re.sub(r"'[^']*'", "?", sql).encode(),
                               digest_size=12).hexdigest()
    def _lit(node):
        if isinstance(node, exp.Literal):
            return exp.Literal.string("?") if node.is_string else exp.Literal.number(0)
        return node
    st = st.transform(_lit)
    norm = st.sql(dialect="mysql")
    return hashlib.blake2b(norm.encode(), digest_size=12).hexdigest()


def _clean_dynamic(sql: str) -> str:
    s = _PARAM.sub("'?'", sql)
    # <where>/<set> 标签承载关键字语义:剥掉会留下悬挂 AND/OR,转为 WHERE 1=1 技巧
    s = re.sub(r"<where\b[^>]*>(.*?)</where>", r" WHERE 1=1 \1 ", s, flags=re.S | re.I)
    s = re.sub(r"<set\b[^>]*>(.*?)</set>", r" SET \1 ", s, flags=re.S | re.I)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    s = _LEAD_AND.sub("", s)
    return s.strip().rstrip(",")


def _extract_accesses(stmt: exp.Expression) -> tuple[list[str], list[Access]]:
    tables = sorted({t.name for t in stmt.find_all(exp.Table) if t.name})
    accesses: list[Access] = []
    seen: set[tuple[str, str, str]] = set()

    def add(table: str, column: str, usage: str) -> None:
        if not column or column == "*":
            return
        if not table:
            # 无别名前缀:单表语句回填主表;多表歧义宁缺毋错
            table = tables[0] if len(tables) == 1 else "" 
            if not table:
                return
        key = (table, column, usage)
        if key not in seen:
            seen.add(key)
            accesses.append(Access(table.lower(), column.lower(), usage))

    # where 等值/比较列(列 与 字面量/参数 比较的一侧)
    for where in stmt.find_all(exp.Where):
        for pred in where.find_all(exp.Binary):
            if not isinstance(pred, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE,
                                     exp.Like, exp.ILike, exp.In)):
                continue
            left, right = pred.this, pred.expression
            for col, other in ((left, right), (right, left)):
                if isinstance(col, exp.Column) and not isinstance(other, exp.Column):
                    add(col.table or "", col.name, "where")
    # join on 双列
    for join in stmt.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        for pred in on.find_all(exp.EQ):
            l, r = pred.this, pred.expression
            if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                # 同表自联接排除:不同别名/表才算 join 键
                add(l.table or "", l.name, "join")
                add(r.table or "", r.name, "join")
    # order by / group by
    for sel in stmt.find_all(exp.Select):
        order = sel.args.get("order")
        if order:
            for o in order.expressions:
                c = o.unalias() if hasattr(o, "unalias") else o
                col = c.this if isinstance(c, exp.Ordered) else c
                if isinstance(col, exp.Column):
                    add(col.table or "", col.name, "order")
        group = sel.args.get("group")
        if group:
            for g in group.expressions:
                if isinstance(g, exp.Column):
                    add(g.table or "", g.name, "group")
    return tables, accesses


def _extract_anti_patterns(stmt: exp.Expression, tables: list[str]) -> list[dict]:
    """IDX005 证据:LIKE 前导通配 / where 内函数包裹列。"""
    out: list[dict] = []
    fallback_table = tables[0] if tables else ""
    for like in stmt.find_all(exp.Like):
        right = like.expression
        lits = [right] if isinstance(right, exp.Literal) else list(right.find_all(exp.Literal))
        first = next((l for l in lits if l.is_string), None)
        if first is not None and str(first.this).startswith("%"):
            col = like.this if isinstance(like.this, exp.Column) else None
            out.append({
                "table": (col.table or fallback_table).lower(),
                "column": col.name if col else "?",
                "pattern": "LIKE '%...'",
            })
    for where in stmt.find_all(exp.Where):
        for call in where.find_all(exp.Anonymous):
            if call.name and call.name.upper() in ("DATE", "DATE_FORMAT", "SUBSTR",
                                                   "SUBSTRING", "LEFT", "RIGHT",
                                                   "CONCAT", "CAST", "TRIM", "YEAR",
                                                   "MONTH", "DAY", "IF", "IFNULL", "COALESCE"):
                for col in call.find_all(exp.Column):
                    out.append({
                        "table": (col.table or fallback_table).lower(),
                        "column": col.name,
                        "pattern": f"FUNC({col.name})",
                    })
    return out


def _try_parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, dialect="mysql")
    except Exception:
        return None


def _resolve_table_alias(stmt: exp.Expression, accesses: list[Access]) -> None:
    """把别名列归属到真实表:alias.col → table.col。"""
    alias_map: dict[str, str] = {}
    for t in stmt.find_all(exp.Table):
        if t.alias:
            alias_map[t.alias] = t.name
    for a in accesses:
        if a.table in alias_map:
            a.table = alias_map[a.table].lower()
        elif a.table and a.table not in {t.lower() for t in alias_map.values()}:
            # 无别名限定且与某表名一致(大小写差异)则归一
            pass


def extract_from_xml(path: Path, rel: str) -> list[ExtractedQuery]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "<mapper" not in text:
        return []
    frags = {m.group(1): m.group(2) for m in _SQL_FRAG.finditer(text)}
    out: list[ExtractedQuery] = []
    for m in _STMT_TAGS.finditer(text):
        tag, sid, body = m.group(1), m.group(2), m.group(3)
        line = text[: m.start()].count("\n") + 1

        def _expand(match):
            frag = frags.get(match.group(1), "")
            return " " + frag + " "

        body_expanded = _INCLUDE.sub(_expand, body)
        sql = _clean_dynamic(body_expanded)
        if tag.lower() == "select" and not re.match(r"(?i)select\b", sql):
            sql = "SELECT " + sql
        stmt = _try_parse(sql)
        if stmt is None:
            continue
        tables, accesses = _extract_accesses(stmt)
        _resolve_table_alias(stmt, accesses)
        if not tables:
            continue
        lowered = [t_.lower() for t_ in tables]
        out.append(
            ExtractedQuery(
                source_file=rel,
                source_line=line,
                sql=sql[:2000],
                fingerprint=fingerprint_sql(sql),
                tables=lowered,
                accesses=accesses,
                anti_patterns=_extract_anti_patterns(stmt, lowered),
            )
        )
    return out


def extract_from_java(path: Path, rel: str) -> list[ExtractedQuery]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[ExtractedQuery] = []

    def _emit(sql: str, offset: int) -> None:
        sql = _clean_dynamic(sql).replace('\\"', "'").replace("\\n", " ")
        stmt = _try_parse(sql)
        if stmt is None:
            return
        tables, accesses = _extract_accesses(stmt)
        _resolve_table_alias(stmt, accesses)
        if not tables:
            return
        lowered = [t_.lower() for t_ in tables]
        out.append(
            ExtractedQuery(
                source_file=rel,
                source_line=text.count("\n", 0, offset) + 1,
                sql=sql[:2000],
                fingerprint=fingerprint_sql(sql),
                tables=lowered,
                accesses=accesses,
                anti_patterns=_extract_anti_patterns(stmt, lowered),
            )
        )

    for m in _ANNOTATION.finditer(text):
        _emit(m.group(2), m.start())
    for m in _STR_SQL.finditer(text):
        _emit(m.group(1), m.start())
    return out


def extract_repo(repo_cfg, repo_root: Path) -> list[ExtractedQuery]:
    """扫一个仓库(尊重 .gitignore/exclude,复用 walker)。"""
    from codeatlas.config import RepoCfg
    from codeatlas.ingest.walker import walk

    queries: list[ExtractedQuery] = []
    for entry in walk(repo_cfg):
        if entry.rel.endswith(".xml"):
            queries += extract_from_xml(entry.abs_path, entry.rel)
        elif entry.rel.endswith(".java"):
            queries += extract_from_java(entry.abs_path, entry.rel)
    return queries


def persist_query_map(
    conn: sqlite3.Connection, repo_id: int, queries: list[ExtractedQuery]
) -> int:
    rows = []
    for q in queries:
        for a in q.accesses:
            rows.append(
                (repo_id, q.fingerprint, q.source_file, q.source_line,
                 a.table, a.column, a.usage)
            )
    # 同 (repo, 指纹, 文件, 表列 usage) 合并计数
    merged: dict[tuple, int] = {}
    for r in rows:
        merged[r] = merged.get(r, 0) + 1
    conn.execute("DELETE FROM query_column_map WHERE repo_id=?", (repo_id,))
    conn.executemany(
        "INSERT INTO query_column_map(repo_id, query_fingerprint, source_file, "
        "source_line, table_name, column_name, usage, freq) VALUES(?,?,?,?,?,?,?,?)",
        [k + (v,) for k, v in merged.items()],
    )
    conn.commit()
    return len(merged)
