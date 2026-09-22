"""DDL 解析(PLAN §9.8):.sql / .md(OB 工单常见格式)→ ddl_tables/columns/indexes。

OB MySQL 模式 DDL 带 sqlglot 不认识的选项(KEY 内联 BLOCK_SIZE n GLOBAL、表尾
REPLICA_NUM=n 等),先做清洗预处理再交给 sqlglot(mysql 方言)。
ddl_tables.name 存 "{schema}.{table}" 以支持多库(§7 的 UNIQUE 约束保留)。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp

# OB 特有选项(键值型与内联型),清洗后交给 sqlglot
_OB_KV = re.compile(r"(BLOCK_SIZE|TABLET_SIZE|PCTFREE|REPLICA_NUM)\s*=\s*\d+")
_OB_FLAG = re.compile(r"USE_BLOOM_FILTER\s*=\s*\w+")
_OB_INLINE = re.compile(r"\)\s+BLOCK_SIZE\s+\d+(\s+(GLOBAL|LOCAL))?")
_OB_SCOPE = re.compile(r"\)\s+(GLOBAL|LOCAL)\b")
_SQL_FENCE = re.compile(r"```sql\n(.*?)```", re.S)


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool
    is_pk: bool = False


@dataclass
class Index:
    name: str
    columns: list[str]
    is_unique: bool = False


@dataclass
class Table:
    name: str  # 裸表名(不含 schema)
    ddl_text: str
    columns: list[Column] = field(default_factory=list)
    indexes: list[Index] = field(default_factory=list)


@dataclass
class ParsedDDL:
    schema: str
    tables: list[Table] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)  # 解析失败的 CREATE TABLE 段


def scrub_ob_options(sql: str) -> str:
    s = _OB_KV.sub("", sql)
    s = _OB_FLAG.sub("", s)
    s = _OB_INLINE.sub(")", s)
    s = _OB_SCOPE.sub(")", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\(\s*,", "(", s)
    s = re.sub(r",\s*\)", ")", s)
    return s


def load_ddl_text(path: Path) -> str:
    """`.md`(提取 ```sql 块)或 `.sql`(全文)。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".md":
        blocks = _SQL_FENCE.findall(text)
        return "\n\n".join(blocks) if blocks else text
    return text


def _column_type(cd: exp.ColumnDef) -> str:
    kind = cd.args.get("kind")
    return kind.sql(dialect="mysql") if kind is not None else ""


def _column_nullable(cd: exp.ColumnDef) -> bool:
    for c in cd.constraints or []:
        if isinstance(c, exp.ColumnConstraint) and isinstance(
            c.kind, exp.NotNullColumnConstraint
        ):
            return False
    return True


def parse_ddl(schema: str, sql_text: str) -> ParsedDDL:
    """优先整体交给 sqlglot(语句边界感知,不会被 COMMENT 里的分号截断);
    失败再按 CREATE TABLE 边界降级逐块解析,块级失败只记不炸。"""
    out = ParsedDDL(schema=schema)
    statements = None
    try:
        statements = sqlglot.parse(scrub_ob_options(sql_text), dialect="mysql")
    except Exception:
        statements = None
    if statements is None:
        chunks = re.split(r"(?=CREATE TABLE)", sql_text, flags=re.I)
        # 清理 markdown fence 残留(``` 行)与标题行
        chunks = [re.sub(r"^\s*`{3,}.*$", "", c, flags=re.M) for c in chunks]
        statements = []
        for chunk in chunks:
            if not re.match(r"CREATE TABLE", chunk.strip(), re.I):
                continue
            try:
                st = sqlglot.parse(scrub_ob_options(chunk), dialect="mysql")[0]
                statements.append(st)
            except Exception:
                out.failures.append(chunk.strip()[:80])
        return _from_statements(out, statements)

    stmts = []
    for st in statements:
        if st is None:
            continue
        chunk_ok = True
        try:
            stmts.append(st)
        except Exception:  # pragma: no cover
            chunk_ok = False
        if not chunk_ok:
            out.failures.append("unknown")
    return _from_statements(out, stmts)


def _from_statements(out: ParsedDDL, statements: list) -> ParsedDDL:
    for stmt in statements:
        if not isinstance(stmt, exp.Create):
            continue
        try:
            schema_node = stmt.this
            tbl = stmt.find(exp.Table)
            if tbl is None or not isinstance(schema_node, exp.Schema):
                continue
            table = Table(name=tbl.name, ddl_text=stmt.sql(dialect="mysql"))
            pk_cols: set[str] = set()
            body = list(schema_node.expressions)
            for e in body:
                if isinstance(e, exp.PrimaryKey):
                    pk_cols = {
                        i.this for i in e.expressions if isinstance(i, exp.Identifier)
                    }
            for e in body:
                if isinstance(e, exp.ColumnDef):
                    table.columns.append(
                        Column(
                            name=e.name,
                            data_type=_column_type(e),
                            nullable=_column_nullable(e),
                            is_pk=e.name in pk_cols,
                        )
                    )
            for e in body:
                if isinstance(e, exp.IndexColumnConstraint):
                    cols = [c.name for c in e.find_all(exp.Ordered) if c.name]
                    if not cols:
                        cols = [c.name for c in e.find_all(exp.Column) if c.name]
                    table.indexes.append(
                        Index(name=e.name or f"idx_{table.name}", columns=cols)
                    )
                elif isinstance(e, exp.UniqueColumnConstraint):
                    # 结构:this=Schema(this=Identifier(索引名), expressions=[列...])
                    cols = [c.name for c in e.find_all(exp.Column) if c.name]
                    if not cols:
                        cols = [
                            i.name
                            for i in (e.this.expressions if isinstance(e.this, exp.Schema) else [])
                            if isinstance(i, exp.Identifier)
                        ]
                    idx_name = ""
                    if isinstance(e.this, exp.Schema) and isinstance(e.this.this, exp.Identifier):
                        idx_name = e.this.this.name
                    table.indexes.append(
                        Index(name=idx_name or f"uk_{table.name}", columns=cols, is_unique=True)
                    )
            if table.columns:
                out.tables.append(table)
            else:
                out.failures.append(table.name)
        except Exception:
            out.failures.append("?")
    return out


def load_ddl_dir(ddl_dir: Path) -> dict[str, ParsedDDL]:
    """data/ddl/ 下每个文件一个库:文件名(去后缀)= 库名。"""
    result: dict[str, ParsedDDL] = {}
    for path in sorted(ddl_dir.glob("*.sql")) + sorted(ddl_dir.glob("*.md")):
        schema = path.stem
        result[schema] = parse_ddl(schema, load_ddl_text(path))
    return result


def persist_ddl(conn: sqlite3.Connection, parsed: ParsedDDL) -> None:
    """写库(幂等:先清该 schema 的旧记录)。name 存 schema.table。"""
    prefix = f"{parsed.schema}.%"
    conn.execute("DELETE FROM ddl_tables WHERE name LIKE ?", (prefix,))
    for t in parsed.tables:
        full = f"{parsed.schema}.{t.name}"
        cur = conn.execute(
            "INSERT INTO ddl_tables(name, ddl_text) VALUES(?, ?)", (full, t.ddl_text)
        )
        table_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO ddl_columns(table_id, name, data_type, nullable, is_pk) "
            "VALUES(?,?,?,?,?)",
            [(table_id, c.name, c.data_type, 1 if c.nullable else 0, 1 if c.is_pk else 0)
             for c in t.columns],
        )
        import json

        conn.executemany(
            "INSERT INTO ddl_indexes(table_id, name, columns, is_unique) "
            "VALUES(?,?,?,?)",
            [(table_id, i.name, json.dumps(i.columns), 1 if i.is_unique else 0)
             for i in t.indexes],
        )
    conn.commit()
