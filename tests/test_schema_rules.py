"""OB 体检规则与解析回归:分区表/内联主键/大小写/冗余索引/IN-BETWEEN/别名/
MyBatis 写法/慢查询文件格式/画像文件不被覆盖。"""

import json

import pytest

from codeatlas.db.models import connect, init_db
from codeatlas.schema.audit import run_audit
from codeatlas.schema.ddl import parse_ddl, persist_ddl
from codeatlas.schema.sql_extract import (
    _extract_accesses,
    _extract_anti_patterns,
    _try_parse,
    extract_from_java,
    extract_from_xml,
)


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "kb.sqlite")
    init_db(conn)
    yield conn
    conn.close()


def _audit(db, ddl: str, qmap=(), anti=(), slow=None, schema="s"):
    persist_ddl(db, parse_ddl(schema, ddl))
    return run_audit(db, schema, list(qmap), list(anti), slow_queries=slow)


def _q(table, column, usage="where", fp="fp1", freq=1):
    return {"query_fingerprint": fp, "source_file": "M.xml", "source_line": 1,
            "table_name": table, "column_name": column, "usage": usage, "freq": freq}


def _rules(result, rule_id):
    return [f for f in result.findings if f.rule_id == rule_id]


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

def test_partitioned_and_option_heavy_tables_not_dropped():
    ddl = """
CREATE TABLE t_a (id bigint NOT NULL, uid bigint, PRIMARY KEY (id), KEY idx_uid (uid))
  PARTITION BY HASH(id) PARTITIONS 16;
CREATE TABLE t_b (id bigint NOT NULL, gmt datetime, PRIMARY KEY (id, gmt))
  DEFAULT CHARSET = utf8mb4 PARTITION BY RANGE COLUMNS(gmt)
  (PARTITION p0 VALUES LESS THAN ('2024-01-01'), PARTITION p1 VALUES LESS THAN (MAXVALUE));
CREATE TABLE t_c (id bigint NOT NULL, PRIMARY KEY (id)) TABLEGROUP = 'tg_x' LOCALITY = 'F@z1';
"""
    p = parse_ddl("s", ddl)
    assert sorted(t.name for t in p.tables) == ["t_a", "t_b", "t_c"]
    assert not p.failures
    a = next(t for t in p.tables if t.name == "t_a")
    assert [i.columns for i in a.indexes] == [["uid"]]


def test_unparseable_create_table_counted_as_failure():
    p = parse_ddl("s", "CREATE TABLE t_bad (id bigint NOT NULL,, ((( garbage")
    assert p.failures and not p.tables


def test_inline_primary_key_is_pk(db):
    r = _audit(db, "CREATE TABLE t_i (id bigint PRIMARY KEY AUTO_INCREMENT, name varchar(20));")
    assert not _rules(r, "TBL001")


def test_uppercase_ddl_matches_lowercase_queries(db):
    r = _audit(db, "CREATE TABLE T_ORDER (ID bigint NOT NULL, STATUS int, PRIMARY KEY (ID));",
               qmap=[_q("t_order", "status")])
    assert [f.evidence["column"] for f in _rules(r, "IDX001")] == ["status"]
    assert not _rules(r, "TBL001")


def test_schema_prefix_is_not_a_like_wildcard(db):
    persist_ddl(db, parse_ddl("a_b", "CREATE TABLE t1 (id int, PRIMARY KEY (id));"))
    persist_ddl(db, parse_ddl("axb", "CREATE TABLE t2 (id int, PRIMARY KEY (id));"))
    persist_ddl(db, parse_ddl("a_b", "CREATE TABLE t1 (id int, PRIMARY KEY (id));"))
    names = sorted(r["name"] for r in db.execute("SELECT name FROM ddl_tables"))
    assert names == ["a_b.t1", "axb.t2"]
    assert run_audit(db, "a_b", [], []).table_count == 1


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------

def test_idx003_never_suggests_dropping_unique_index(db):
    r = _audit(db, """CREATE TABLE t (id int, a int, b int, PRIMARY KEY (id),
        UNIQUE KEY uk_a (a), UNIQUE KEY uk_ab (a, b));""")
    assert not _rules(r, "IDX003")


def test_idx003_duplicates_and_nonunique_covered_by_unique(db):
    r = _audit(db, """CREATE TABLE t (id int, a int, b int, c int, PRIMARY KEY (id),
        KEY idx_c1 (c), KEY idx_c2 (c),
        KEY idx_a (a), UNIQUE KEY uk_ab (a, b),
        KEY idx_id (id));""")
    redundant = sorted(f.evidence["redundant"] for f in _rules(r, "IDX003"))
    assert redundant == ["idx_a", "idx_c2", "idx_id"]


def test_idx002_skips_query_served_by_another_index(db):
    ddl = "CREATE TABLE t (id int, a int, b int, PRIMARY KEY (id), KEY idx_ab (a, b), KEY idx_b (b));"
    assert not _rules(_audit(db, ddl, qmap=[_q("t", "b")]), "IDX002")


def test_idx002_still_flags_real_mismatch(db):
    ddl = "CREATE TABLE t (id int, a int, b int, PRIMARY KEY (id), KEY idx_ab (a, b));"
    assert _rules(_audit(db, ddl, qmap=[_q("t", "b")]), "IDX002")


def test_idx005_scoped_to_schema_tables(db):
    persist_ddl(db, parse_ddl("other", "CREATE TABLE t_x (id int, PRIMARY KEY (id));"))
    anti = [{"table": "t_order", "column": "name", "pattern": "LIKE '%...'",
             "fingerprint": "f", "source_file": "M.xml", "source_line": 3}]
    r1 = _audit(db, "CREATE TABLE t_order (id int, name varchar(9), PRIMARY KEY (id));",
                anti=anti, schema="s")
    r2 = run_audit(db, "other", [], anti)
    assert len(_rules(r1, "IDX005")) == 1 and not _rules(r2, "IDX005")


def test_qry101_flags_full_scan_returning_nothing(db):
    slow = [{"sql": "SELECT * FROM t WHERE a = 1", "rows_scanned": 5_000_000,
             "rows_returned": 0, "freq": 3}]
    r = _audit(db, "CREATE TABLE t (id int, a int, PRIMARY KEY (id));", slow=slow)
    assert _rules(r, "QRY101")


def test_column_stat_rules_run_when_profile_present(db):
    from codeatlas.schema.collect_ob import apply_profile

    persist_ddl(db, parse_ddl("s", """CREATE TABLE t (id int, status int, PRIMARY KEY (id),
        KEY idx_status (status));"""))
    apply_profile(db, "s", {"tables": {"t": {
        "row_count": 1_000_000,
        "index_cardinality": {"idx_status": 4},
        "column_stats": {"status": {"cardinality": 4, "top_values": [
            {"v": "DONE", "n": 990_000}, {"v": "INIT", "n": 10_000}]}},
    }}})
    r = run_audit(db, "s", [], [])
    assert "IDX102" not in r.unavailable and "IDX103" not in r.unavailable
    assert _rules(r, "IDX102") and _rules(r, "IDX103")


# ---------------------------------------------------------------------------
# SQL 访问路径提取
# ---------------------------------------------------------------------------

def _acc(sql):
    stmt = _try_parse(sql)
    _, acc = _extract_accesses(stmt)
    return {(a.table, a.column, a.usage) for a in acc}


def test_in_and_between_are_where_accesses():
    acc = _acc("SELECT id FROM t_order WHERE status IN (1, 2) AND gmt BETWEEN 1 AND 2")
    assert {("t_order", "status", "where"), ("t_order", "gmt", "where")} <= acc


def test_uppercase_alias_resolved(tmp_path):
    xml = tmp_path / "M.xml"
    xml.write_text("""<mapper namespace="x">
  <select id="q">SELECT O.id FROM t_order O JOIN t_item I ON O.ID = I.ORDER_ID
    WHERE O.STATUS = #{s} AND I.NAME LIKE CONCAT('%', #{n})</select>
</mapper>""", encoding="utf-8")
    q = extract_from_xml(xml, "M.xml")[0]
    acc = {(a.table, a.column, a.usage) for a in q.accesses}
    assert ("t_order", "status", "where") in acc
    assert ("t_item", "order_id", "join") in acc
    assert all(ap["table"] in ("t_order", "t_item") for ap in q.anti_patterns)


def test_function_wrapped_column_detected():
    stmt = _try_parse("SELECT id FROM t WHERE DATE_FORMAT(gmt, '%Y') = '2024' AND YEAR(c) = 1")
    out = _extract_anti_patterns(stmt, ["t"])
    assert {o["column"] for o in out if o["pattern"].startswith("FUNC")} == {"gmt", "c"}
    stmt2 = _try_parse("SELECT id FROM t WHERE gmt > NOW()")  # 函数在常量侧:不算
    assert not _extract_anti_patterns(stmt2, ["t"])


MAPPER_VARIANTS = """<?xml version="1.0"?>
<mapper namespace="x">
  <!-- <select id="old">SELECT * FROM t_dead WHERE zz = 1</select> -->
  <select id='single'>SELECT id FROM t_a WHERE a1 = #{a}</select>
  <select id="cdata">SELECT id FROM t_b WHERE <![CDATA[ b1 < #{x} ]]></select>
  <select id="escaped">SELECT id FROM t_c WHERE c1 &lt;= #{x} AND c2 &gt; 1</select>
  <select id="trim">SELECT id FROM t_d
    <trim prefix="WHERE" prefixOverrides="AND |OR ">
      <if test="x">AND d1 = #{x}</if>
    </trim>
  </select>
  <insert id="ins">INSERT INTO t_e
    <trim prefix="(" suffix=")" suffixOverrides=",">e1, e2,</trim>
    VALUES <trim prefix="(" suffix=")" suffixOverrides=",">#{a}, #{b},</trim>
  </insert>
</mapper>"""


def test_mybatis_variants_extracted(tmp_path):
    xml = tmp_path / "M.xml"
    xml.write_text(MAPPER_VARIANTS, encoding="utf-8")
    qs = extract_from_xml(xml, "M.xml")
    tables = {t for q in qs for t in q.tables}
    assert {"t_a", "t_b", "t_c", "t_d", "t_e"} <= tables
    assert "t_dead" not in tables  # 注释掉的语句不提取
    cols = {(a.table, a.column) for q in qs for a in q.accesses}
    assert {("t_a", "a1"), ("t_b", "b1"), ("t_c", "c1"), ("t_d", "d1")} <= cols


def test_java_annotation_counted_once_and_script_supported(tmp_path):
    java = tmp_path / "M.java"
    java.write_text('''interface M {
    @Select("SELECT id FROM t_order WHERE status = #{s}")
    List<Long> a(int s);
    @Select("<script>SELECT id FROM t_item <where><if test='x'>AND sku = #{x}</if></where></script>")
    List<Long> b(String x);
}''', encoding="utf-8")
    qs = extract_from_java(java, "M.java")
    assert sorted(t for q in qs for t in q.tables) == ["t_item", "t_order"]


# ---------------------------------------------------------------------------
# 画像 / 慢查询文件
# ---------------------------------------------------------------------------

def test_sql_tables_uses_parser_not_regex():
    from codeatlas.schema.collect_ob import _sql_tables

    assert _sql_tables("SELECT * FROM db2.t_item i WHERE EXTRACT(YEAR FROM gmt) = 1") == {"t_item"}


def test_parse_slow_file_formats(tmp_path):
    from codeatlas.schema.collect_ob import parse_slow_file

    bom = tmp_path / "a.csv"
    bom.write_text('﻿sql,freq,rows_scanned,rows_returned\n'
                   '"SELECT id\nFROM t_order WHERE a = 1","1,200",100,1\n', encoding="utf-8")
    rows = parse_slow_file(bom)
    assert len(rows) == 1 and "\n" in rows[0]["sql"] or "id FROM" in rows[0]["sql"]
    assert rows[0]["freq"] == 1200

    plain = tmp_path / "b.sql"
    plain.write_text("SELECT a, sql_text FROM t_x WHERE a = 1;\nSELECT id FROM t_y WHERE b = 2;\n", encoding="utf-8")
    assert [r["sql"].split()[-1] for r in parse_slow_file(plain)] == ["1", "2"]

    empty = tmp_path / "c.txt"
    empty.write_text("", encoding="utf-8")
    assert parse_slow_file(empty) == []

    nd = tmp_path / "d.json"
    nd.write_text('{"sql": "SELECT 1 FROM t", "freq": 2}\n{"sql": "SELECT 2 FROM t"}\n', encoding="utf-8")
    assert len(parse_slow_file(nd)) == 2


def test_collect_rerun_keeps_filled_profile(tmp_path):
    from codeatlas.schema.collect_ob import (
        generate_fill_sheet,
        import_fill_sheet,
        manual_template,
    )

    parsed = {"s": parse_ddl("s", "CREATE TABLE t_order (id int, status int, PRIMARY KEY (id));")}
    mp = manual_template("s", parsed["s"], out_dir=tmp_path)
    data = json.loads(mp.read_text(encoding="utf-8"))
    data["tables"]["t_order"]["index_cardinality"] = {"PRIMARY": 99}
    data["slow_queries"] = [{"sql": "SELECT 1 FROM t_order", "freq": 1}]
    mp.write_text(json.dumps(data), encoding="utf-8")

    sheet = generate_fill_sheet(parsed, out_dir=tmp_path)
    sheet.write_text(sheet.read_text(encoding="utf-8").replace("| t_order |  |", "| t_order | 12345 |"), encoding="utf-8")
    import_fill_sheet(sheet, parsed, out_dir=tmp_path)

    # 重跑 collect:模板与填报表都不得抹掉已填内容
    manual_template("s", parsed["s"], out_dir=tmp_path)
    generate_fill_sheet(parsed, out_dir=tmp_path)
    data = json.loads(mp.read_text(encoding="utf-8"))
    t = data["tables"]["t_order"]
    assert t["row_count"] == 12345
    assert t["index_cardinality"] == {"PRIMARY": 99}       # 导入不覆盖整表字典
    assert data["slow_queries"] == [{"sql": "SELECT 1 FROM t_order", "freq": 1}]
    assert "| t_order | 12345 |" in sheet.read_text(encoding="utf-8")


def test_generated_script_escapes_identifiers(tmp_path):
    from codeatlas.schema.collect_ob import generate_scripts

    parsed = {"x'; DROP TABLE y; --": parse_ddl("x", "CREATE TABLE `t``q` (id int, PRIMARY KEY (id));")}
    script = generate_scripts(parsed, out_dir=tmp_path)[0].read_text(encoding="utf-8")
    assert "TABLE_SCHEMA = 'x''; DROP TABLE y; --'" in script
    assert "`t``q`" in script


def test_slow_query_accesses_not_doubled_across_runs(db, tmp_path):
    from codeatlas.schema.audit import merge_slow_query_accesses

    persist_ddl(db, parse_ddl("s", "CREATE TABLE t_order (id int, st int, PRIMARY KEY (id));"))
    (tmp_path / "s_manual.json").write_text(json.dumps({"tables": {}, "slow_queries": [
        {"sql": "SELECT id FROM t_order o WHERE o.st = 1", "freq": "1,200"}]}), encoding="utf-8")
    for _ in range(2):
        slow_by_schema = merge_slow_query_accesses(db, ["s"], tmp_path)
    rows = db.execute(
        "SELECT table_name, column_name, freq FROM query_column_map "
        "WHERE source_file LIKE 'slow_query:%'").fetchall()
    assert [tuple(r) for r in rows] == [("t_order", "st", 1200)]  # 别名已还原,频次不翻倍
    assert len(slow_by_schema["s"]) == 1
    (tmp_path / "s_manual.json").write_text(json.dumps({"tables": {}, "slow_queries": []}), encoding="utf-8")
    merge_slow_query_accesses(db, ["s"], tmp_path)
    assert not db.execute(
        "SELECT 1 FROM query_column_map WHERE source_file LIKE 'slow_query:%'").fetchone()
