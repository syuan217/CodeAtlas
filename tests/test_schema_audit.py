"""M4 schema 模块测试:DDL 解析(OB 选项/嵌套注释)、SQL 提取(动态标签)、规则引擎。"""

import json

import pytest

from codeatlas.db.models import connect, init_db
from codeatlas.schema.audit import run_audit
from codeatlas.schema.collect_ob import apply_profile, generate_scripts, manual_template
from codeatlas.schema.ddl import load_ddl_text, parse_ddl, persist_ddl
from codeatlas.schema.sql_extract import (
    extract_from_xml,
    fingerprint_sql,
    persist_query_map,
)

# ---------------------------------------------------------------------------
# DDL 解析
# ---------------------------------------------------------------------------

OB_DDL = """## t_order

订单表

```sql
CREATE TABLE `t_order` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '主键;含分号测试',
  `biz_no` varchar(64) NOT NULL DEFAULT '' COMMENT '业务单号',
  `status` varchar(16) NOT NULL DEFAULT 'INIT' COMMENT '状态',
  `org_code` varchar(32) DEFAULT NULL,
  `creator` varchar(64) NOT NULL DEFAULT '',
  `gmt_create` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `gmt_modified` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_biz_no` (`biz_no`),
  KEY `idx_status_org` (`org_code`,`status`) BLOCK_SIZE 16384 GLOBAL,
  KEY `idx_org_code` (`org_code`) BLOCK_SIZE 16384 GLOBAL
) AUTO_INCREMENT = 100 DEFAULT CHARSET = utf8mb4 ROW_FORMAT = DYNAMIC COMPRESSION = 'zstd_1.0' REPLICA_NUM = 3 BLOCK_SIZE = 16384 USE_BLOOM_FILTER = FALSE TABLET_SIZE = 134217728 PCTFREE = 0;
```

## t_no_pk

```sql
CREATE TABLE `t_no_pk` (
  `id` bigint(20) NOT NULL,
  `code` varchar(32) NOT NULL
);
```

## t_tmp_scratch

```sql
CREATE TABLE `t_tmp_scratch` (
  `id` bigint(20) NOT NULL
);
```
"""


@pytest.fixture
def parsed():
    return parse_ddl("testdb", OB_DDL)


def test_parse_ddl_md_fences_and_ob_options(parsed):
    assert len(parsed.tables) == 3
    assert not parsed.failures
    order = next(t for t in parsed.tables if t.name == "t_order")
    cols = {c.name: c for c in order.columns}
    assert cols["id"].is_pk and not cols["id"].nullable
    assert not cols["biz_no"].is_pk and not cols["biz_no"].nullable
    assert cols["org_code"].nullable
    assert cols["status"].data_type.upper().startswith("VARCHAR")
    idxs = {i.name: i for i in order.indexes}
    assert idxs["uk_biz_no"].is_unique
    assert idxs["idx_status_org"].columns == ["org_code", "status"]


def test_load_ddl_text_formats(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("标题\n```sql\nSELECT 1;\n```\n其它 ```text\nnope\n```")
    assert "SELECT 1" in load_ddl_text(md)
    sql = tmp_path / "y.sql"
    sql.write_text("SELECT 2;")
    assert load_ddl_text(sql) == "SELECT 2;"


def test_persist_and_reload(parsed, tmp_path, monkeypatch):
    from codeatlas import config as config_mod

    monkeypatch.setattr(config_mod, "DB_PATH", tmp_path / "t.sqlite")
    conn = connect()
    init_db(conn)
    persist_ddl(conn, parsed)
    n = conn.execute("SELECT COUNT(*) AS c FROM ddl_tables WHERE name LIKE 'testdb.%'").fetchone()["c"]
    assert n == 3
    persist_ddl(conn, parsed)  # 幂等
    n2 = conn.execute("SELECT COUNT(*) AS c FROM ddl_tables WHERE name LIKE 'testdb.%'").fetchone()["c"]
    assert n2 == 3
    conn.close()


# ---------------------------------------------------------------------------
# MyBatis SQL 提取
# ---------------------------------------------------------------------------

MAPPER_XML = """<?xml version="1.0"?>
<mapper namespace="test.OrderMapper">
  <sql id="baseWhere">
    <if test="bizNo != null">AND biz_no = #{bizNo}</if>
    <if test="status != null">AND status = #{status}</if>
  </sql>
  <select id="selectByCondition" resultMap="rm">
    SELECT id, biz_no, status FROM t_order
    <where>
      <include refid="baseWhere"/>
      <if test="orgCode != null">AND org_code like CONCAT('%', #{orgCode}, '%')</if>
    </where>
    ORDER BY gmt_create DESC
  </select>
  <select id="joinCustomer" resultType="map">
    SELECT o.id FROM t_order o
    JOIN t_no_pk p ON o.org_code = p.code
    WHERE o.status = #{status} GROUP BY o.biz_no
  </select>
</mapper>
"""


def test_extract_xml_dynamic_tags(tmp_path):
    f = tmp_path / "OrderMapper.xml"
    f.write_text(MAPPER_XML, encoding="utf-8")
    qs = extract_from_xml(f, "mapper/OrderMapper.xml")
    assert len(qs) == 2
    by_acc = {}
    for q in qs:
        for a in q.accesses:
            by_acc.setdefault(a.usage, set()).add((a.table, a.column))
    assert ("t_order", "biz_no") in by_acc.get("where", set())
    assert ("t_order", "status") in by_acc.get("where", set())
    assert ("t_order", "org_code") in by_acc.get("where", set())  # like CONCAT
    assert ("t_order", "gmt_create") in by_acc.get("order", set())
    assert ("t_order", "biz_no") in by_acc.get("group", set())
    # join:别名 o/p 解析回真实表
    assert ("t_order", "org_code") in by_acc.get("join", set())
    assert ("t_no_pk", "code") in by_acc.get("join", set())


def test_fingerprint_stable_across_literals():
    a = fingerprint_sql("SELECT * FROM t WHERE name = 'alice' AND n = 3")
    b = fingerprint_sql("SELECT * FROM t WHERE name = 'bob' AND n = 7")
    assert a == b
    c = fingerprint_sql("SELECT * FROM t WHERE name = 'alice' AND n = 3 AND x = 1")
    assert a != c


# ---------------------------------------------------------------------------
# 规则引擎(端到端)
# ---------------------------------------------------------------------------

@pytest.fixture
def audit_env(tmp_path, monkeypatch):
    from codeatlas import config as config_mod

    monkeypatch.setattr(config_mod, "DB_PATH", tmp_path / "a.sqlite")
    conn = connect()
    init_db(conn)
    persist_ddl(conn, parse_ddl("testdb", OB_DDL))
    f = tmp_path / "OrderMapper.xml"
    f.write_text(MAPPER_XML, encoding="utf-8")
    qs = extract_from_xml(f, "mapper/OrderMapper.xml")
    persist_query_map(conn, 1, qs)
    qmap = [
        dict(r)
        for r in conn.execute(
            "SELECT DISTINCT query_fingerprint, source_file, source_line, "
            "table_name, column_name, usage, freq FROM query_column_map"
        )
    ]
    anti = []
    for q in qs:
        for ap in q.anti_patterns:
            anti.append({**ap, "fingerprint": q.fingerprint,
                         "source_file": q.source_file, "source_line": q.source_line})
    yield conn, qmap, anti
    conn.close()


def test_rules_end_to_end(audit_env):
    conn, qmap, anti = audit_env
    result = run_audit(conn, "testdb", qmap, anti)
    rules = {f.rule_id for f in result.findings}

    assert "TBL001" in rules          # t_no_pk 无主键
    tbl001_tables = {f.table for f in result.findings if f.rule_id == "TBL001"}
    assert "t_no_pk" in tbl001_tables
    assert "t_tmp_scratch" not in tbl001_tables  # temp/tmp 排除约定
    assert "t_tmp_scratch" in result.excluded_tables
    assert "IDX003" in rules          # idx_org_code ⊂ idx_status_org
    assert "IDX004" in rules          # t_no_pk.code join 无索引
    assert "IDX005" in rules          # like CONCAT('%'...) 前导通配
    # 画像缺失 → unavailable
    assert set(result.unavailable) == {"IDX101", "IDX102", "IDX103", "QRY101"}
    # IDX001:t_order 的 creator/gmt_* 等未在条件中,不误报;检查无索引条件列不出现已索引列
    idx001 = [f for f in result.findings if f.rule_id == "IDX001"]
    for f in idx001:
        col = f.evidence["column"]
        assert col not in {"status", "org_code", "biz_no"}  # 这些已有索引


def test_profile_apply_activates_rules(audit_env, tmp_path):
    conn, qmap, anti = audit_env
    tpl = manual_template("testdb", parse_ddl("testdb", OB_DDL), out_dir=tmp_path)
    data = json.loads(tpl.read_text())
    data["tables"]["t_order"]["row_count"] = 1200
    # 6 个索引(pk 不算,idx 有 3)→ 4 个以上?uk+2 idx=3,改加一个再填
    data["tables"]["t_order"]["index_cardinality"] = {}
    mp = tmp_path / "filled.json"
    mp.write_text(json.dumps(data))
    n = apply_profile(conn, "testdb", json.loads(mp.read_text()))
    assert n == 3
    row = conn.execute(
        "SELECT row_count FROM ddl_tables WHERE name='testdb.t_order'"
    ).fetchone()
    assert row["row_count"] == 1200
    # IDX101 需 >4 个索引;t_order 只有 3 个索引(uk+2),不触发——验证不误报
    result = run_audit(conn, "testdb", qmap, anti)
    assert "IDX101" not in {f.rule_id for f in result.findings}
    assert "IDX101" not in result.unavailable  # 画像已有,规则已执行(只是没命中)


def test_generate_scripts_readonly(audit_env, tmp_path):
    p = parse_ddl("testdb", OB_DDL)
    scripts = generate_scripts({"testdb": p}, out_dir=tmp_path)
    text = scripts[0].read_text()
    assert "information_schema" in text
    assert "SHOW INDEX" in text
    assert "GROUP BY" in text  # 低基数列抽样(t_order.status)
    # 只读保障:全文不含任何写操作语句
    for kw in ("INSERT", "UPDATE ", "DELETE", "ALTER", "DROP", "CREATE", "TRUNCATE"):
        assert f" {kw}" not in text.upper().replace("--", ""), kw


def test_fill_sheet_roundtrip(tmp_path, monkeypatch):
    """填报表生成 → 人工填写 → 导入为 manual.json → apply_profile 生效。"""
    import json as _json

    from codeatlas import config as _cfg
    monkeypatch.setattr(_cfg, "DB_PATH", tmp_path / "rt.sqlite")

    from codeatlas.schema.collect_ob import (
        apply_profile,
        generate_fill_sheet,
        import_fill_sheet,
    )

    p = parse_ddl("testdb", OB_DDL)
    sheet = generate_fill_sheet({"testdb": p}, out_dir=tmp_path)
    text = sheet.read_text(encoding="utf-8")
    assert "t_order" in text
    assert "t_tmp_scratch" not in text  # temp/tmp 排除
    assert "import-slow" in text  # 慢查询走文件通道指引

    # 模拟人工填写
    text = text.replace(
        "| t_order |  |  |  |  |  |", "| t_order | 2,300 | 5 | 1 | 90000 | 重点表 |"
    ).replace(
        "# 慢查询不再填本表:请单独导入文件(CSV/JSON/纯文本 SQL 均可):",
        "# 慢查询不再填本表:请单独导入文件(CSV/JSON/纯文本 SQL 均可):",
    )
    filled = tmp_path / "filled_sheet.md"
    filled.write_text(text, encoding="utf-8")
    written = import_fill_sheet(filled, {"testdb": p}, out_dir=tmp_path)
    assert len(written) == 1
    data = _json.loads(written[0].read_text(encoding="utf-8"))
    assert data["tables"]["t_order"]["row_count"] == 2300
    assert data["tables"]["t_order"]["data_length"] == 5

    conn = connect()
    init_db(conn)
    persist_ddl(conn, p)
    apply_profile(conn, "testdb", data)
    row = conn.execute(
        "SELECT row_count, data_length FROM ddl_tables WHERE name='testdb.t_order'"
    ).fetchone()
    assert row["row_count"] == 2300 and row["data_length"] == 5
    conn.close()


def test_slow_query_import_and_qry101(tmp_path, monkeypatch):
    """慢查询文件通道:三种格式解析 + QRY101 规则 + 访问路径并入证据。"""
    from pathlib import Path as P

    from codeatlas.schema.audit import run_audit
    from codeatlas.schema.collect_ob import import_slow_queries, parse_slow_file

    parsed = {"testdb": parse_ddl("testdb", OB_DDL)}

    # JSON 格式(带统计)
    jf = tmp_path / "slow.json"
    jf.write_text(json.dumps([
        {"schema": "testdb", "sql": "SELECT * FROM t_order WHERE status = 'PAID'",
         "freq": 500, "avg_ms": 3200, "rows_scanned": 5000000, "rows_returned": 12},
        {"sql": "SELECT 1", "freq": 1},  # 无法归属(schema 空/表不命中)→ 丢弃
    ]))
    rows = parse_slow_file(jf)
    assert len(rows) == 2 and rows[0]["rows_scanned"] == 5000000
    counts = import_slow_queries(jf, parsed, out_dir=tmp_path)
    assert counts == {"testdb": 1}
    data = json.loads((tmp_path / "testdb_manual.json").read_text())
    assert data["slow_queries"][0]["freq"] == 500

    # CSV 格式(中文表头)
    cf = tmp_path / "slow.csv"
    cf.write_text("库名,sql,频次,平均耗时ms,扫描行数,返回行数\n"
                  "testdb,\"SELECT id FROM t_order WHERE org_code = 'X'\",80,900,900000,80\n")
    counts2 = import_slow_queries(cf, parsed, out_dir=tmp_path)
    assert counts2 == {"testdb": 1}
    data2 = json.loads((tmp_path / "testdb_manual.json").read_text())
    assert len(data2["slow_queries"]) == 2  # 合并不覆盖

    # 纯文本 SQL(无统计)
    tf = tmp_path / "slow.txt"
    tf.write_text("SELECT * FROM t_no_pk WHERE code = 'a'\n\nSELECT count(*) FROM t_order\n")
    counts3 = import_slow_queries(tf, parsed, out_dir=tmp_path)
    assert counts3 == {"testdb": 2}

    # QRY101:ratio>1000 命中
    from codeatlas.db.models import connect as _c, init_db as _i
    from codeatlas import config as _cfg
    monkeypatch.setattr(_cfg, "DB_PATH", tmp_path / "q.sqlite")
    conn = _c(); _i(conn)
    persist_ddl(conn, parsed["testdb"])
    slow = json.loads((tmp_path / "testdb_manual.json").read_text())["slow_queries"]
    result = run_audit(conn, "testdb", [], [], slow_queries=slow)
    qry = [f for f in result.findings if f.rule_id == "QRY101"]
    assert len(qry) == 2  # JSON 条(5e6/12)与 CSV 条(9e5/80=11250)均超线
    ratios = {f.evidence["ratio"] for f in qry}
    assert 416666 in ratios and 11250 in ratios
    assert all(f.evidence["tables"] == ["t_order"] for f in qry)
    # 无统计条目不进 QRY101
    assert all(f.rule_id != "QRY101" or f.evidence.get("ratio") for f in result.findings)
    conn.close()


def test_fill_sheet_no_slow_section(tmp_path):
    """fill_sheet 不再包含慢查询填表段。"""
    from codeatlas.schema.collect_ob import generate_fill_sheet

    sheet = generate_fill_sheet({"testdb": parse_ddl("testdb", OB_DDL)}, out_dir=tmp_path)
    text = sheet.read_text(encoding="utf-8")
    assert "import-slow" in text  # 指引走文件通道
    assert "| 库名 | SQL |" not in text  # 旧慢查询表格段已移除
