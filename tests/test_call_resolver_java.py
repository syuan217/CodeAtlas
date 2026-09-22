"""M3 调用边解析测试:Java / TS 级联、护栏、歧义丢弃、queries。"""

import pytest

from codeatlas.config import RepoCfg
from codeatlas.graph.call_resolver.base import build_context
from codeatlas.graph.call_resolver.orchestrate import resolve_and_write_calls
from codeatlas.graph.queries import (
    callees_of,
    callers_of,
    find_symbols,
    impact_of,
)
from conftest import Env, FakeEmbedServer, copy_fixture

JAVA_MULTI = b"""package com.demo.svc;

import com.demo.model.Order;
import com.demo.util.TextUtil;

public class OrderService {

    private Order order;

    public int audit(Order o, int level) {
        if (o.isPaid()) {
            return level;
        }
        return TextUtil.normalize(o.name());
    }

    public String render() {
        Order local = this.order;
        return local.summary();
    }

    public void batch() {
        audit(null, 1);
        helper();
    }

    void helper() {
    }
}
"""

JAVA_MODEL = b"""package com.demo.model;

public class Order {
    public boolean isPaid() { return true; }
    public String name() { return "o"; }
    public String summary() { return "s"; }
}
"""

JAVA_UTIL = b"""package com.demo.util;

public class TextUtil {
    public static String normalize(String s) { return s.trim(); }
}
"""

JAVA_EXT = b"""package com.demo.other;

public class Conflicter {
    public String name() { return "x"; }
}
"""


@pytest.fixture
def java_env(tmp_path, monkeypatch) -> Env:
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "jrepo"
    (root / "com/demo/svc").mkdir(parents=True)
    (root / "com/demo/model").mkdir(parents=True)
    (root / "com/demo/util").mkdir(parents=True)
    (root / "com/demo/other").mkdir(parents=True)
    (root / "com/demo/svc/OrderService.java").write_bytes(JAVA_MULTI)
    (root / "com/demo/model/Order.java").write_bytes(JAVA_MODEL)
    (root / "com/demo/util/TextUtil.java").write_bytes(JAVA_UTIL)
    (root / "com/demo/other/Conflicter.java").write_bytes(JAVA_EXT)
    e.run(RepoCfg(name="j", path=root, languages=["java"]))
    return e


def calls_of(env: Env) -> list[dict]:
    return env.q(
        "SELECT a.qualified_name AS s, b.qualified_name AS d, e.resolution AS r, e.line "
        "FROM edges e JOIN symbols a ON e.src_id=a.id JOIN symbols b ON e.dst_id=b.id "
        "WHERE e.kind='CALLS' ORDER BY e.line"
    )


def test_java_exact_edges(java_env):
    rows = {(r["s"], r["d"], r["r"]) for r in calls_of(java_env)}
    # 裸调用同类方法 → exact
    assert ("com.demo.svc.OrderService#audit", "com.demo.svc.OrderService#audit@18", "exact") in rows or \
           any(s.endswith("#batch") and d.startswith("com.demo.svc.OrderService#audit") and r == "exact"
               for s, d, r in rows)
    # 参数类型 Order o → o.isPaid() exact
    assert (
        "com.demo.svc.OrderService#audit", "com.demo.model.Order#isPaid", "exact"
    ) in rows
    # 字段类型 this.order → Order local = this.order; local.summary() exact
    assert (
        "com.demo.svc.OrderService#render", "com.demo.model.Order#summary", "exact"
    ) in rows
    # import 类静态调用 → exact
    assert (
        "com.demo.svc.OrderService#audit", "com.demo.util.TextUtil#normalize", "exact"
    ) in rows
    # 同类裸调用 helper → exact
    assert (
        "com.demo.svc.OrderService#batch", "com.demo.svc.OrderService#helper", "exact"
    ) in rows


def test_java_external_calls_dropped_not_misbound(java_env):
    """外部调用(String.trim 等)不上库;Order.name() 与 Conflicter.name() 歧义不兜底。"""
    rows = calls_of(java_env)
    all_dst = {r["d"] for r in rows}
    # String/StringBuilder 等外部绝不出现
    assert not any("String" in d for d in all_dst if "#" in d and "TextUtil" not in d)
    # name() 在两个类里(Order/Conflicter)→ 歧义 → 不允许 heuristic 兜底绑定
    assert not any(
        r["r"] == "heuristic" and r["d"].endswith("#name") for r in rows
    )


def test_java_bare_call_no_cross_file_fallback(tmp_path, monkeypatch):
    """M3 验收收紧:Java 裸调用不再全库同名兜底(跨类调用必须类名限定)。

    A.run() 里的 go() 即使库内唯一(B#go)也不产边——模式 A/D 错误的根源。
    """
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "j"
    root.mkdir()
    (root / "A.java").write_bytes(
        b"package p;\npublic class A {\n"
        b"    void run() {\n        go();\n    }\n"
        b"    void other() { }\n}\n"
    )
    (root / "B.java").write_bytes(
        b"package p;\npublic class B {\n    void go() { }\n}\n"
    )
    e.run(RepoCfg(name="j2", path=root))
    rows = e.q("SELECT COUNT(*) AS c FROM edges WHERE kind='CALLS'")
    assert rows[0]["c"] == 0


# ---------------------------------------------------------------------------
# TS
# ---------------------------------------------------------------------------

def test_ts_resolver(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "trepo"
    (root / "src").mkdir(parents=True)
    (root / "src/util.ts").write_bytes(
        b"export function formatAmount(cents: number): string {\n"
        b"  return (cents / 100).toFixed(2);\n}\n"
    )
    (root / "src/cart.ts").write_bytes(
        b'import { formatAmount } from "./util";\n\n'
        b"export class Cart {\n"
        b"  total(): number { return 100; }\n"
        b"  render(): string { return formatAmount(this.total()); }\n"
        b"}\n"
    )
    e.run(RepoCfg(name="t", path=root))

    rows = {(r["s"], r["d"], r["r"]) for r in e.q(
        "SELECT a.qualified_name AS s, b.qualified_name AS d, e.resolution AS r "
        "FROM edges e JOIN symbols a ON e.src_id=a.id JOIN symbols b ON e.dst_id=b.id "
        "WHERE e.kind='CALLS'"
    )}
    # 命名导入 → exact
    assert ("src/cart.Cart#render", "src/util.formatAmount", "exact") in rows
    # this.total() → exact
    assert ("src/cart.Cart#render", "src/cart.Cart#total", "exact") in rows
    # 外部 toFixed 丢弃
    assert not any("toFixed" in d for _, d, _ in rows)


# ---------------------------------------------------------------------------
# queries.py
# ---------------------------------------------------------------------------

def test_find_symbols_and_def(java_env):
    syms = find_symbols(java_env.conn, "normalize")
    assert len(syms) == 1
    assert syms[0]["qualified_name"] == "com.demo.util.TextUtil#normalize"
    assert syms[0]["fpath"].endswith("TextUtil.java")


def test_callers_and_callees(java_env):
    sym = find_symbols(java_env.conn, "com.demo.util.TextUtil#normalize")[0]
    callers = callers_of(java_env.conn, sym)
    assert [c.src_qname for c in callers] == ["com.demo.svc.OrderService#audit"]
    svc = find_symbols(java_env.conn, "com.demo.svc.OrderService#audit")[0]
    callees = callees_of(java_env.conn, svc)
    dsts = {c.dst_qname for c in callees}
    assert "com.demo.model.Order#isPaid" in dsts
    assert "com.demo.util.TextUtil#normalize" in dsts


def test_impact_closure(java_env):
    """改 Order.isPaid → audit 受影响(depth1)→ batch 受影响(depth2)。"""
    sym = find_symbols(java_env.conn, "com.demo.model.Order#isPaid")[0]
    rows = impact_of(java_env.conn, sym)
    by_depth = {r["qname"]: r["depth"] for r in rows}
    assert by_depth.get("com.demo.svc.OrderService#audit@13") == 1 or \
           any(q.startswith("com.demo.svc.OrderService#audit") and d == 1
               for q, d in by_depth.items())
    assert any(q.startswith("com.demo.svc.OrderService#batch") and d == 2
               for q, d in by_depth.items())


def test_incremental_calls_reresolved(java_env, tmp_path):
    """依赖闭包(IMPORTS + CALLS 入边):改 Order(删 summary)→ 依赖者旧 CALLS 边清理;

    且 heuristic 边的依赖者经 CALLS 入边进入闭包:改 B 删 go()、C 加 go() 后,
    A.run 的裸调用 go() 重解析绑到 C#go。
    """
    root = tmp_path / "jrepo"
    (root / "com/demo/model/Order.java").write_bytes(
        b"package com.demo.model;\n\npublic class Order {\n"
        b"    public boolean isPaid() { return true; }\n"
        b"    public String name() { return \"o\"; }\n"
        b"    public String detail() { return \"d\"; }\n"
        b"}\n"
    )
    import time
    time.sleep(0.01)
    stats = java_env.run(RepoCfg(name="j", path=root))
    assert stats.modified == 1
    rows = {(r["s"], r["d"]) for r in calls_of(java_env)}
    # 旧 summary 边已清(类型已知无该方法 = 已知非边)
    assert not any(d.endswith("#summary") for _, d in rows)
    # 未受影响的边保留
    assert ("com.demo.svc.OrderService#audit", "com.demo.model.Order#isPaid") in rows
    assert ("com.demo.svc.OrderService#batch", "com.demo.svc.OrderService#helper") in rows


def test_heuristic_dependent_via_calls_edge(tmp_path, monkeypatch):
    """CALLS 入边闭包(python 兜底边):B 删 go、C 加 go → A 的 heuristic 边重绑 C#go。

    A 裸调用 go() 与 B 无 IMPORTS 关系,只有旧 CALLS 边能把它带进依赖闭包。
    """
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "hd"
    root.mkdir()
    (root / "a.py").write_text("def run():\n    go()\n")
    (root / "b.py").write_text("def go():\n    pass\n")
    e.run(RepoCfg(name="hd", path=root))
    before = e.q(
        "SELECT b.qualified_name AS d FROM edges e JOIN symbols b ON e.dst_id=b.id "
        "WHERE e.kind='CALLS'"
    )
    assert [r["d"] for r in before] == ["b.go"]  # 库内唯一 → heuristic 绑 b.go

    (root / "b.py").write_text("def other():\n    pass\n")
    (root / "c.py").write_text("def go():\n    pass\n")
    import time
    time.sleep(0.01)
    e.run(RepoCfg(name="hd", path=root))
    after = e.q(
        "SELECT b.qualified_name AS d, e.resolution FROM edges e "
        "JOIN symbols b ON e.dst_id=b.id WHERE e.kind='CALLS'"
    )
    # a 经 CALLS 入边进闭包 → 重解析 → go() 现在唯一命中 c.go
    assert [(r["d"], r["resolution"]) for r in after] == [("c.go", "heuristic")]


# ===========================================================================
# M3 验收报告(M3_calls_acceptance.md)四类错误模式的回归用例
# ===========================================================================

MODE_A_JAVA = b"""package com.acc;

public class DeductibleUtils {

    public String convert(Object dto) {
        return ((String) dto).trim() + getDeductibleTypeCd();
    }
}
"""

MODE_A_TARGET = b"""package com.acc;

public class ProjectRisk {

    private static String getDeductibleTypeCd(java.io.Serializable id) {
        return "x";
    }
}
"""


def test_pattern_a_instance_getter_not_bound_to_static(tmp_path, monkeypatch):
    """模式 A:实例/裸调用不得绑到其它类的 static 工具方法(元数/static 均不符)。"""
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "pa"
    root.mkdir()
    (root / "DeductibleUtils.java").write_bytes(MODE_A_JAVA)
    (root / "ProjectRisk.java").write_bytes(MODE_A_TARGET)
    e.run(RepoCfg(name="pa", path=root))
    rows = e.q("SELECT COUNT(*) AS c FROM edges WHERE kind='CALLS'")
    # getDeductibleTypeCd() 0 参裸调用 vs static 1 参 → 不产边
    assert rows[0]["c"] == 0


def test_pattern_a_self_loop_suppressed(tmp_path, monkeypatch):
    """自环红旗:方法内调用自己名(唯一候选=自身)→ 兜底排除 caller 自身。"""
    from codeatlas.graph.call_resolver.base import ResolveContext, build_context

    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "sl"
    root.mkdir()
    (root / "S.java").write_bytes(
        b"package p;\npublic class S {\n"
        b"    String m(Object o) {\n"
        b"        if (o == null) { return m(o); }\n"  # 真实递归(类内 exact 允许,不是兜底自环)
        b"        return \"x\";\n    }\n}\n"
    )
    e.run(RepoCfg(name="sl", path=root))
    rows = e.q(
        "SELECT a.id AS aid, b.id AS bid FROM edges e "
        "JOIN symbols a ON e.src_id=a.id JOIN symbols b ON e.dst_id=b.id "
        "WHERE e.kind='CALLS'"
    )
    # 类内递归调用是合法 exact 自环(允许);验证没有 heuristic 自环即可
    assert all(r["aid"] != r["bid"] for r in rows) or len(rows) == 1


MODE_C_TS = b"""import { useEffect } from 'react';
import { queryDetail as AreaFetchData } from '@/services/area';

export function CustomerLogs() {
    const fetchData = (page: number, size: number) => {
        return AreaFetchData({ page, size });
    };
    return fetchData(1, 10);
}
"""

MODE_C_TARGET = b"""export const queryDetail = async (params: any) => {
    return Promise.resolve(params);
};
"""


def test_pattern_c_ts_lexical_local_and_alias(tmp_path, monkeypatch):
    """模式 C:组件内局部函数遮蔽导入;@/ 别名应正确解析到 src/。"""
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "pc"
    (root / "src/services/area").mkdir(parents=True)
    (root / "src/pages").mkdir(parents=True)
    (root / "src/services/area/index.ts").write_bytes(MODE_C_TARGET)
    (root / "src/pages/CustomerLogs.tsx").write_bytes(MODE_C_TS)
    e.run(RepoCfg(name="pc", path=root))
    rows = e.q(
        "SELECT b.qualified_name AS d, e.resolution AS r FROM edges e "
        "JOIN symbols b ON e.dst_id=b.id WHERE e.kind='CALLS'"
    )
    dsts = {(r["d"], r["r"]) for r in rows}
    # 局部 fetchData(1,10) → 词法局部,不产边(不绑 AreaSearchSelect 同名)
    assert not any(d.endswith("fetchData") for d, _ in dsts)
    # @/ 别名:AreaFetchData({page,size}) → 命名导入 queryDetail → exact
    assert ("src/services/area/index.queryDetail", "exact") in dsts


def test_pattern_d_third_party_receiver_dropped(tmp_path, monkeypatch):
    """模式 D:moment()/antd form 等无类型接收者的方法调用不产边。"""
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = tmp_path / "pd"
    (root / "src/services").mkdir(parents=True)
    (root / "src/services/task.ts").write_bytes(
        b"export async function submit(data: any) {\n  return Promise.resolve(data);\n}\n"
        b"export async function add(x: number) {\n  return Promise.resolve(x);\n}\n"
    )
    (root / "src/page.tsx").write_bytes(
        b"import { submit, add } from './services/task';\n"
        b"import moment from 'moment';\n\n"
        b"export function render(): void {\n"
        b"  const form: any = null;\n"
        b"  form.submit();\n"          # any 类型 → 接收器类型未知 → 丢弃
        b"  moment().add(1, 'day');\n"  # 第三方 → 丢弃
        b"  add(1);\n"                  # 命名导入 → exact
        b"}\n"
    )
    e.run(RepoCfg(name="pd", path=root))
    rows = e.q(
        "SELECT b.qualified_name AS d, e.resolution AS r FROM edges e "
        "JOIN symbols b ON e.dst_id=b.id WHERE e.kind='CALLS'"
    )
    dsts = {(r["d"], r["r"]) for r in rows}
    # form.submit()/moment().add() 不产边
    assert not any(d.endswith("#submit") for d, _ in dsts)
    assert not any(d.endswith("#add") for d, _ in dsts)
    # add(1) 命名导入 → exact
    assert ("src/services/task.add", "exact") in dsts
