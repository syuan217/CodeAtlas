"""symbols.py:多语言符号提取、qualified_name、parent/CONTAINS 数据。"""

from pathlib import Path

from codeatlas.ingest.symbols import detect_language, extract_symbols

FIXTURES = Path(__file__).parent / "fixtures"


def extract(rel: str, root: str):
    p = FIXTURES / root / rel
    return extract_symbols(rel, detect_language(rel), p.read_bytes())


def syms_by_qname(fs):
    return {s.qualified_name: s for s in fs.symbols}


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

def test_java_class_and_methods():
    fs = extract("com/example/service/UserService.java", "java_mini")
    m = syms_by_qname(fs)
    assert "com.example.service.UserService" in m
    assert "com.example.service.UserService#validate" in m
    assert "com.example.service.UserService#summarize" in m
    cls = m["com.example.service.UserService"]
    assert (cls.line_start, cls.line_end) == (6, 45)
    assert cls.parent is None
    val = m["com.example.service.UserService#validate"]
    assert val.kind == "method"
    assert val.parent == fs.symbols.index(cls)
    assert val.signature.startswith("public boolean validate(User user)")
    assert fs.package == "com.example.service"
    assert set(fs.imports) == {"com.example.model.User", "com.example.util.Strings"}


def test_java_javadoc_attached():
    fs = extract("com/example/model/User.java", "java_mini")
    cls = syms_by_qname(fs)["com.example.model.User"]
    assert cls.javadoc_start == 3  # /** User entity. */ 起始行


def test_java_overload_disambiguation(tmp_path):
    src = b"""package p;

public class Demo {
    int calc(int a) { return a; }
    int calc(int a, int b) { return a + b; }
}
"""
    fs = extract_symbols("Demo.java", "java", src)
    qnames = {s.qualified_name for s in fs.symbols if "#" in s.qualified_name}
    assert len(qnames) == 2  # 重载不撞 UNIQUE 索引
    assert any("@5" in q for q in qnames)
    assert "p.Demo#calc" in qnames


def test_java_nested_class():
    src = b"""package p;

public class Outer {
    public class Inner {
        void go() {}
    }
}
"""
    fs = extract_symbols("Outer.java", "java", src)
    m = syms_by_qname(fs)
    assert "p.Outer.Inner" in m
    assert "p.Outer.Inner#go" in m
    inner = m["p.Outer.Inner"]
    assert inner.parent == fs.symbols.index(m["p.Outer"])


def test_java_module_symbol():
    fs = extract("com/example/model/User.java", "java_mini")
    assert fs.module.kind == "module"
    assert fs.module.qualified_name == "com/example/model/User.java"


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def test_python_symbols():
    fs = extract("pkg/mod.py", "py_mini")
    m = syms_by_qname(fs)
    assert m["pkg.mod.clamp"].kind == "function"
    assert m["pkg.mod.Counter"].kind == "class"
    assert m["pkg.mod.Counter#bump"].kind == "method"
    assert m["pkg.mod.Counter#bump"].parent == fs.symbols.index(m["pkg.mod.Counter"])
    assert (m["pkg.mod.Counter#bump"].line_start, m["pkg.mod.Counter#bump"].line_end) == (15, 17)


def test_python_imports_absolute_and_from():
    fs = extract("main.py", "py_mini")
    assert "pkg.mod" in fs.imports


def test_python_relative_import_level():
    src = b"from ..pkg.mod import x\n"
    fs = extract_symbols("a/b/c.py", "python", src)
    assert fs.imports == ["..pkg.mod"]


def test_python_nested_function_not_extracted():
    src = b"""def outer():
    def inner():
        pass
    return inner
"""
    fs = extract_symbols("m.py", "python", src)
    assert [s.name for s in fs.symbols] == ["outer"]


# ---------------------------------------------------------------------------
# TS / JS
# ---------------------------------------------------------------------------

def test_ts_symbols():
    fs = extract("src/order.ts", "ts_mini")
    m = syms_by_qname(fs)
    assert m["src/order.OrderLine"].kind == "interface"
    assert m["src/order.Cart"].kind == "class"
    assert m["src/order.Cart#add"].kind == "method"
    assert m["src/order.renderCart"].kind == "function"
    assert m["src/order.Cart#totalCents"].parent == fs.symbols.index(m["src/order.Cart"])
    assert fs.imports == ["./util"]


def test_ts_arrow_function_variable():
    src = b'const greet = (name: string) => `hi ${name}`;\nexport const bye = function () {};\n'
    fs = extract_symbols("src/g.ts", "tsx", src)
    m = syms_by_qname(fs)
    assert m["src/g.greet"].kind == "function"
    assert m["src/g.bye"].kind == "function"


def test_jsx_uses_tsx_grammar():
    src = b"export const App = () => <div>hi</div>;\n"
    fs = extract_symbols("src/App.jsx", "tsx", src)
    assert any(s.name == "App" for s in fs.symbols)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

def test_go_symbols():
    src = b"""package main

import "fmt"

type Cart struct {
    lines []string
}

type Printer interface {
    Print()
}

func total(c *Cart) int { return len(c.lines) }

func (c *Cart) add(line string) { c.lines = append(c.lines, line) }

func main() { fmt.Println("x") }
"""
    fs = extract_symbols("main.go", "go", src)
    m = syms_by_qname(fs)
    assert m["main.Cart"].kind == "class"
    assert m["main.Printer"].kind == "interface"
    assert m["main.total"].kind == "function"
    assert m["main.Cart#add"].kind == "method"
    assert fs.imports == ["fmt"]


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

def test_xml_markdown_only_module():
    fs = extract_symbols("mapper/UserMapper.xml", "xml", b"<mapper></mapper>")
    assert fs.symbols == []
    assert fs.module.qualified_name == "mapper/UserMapper.xml"
    md = extract_symbols("README.md", "markdown", b"# t\n\ntext\n")
    assert md.symbols == []


def test_detect_language():
    assert detect_language("a/b/C.java") == "java"
    assert detect_language("x.tsx") == "tsx"
    assert detect_language("x.jsx") == "tsx"
    assert detect_language("x.py") == "python"
    assert detect_language("x.txt") is None


def test_invalid_source_returns_module_only():
    fs = extract_symbols("Bad.java", "java", b"\x00\x01 not java at all {{{")
    # 解析不炸、不崩溃;符号可能为空,但 module 存在
    assert fs.module.kind == "module"


def test_contains_edges_shape():
    fs = extract("com/example/model/User.java", "java_mini")
    edges = fs.contains_edges()
    # module→class 是 (None, class_idx);class→method 是 (0, method_idx)
    assert (None, 0) in edges  # User 类挂 module
    assert (0, 1) in edges and (0, 2) in edges and (0, 3) in edges


# ---------------------------------------------------------------------------
# 调用点 caller 归属(注解方法 / 箭头函数 / 类字段箭头函数)
# ---------------------------------------------------------------------------

def _callers_by_name(fs, sites) -> dict[str, set]:
    out: dict[str, set] = {}
    for s in sites:
        q = fs.symbols[s.caller_local].qualified_name if s.caller_local is not None else None
        out.setdefault(s.name, set()).add(q)
    return out


def test_java_annotated_method_calls_attributed_to_method():
    """注解使方法节点起点落在注解行;caller 必须按签名行对齐到方法,而非落到 module。"""
    from codeatlas.ingest.symbols import extract_call_sites

    src = b"""package com.x;
public class S {
    @Override
    public String toString() { return helper(); }

    @Transactional
    @SuppressWarnings("unchecked")
    public void save() {
        audit();
    }

    @Autowired
    public S() { init(); }

    private String helper() { return "a"; }
    private void audit() {}
    private void init() {}
}
"""
    fs = extract_symbols("com/x/S.java", "java", src)
    by_name = _callers_by_name(fs, extract_call_sites("com/x/S.java", "java", src, fs))
    assert by_name["helper"] == {"com.x.S#toString"}
    assert by_name["audit"] == {"com.x.S#save"}
    assert by_name["init"] == {"com.x.S#S"}


def test_ts_arrow_and_class_field_calls_attributed():
    from codeatlas.ingest.symbols import extract_call_sites

    src = b"""import { api } from "./api";
export const Index = (props: Props) => {
  const onOk = () => { api.save(props.id); };
  return load();
};
const helper = function () { return fetchAll(); };
class Page extends Component {
  handleClick = (e) => { this.props.dispatch(e); };
  render() { return helper(); }
}
boot();
"""
    fs = extract_symbols("src/page.tsx", "tsx", src)
    m = syms_by_qname(fs)
    assert m["src/page.Page#handleClick"].kind == "method"  # 类字段箭头函数 = 方法
    by_name = _callers_by_name(fs, extract_call_sites("src/page.tsx", "tsx", src, fs))
    assert by_name["save"] == {"src/page.Index"}       # 嵌套箭头函数归外层符号
    assert by_name["load"] == {"src/page.Index"}
    assert by_name["fetchAll"] == {"src/page.helper"}
    assert by_name["dispatch"] == {"src/page.Page#handleClick"}
    assert by_name["helper"] == {"src/page.Page#render"}
    assert by_name["boot"] == {None}                    # 真正的模块级调用


def test_gbk_source_parsed_not_crashing():
    """GBK 编码(国内老 Java 项目常见)源文件:不得 UnicodeDecodeError 中断整次索引。"""
    from codeatlas.ingest.symbols import decode_source, extract_call_sites

    src = ('package com.x;\npublic class G {\n    // 中文注释\n'
           '    public boolean ok(String x) { return "中文".equals(x); }\n}\n').encode("gbk")
    assert "中文注释" in decode_source(src)
    fs = extract_symbols("com/x/G.java", "java", src)
    assert "com.x.G#ok" in syms_by_qname(fs)
    sites = extract_call_sites("com/x/G.java", "java", src, fs)
    assert [s.name for s in sites] == ["equals"]
