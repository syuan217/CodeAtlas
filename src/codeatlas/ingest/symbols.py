"""tree-sitter 符号提取(PLAN §9.1)。

多语言注册表:扩展名 → 语言 → AST 抽取规则。
产出 FileSymbols:module 符号(每文件一个,图遍历的文件入口)+
class/function/method/interface 符号(含 parent 索引,CONTAINS 边据此构建)+
package/import 原始信息(IMPORTS 两阶段解析的 Pass1 输入)。

qualified_name 约定:容器层级用 '.',方法用 '#'(如 com.x.OrderService#create,
PLAN §7 示例);module 符号的 qualified_name = 仓库内相对路径(全局唯一)。
同文件内重名(如 Java 重载)追加 "@行号" 消歧,保证 UNIQUE(repo_id, qualified_name)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter_language_pack import get_parser

from codeatlas.graph.call_resolver.base import CallSite, ImportBinding

EXT_LANG: dict[str, str] = {
    ".java": "java",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "tsx",
    ".ts": "typescript", ".tsx": "tsx",
    ".py": "python",
    ".go": "go",
    ".sql": "sql",
    ".md": "markdown", ".markdown": "markdown",
    ".xml": "xml",
}

# 首版支持符号提取的语言;其余(xml/sql/markdown)只建 module 符号
SYMBOL_LANGS = {"java", "javascript", "typescript", "tsx", "python", "go"}

_COMMENT_TYPES = {"block_comment", "comment", "line_comment", "doc_comment"}


def detect_language(rel: str) -> str | None:
    return EXT_LANG.get(Path(rel).suffix.lower())


@dataclass
class Symbol:
    kind: str  # module / class / interface / function / method
    name: str
    qualified_name: str
    line_start: int  # 1-based,含签名行
    line_end: int
    signature: str | None
    parent: int | None  # FileSymbols.symbols 内的父符号索引;None = 挂 module
    javadoc_start: int | None = None  # 紧邻注释块起始行(class chunk 用)


@dataclass
class FileSymbols:
    rel: str
    lang: str
    module: Symbol
    symbols: list[Symbol] = field(default_factory=list)
    package: str | None = None  # java 包声明
    imports: list[str] = field(default_factory=list)  # 原始 import 目标(IMPORTS 边用)
    import_bindings: list[ImportBinding] = field(default_factory=list)  # 名字级绑定(M3 调用边用)

    def contains_edges(self) -> list[tuple[int | None, int]]:
        """CONTAINS 边:(parent_symbol_idx or None→module, child_symbol_idx)。"""
        return [(s.parent, i) for i, s in enumerate(self.symbols)]


def _signature(lines: list[str], row0: int) -> str | None:
    if 0 <= row0 < len(lines):
        sig = lines[row0].strip()
        return sig[:200] if sig else None
    return None


def _comment_start(node, lines) -> int | None:
    """符号上方紧邻(中间无空行)的注释块起始行。"""
    prev = node.prev_named_sibling
    if (
        prev is not None
        and prev.type in _COMMENT_TYPES
        and prev.end_point[0] + 1 == node.start_point[0]
    ):
        return prev.start_point[0] + 1
    return None


def _range(node) -> tuple[int, int, int, int]:
    sr, sc = node.start_point
    er, ec = node.end_point
    return sr, sc, er, ec


class _Collector:
    """按语言共用的符号收集器(消除 dataclass 不可变限制外的重复)。"""

    def __init__(self, lines: list[str]):
        self.lines = lines
        self.symbols: list[Symbol] = []

    def add(
        self,
        node,
        kind: str,
        name: str,
        qualified_name: str,
        parent: int | None,
        *,
        with_comment: bool = False,
        line_override: int | None = None,
    ) -> int:
        sr, _, er, _ = _range(node)
        if line_override is not None:
            # Java 注解使节点起点落在注解行;行号/签名统一用签名行(验收 #27/#28/#33)
            sr = line_override - 1
        sym = Symbol(
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            line_start=sr + 1,
            line_end=er + 1,
            signature=_signature(self.lines, sr),
            parent=parent,
            javadoc_start=_comment_start(node, self.lines) if with_comment else None,
        )
        self.symbols.append(sym)
        return len(self.symbols) - 1

    def dedup_qnames(self) -> None:
        """同文件重名符号(Java 重载等)加 @行号 后缀,保证唯一索引安全。"""
        seen: set[str] = set()
        for s in self.symbols:
            q = s.qualified_name
            if q in seen:
                s.qualified_name = f"{q}@{s.line_start}"
            seen.add(s.qualified_name)


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

_JAVA_TYPES = {
    "class_declaration": "class",
    "enum_declaration": "class",
    "record_declaration": "class",
    "interface_declaration": "interface",
}
# 透明容器:递归穿透但本身不产生符号(方法藏在 class_body 内)
_JAVA_TRANSPARENT = {"class_body", "interface_body", "enum_body", "record_body"}


def _extract_java(
    tree, lines: list[str]
) -> tuple[_Collector, str | None, list[str], list[ImportBinding]]:
    col = _Collector(lines)
    package: str | None = None
    imports: list[str] = []
    bindings: list[ImportBinding] = []

    def walk(node, prefix: str, parent: int | None) -> None:
        nonlocal package
        for child in node.named_children:
            t = child.type
            if t == "package_declaration" and package is None:
                for c in child.children:
                    if c.type in ("scoped_identifier", "identifier"):
                        package = c.text.decode("utf-8")
                        break
            elif t == "import_declaration":
                is_static = any(c.type == "static" for c in child.children)
                for c in child.children:
                    if c.type in ("scoped_identifier", "identifier"):
                        fq = c.text.decode("utf-8")
                        imports.append(fq)
                        parts = fq.split(".")
                        if is_static and len(parts) >= 2:
                            # static a.b.C.m → 绑定 m,来源类 a.b.C
                            bindings.append(
                                ImportBinding(parts[-1], ".".join(parts[:-1]), "named")
                            )
                        else:
                            bindings.append(ImportBinding(parts[-1], fq, "class"))
                        break
            elif t in _JAVA_TYPES:
                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8")
                qname = f"{prefix}{name}"
                idx = col.add(child, _JAVA_TYPES[t], name, qname, parent,
                              with_comment=True, line_override=name_node.start_point[0] + 1)
                walk(child, f"{qname}.", idx)
            elif t == "method_declaration" or t == "constructor_declaration":
                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8")
                kind = "method"
                qname = f"{prefix.rstrip('.') or 'root'}#{name}"
                col.add(child, kind, name, qname, parent,
                        line_override=name_node.start_point[0] + 1)
            elif t in _JAVA_TRANSPARENT:
                walk(child, prefix, parent)

    walk(tree.root_node, "", None)
    if package:
        # 顶层类前缀补包名
        prefix = f"{package}."
        for s in col.symbols:
            if "#" not in s.qualified_name:
                s.qualified_name = prefix + s.qualified_name
            else:
                cls, _, m = s.qualified_name.partition("#")
                s.qualified_name = f"{prefix}{cls}#{m}"
    col.dedup_qnames()
    return col, package, imports, bindings


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

_PY_FROM_RE = re.compile(r"^from\s+(\.+)?([\w.]*)\s+import\s+(.+)$")
_PY_IMPORT_RE = re.compile(r"^import\s+([\w.,\s]+)")
# block 是 class body(要穿透)也是函数体(函数分支不递归,天然剪枝)
_PY_TRANSPARENT = {"block"}


def _extract_python(
    tree, lines: list[str], rel: str
) -> tuple[_Collector, list[str], list[ImportBinding]]:
    col = _Collector(lines)
    mod_qname = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
    imports: list[str] = []
    bindings: list[ImportBinding] = []

    def walk(node, prefix: str, parent: int | None, in_class: bool) -> None:
        for child in node.named_children:
            t = child.type
            if t == "import_from_statement":
                m = _PY_FROM_RE.match(child.text.decode("utf-8"))
                if m:
                    level, mod, names = m.group(1) or "", m.group(2), m.group(3)
                    imports.append(f"{level}{mod}")
                    for part in names.split(","):
                        part = part.strip()
                        if not part or part == "*":
                            continue
                        original = part
                        if " as " in part:
                            original = part.split(" as ")[0].strip()
                            part = part.split(" as ")[1].strip()
                        bindings.append(
                            ImportBinding(
                                part, f"{level}{mod}", "named",
                                target_name=original if original != part else None,
                            )
                        )
            elif t == "import_statement":
                m = _PY_IMPORT_RE.match(child.text.decode("utf-8"))
                if m:
                    for part in m.group(1).split(","):
                        part = part.strip()
                        if not part:
                            continue
                        if " as " in part:
                            mod, _, alias = part.partition(" as ")
                            mod, alias = mod.strip(), alias.strip()
                            bindings.append(ImportBinding(alias, mod, "module"))
                        else:
                            bindings.append(ImportBinding(part, part, "module"))
                        imports.append(part)
            elif t == "class_definition":
                name = child.child_by_field_name("name").text.decode("utf-8")
                qname = f"{prefix}{name}"
                idx = col.add(child, "class", name, qname, parent, with_comment=True)
                walk(child, f"{qname}.", idx, True)
            elif t == "function_definition":
                name = child.child_by_field_name("name").text.decode("utf-8")
                if in_class:
                    qname = f"{prefix.rstrip('.')}#{name}"
                    col.add(child, "method", name, qname, parent)
                else:
                    qname = f"{prefix}{name}"
                    col.add(child, "function", name, qname, parent)
            elif t == "decorated_definition":
                walk(child, prefix, parent, in_class)  # 取内层 definition,装饰器行忽略
            elif t in _PY_TRANSPARENT:
                walk(child, prefix, parent, in_class)

    walk(tree.root_node, f"{mod_qname}.", None, False)
    col.dedup_qnames()
    return col, imports, bindings


# ---------------------------------------------------------------------------
# JS / TS / TSX / JSX
# ---------------------------------------------------------------------------

_TS_CLASSY = {
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "enum_declaration": "class",
}
_TS_FUNCY = {"function_declaration", "generator_function_declaration"}
_TS_TRANSPARENT = {"class_body", "interface_body", "enum_body"}


def _extract_ts(
    tree, lines: list[str], rel: str
) -> tuple[_Collector, list[str], list[ImportBinding]]:
    col = _Collector(lines)
    mod_qname = str(Path(rel).with_suffix(""))
    imports: list[str] = []
    bindings: list[ImportBinding] = []

    def import_source(node) -> str | None:
        src_node = node.child_by_field_name("source")
        if src_node is not None:
            return src_node.text.decode("utf-8").strip("'\"")
        return None

    def collect_bindings(node, source: str) -> None:
        for c in node.named_children:
            if c.type == "import_clause":
                for ic in c.named_children:
                    if ic.type == "identifier":  # 默认导入
                        bindings.append(ImportBinding(ic.text.decode(), source, "named"))
                    elif ic.type == "namespace_import":  # * as U
                        n = ic.child_by_field_name("name")
                        if n is not None:
                            bindings.append(
                                ImportBinding(n.text.decode(), source, "module")
                            )
                    elif ic.type == "named_imports":
                        for spec in ic.named_children:
                            if spec.type == "import_specifier":
                                alias = spec.child_by_field_name("alias")
                                name = spec.child_by_field_name("name")
                                if alias is not None and name is not None:
                                    # import {原名 as 别名}:绑定别名,记忆原名
                                    bindings.append(
                                        ImportBinding(
                                            alias.text.decode(), source, "named",
                                            target_name=name.text.decode(),
                                        )
                                    )
                                elif name is not None:
                                    bindings.append(
                                        ImportBinding(name.text.decode(), source, "named")
                                    )
            elif c.type == "export_specifier":
                n = c.child_by_field_name("name")
                if n is not None:
                    bindings.append(ImportBinding(n.text.decode(), source, "named"))

    def walk(node, prefix: str, parent: int | None, in_class: bool) -> None:
        for child in node.named_children:
            t = child.type
            if t == "import_statement":
                src = import_source(child)
                if src:
                    imports.append(src)
                    collect_bindings(child, src)
            elif t == "export_statement":
                src = import_source(child)  # export ... from "./x"
                if src:
                    imports.append(src)
                    collect_bindings(child, src)
                walk(child, prefix, parent, in_class)
            elif t in _TS_CLASSY:
                name = child.child_by_field_name("name").text.decode("utf-8")
                qname = f"{prefix}{name}"
                idx = col.add(child, _TS_CLASSY[t], name, qname, parent, with_comment=True)
                walk(child, f"{qname}.", idx, True)
            elif t == "interface_declaration":
                name = child.child_by_field_name("name").text.decode("utf-8")
                qname = f"{prefix}{name}"
                idx = col.add(child, "interface", name, qname, parent)
                walk(child, f"{qname}.", idx, True)
            elif t in _TS_FUNCY:
                name = child.child_by_field_name("name").text.decode("utf-8")
                if in_class:
                    qname = f"{prefix.rstrip('.')}#{name}"
                    col.add(child, "method", name, qname, parent)
                else:
                    col.add(child, "function", name, f"{prefix}{name}", parent)
            elif t == "method_definition":
                name = child.child_by_field_name("name").text.decode("utf-8")
                qname = f"{prefix.rstrip('.')}#{name}"
                col.add(child, "method", name, qname, parent)
            elif t == "lexical_declaration":
                # const fn = () => {} / function 表达式(顶层赋值函数)
                for decl in child.named_children:
                    if decl.type != "variable_declarator":
                        continue
                    name_node = decl.child_by_field_name("name")
                    value = decl.child_by_field_name("value")
                    if name_node is None or value is None:
                        continue
                    if value.type in ("arrow_function", "function_expression", "function"):
                        name = name_node.text.decode("utf-8")
                        col.add(decl, "function", name, f"{prefix}{name}", parent)
            elif t in _TS_TRANSPARENT:
                walk(child, prefix, parent, in_class)

    walk(tree.root_node, f"{mod_qname}.", None, False)
    col.dedup_qnames()
    return col, imports, bindings


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

_GO_RECV_RE = re.compile(r"[\w.\s()*]+\)\s*$")


def _extract_go(
    tree, lines: list[str], rel: str
) -> tuple[_Collector, list[str], list[ImportBinding]]:
    col = _Collector(lines)
    mod_qname = str(Path(rel).with_suffix(""))
    imports: list[str] = []
    bindings: list[ImportBinding] = []

    def walk(node, prefix: str, parent: int | None) -> None:
        for child in node.named_children:
            t = child.type
            if t == "import_declaration":
                for spec in child.named_children:
                    if spec.type == "import_spec":
                        path = spec.child_by_field_name("path")
                        if path is not None:
                            p = path.text.decode("utf-8").strip("`\"'")
                            imports.append(p)
                            short = p.rstrip("/").rsplit("/", 1)[-1]
                            bindings.append(ImportBinding(short, p, "module"))
            elif t == "function_declaration":
                name = child.child_by_field_name("name").text.decode("utf-8")
                col.add(child, "function", name, f"{prefix}.{name}", parent)
            elif t == "method_declaration":
                name = child.child_by_field_name("name").text.decode("utf-8")
                recv = child.child_by_field_name("receiver")
                recv_text = recv.text.decode("utf-8") if recv is not None else ""
                m = re.search(r"(\w+)\s*$", recv_text.rstrip(")").strip())
                owner = m.group(1) if m else "unknown"
                col.add(child, "method", name, f"{prefix}.{owner}#{name}", parent)
            elif t == "type_declaration":
                for spec in child.named_children:
                    if spec.type != "type_spec":
                        continue
                    name_node = spec.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = name_node.text.decode("utf-8")
                    kind = "class"
                    for sub in spec.named_children:
                        if sub.type == "interface_type":
                            kind = "interface"
                    col.add(spec, kind, name, f"{prefix}.{name}", parent, with_comment=True)

    walk(tree.root_node, mod_qname, None)
    col.dedup_qnames()
    return col, imports, bindings


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _module(rel: str) -> Symbol:
    return Symbol(
        kind="module",
        name=Path(rel).name,
        qualified_name=rel,
        line_start=1,
        line_end=1,
        signature=None,
        parent=None,
    )


def extract_symbols(rel: str, lang: str, source: bytes) -> FileSymbols:
    """解析单个文件;解析失败/无符号语言返回仅含 module 的结果。"""
    fs = FileSymbols(rel=rel, lang=lang, module=_module(rel))
    if lang not in SYMBOL_LANGS:
        return fs
    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines()
    try:
        parser = get_parser(lang)
        tree = parser.parse(source)
    except Exception:
        return fs

    if lang == "java":
        col, package, imports, bindings = _extract_java(tree, lines)
        fs.package = package
        fs.imports = imports
        fs.import_bindings = bindings
    elif lang == "python":
        col, imports, bindings = _extract_python(tree, lines, rel)
        fs.imports = imports
        fs.import_bindings = bindings
    elif lang in ("javascript", "typescript", "tsx"):
        col, imports, bindings = _extract_ts(tree, lines, rel)
        fs.imports = imports
        fs.import_bindings = bindings
    elif lang == "go":
        col, imports, bindings = _extract_go(tree, lines, rel)
        fs.imports = imports
        fs.import_bindings = bindings
    else:  # pragma: no cover
        return fs
    fs.symbols = col.symbols
    return fs


# ===========================================================================
# 调用点提取(M3,Pass1)
#
# caller 归属:调用点行号落在哪个最小行区间的 method/function 符号内
# (嵌套方法自然取内层);变量表为方法级扁平作用域(Java/TS 实务够用),
# 接收器类型推断只做保守三源:参数类型 / 类字段类型 / 局部声明类型。
# ===========================================================================

_CALLER_KINDS = ("method", "function")


def _caller_index(fs: FileSymbols) -> dict[int, int]:
    """{line_start: 局部索引} 供 AST 方法节点行号对齐符号。"""
    return {
        s.line_start: i
        for i, s in enumerate(fs.symbols)
        if s.kind in _CALLER_KINDS
    }


def _norm_type(type_text: str) -> str:
    """类型文本归一:剥泛型/数组/通配,取末段短名。"""
    t = type_text.strip()
    for sep in ("<", "["):
        t = t.split(sep)[0]
    t = t.strip()
    return t.rsplit(".", 1)[-1] if t else t


def _calls_java(tree, source: bytes, fs: FileSymbols) -> list[CallSite]:
    sites: list[CallSite] = []
    caller_at = _caller_index(fs)

    def walk(node, fields: dict, locals_: dict, caller: int | None):
        for child in node.named_children:
            t = child.type
            if t in ("class_declaration", "interface_declaration",
                     "enum_declaration", "record_declaration"):
                sub_fields = dict(fields)
                for body in child.named_children:
                    if body.type in ("class_body", "interface_body", "enum_body", "record_body"):
                        for fd in body.named_children:
                            if fd.type == "field_declaration":
                                vtype = _java_field_type(fd)
                                for vd in fd.named_children:
                                    if vd.type == "variable_declarator":
                                        n = vd.child_by_field_name("name")
                                        if n is not None and vtype:
                                            sub_fields[n.text.decode()] = vtype
                walk(child, sub_fields, locals_, caller)
            elif t in ("method_declaration", "constructor_declaration"):
                local_vars = dict(locals_)
                params = child.child_by_field_name("parameters")
                if params is not None:
                    for p in params.named_children:
                        n = p.child_by_field_name("name")
                        ty = p.child_by_field_name("type")
                        if n is not None and ty is not None:
                            local_vars[n.text.decode()] = _norm_type(ty.text.decode())
                idx = caller_at.get(child.start_point[0] + 1)
                walk(child, fields, local_vars, idx if idx is not None else caller)
            elif t == "local_variable_declaration":
                vt = _java_field_type(child)
                for vd in child.named_children:
                    if vd.type == "variable_declarator":
                        n = vd.child_by_field_name("name")
                        if n is not None and vt:
                            locals_[n.text.decode()] = vt  # 原地传播到兄弟节点
                walk(child, fields, locals_, caller)
            elif t == "method_invocation":
                name_n = child.child_by_field_name("name")
                obj = child.child_by_field_name("object")
                if name_n is not None:
                    recv = obj.text.decode() if obj is not None else None
                    sites.append(CallSite(
                        caller_local=caller,
                        name=name_n.text.decode(),
                        receiver=recv,
                        receiver_type=(
                            locals_.get(recv) or fields.get(recv)
                            if recv and recv not in ("this", "super")
                            else None
                        ),
                        arg_count=_java_argc(child),
                        kind="super" if (recv == "super") else "call",
                        line=child.start_point[0] + 1,
                        col=child.start_point[1] + 1,
                    ))
                walk(child, fields, locals_, caller)
            elif t == "object_creation_expression":
                ty = child.child_by_field_name("type")
                if ty is not None:
                    sites.append(CallSite(
                        caller_local=caller,
                        name=ty.text.decode(),
                        receiver=None,
                        receiver_type=None,
                        arg_count=_java_argc(child),
                        kind="new",
                        line=child.start_point[0] + 1,
                        col=child.start_point[1] + 1,
                    ))
                walk(child, fields, locals_, caller)
            else:
                walk(child, fields, locals_, caller)

    def _java_field_type(fd) -> str | None:
        ty = fd.child_by_field_name("type")
        return _norm_type(ty.text.decode()) if ty is not None else None

    def _java_argc(node) -> int | None:
        args = node.child_by_field_name("arguments")
        if args is None:
            return None
        return sum(1 for a in args.named_children
                   if a.type not in (".", ",", "(", ")"))

    walk(tree.root_node, {}, {}, None)
    return sites


def _calls_ts(tree, source: bytes, fs: FileSymbols) -> list[CallSite]:
    sites: list[CallSite] = []
    caller_at = _caller_index(fs)
    lex_names: set[str] = set()  # caller 词法域名字(参数/局部 const 与函数)

    def type_of_annotation(node) -> str | None:
        ann = node.child_by_field_name("type_annotation")
        if ann is None:  # 部分语法把注解作为普通子节点
            for c in node.named_children:
                if c.type == "type_annotation":
                    ann = c
                    break
        if ann is None:
            return None
        return _norm_type(ann.text.decode().lstrip(":").strip())

    def walk(node, vars_: dict, caller: int | None, lex: set[str]):
        for child in node.named_children:
            t = child.type
            if t in _TS_FUNCY or t == "method_definition":
                sub = dict(vars_)
                sub_lex = set(lex)  # 词法名字进入函数作用域副本
                params = child.child_by_field_name("parameters")
                if params is not None:
                    for p in params.named_children:
                        n = p.child_by_field_name("pattern")
                        if n is not None:
                            sub_lex.add(_ts_pattern_name(n))
                        ty = type_of_annotation(p)
                        if n is not None and ty:
                            sub[_ts_pattern_name(n)] = ty
                idx = caller_at.get(child.start_point[0] + 1)
                walk(child, sub, idx if idx is not None else caller, sub_lex)
            elif t in ("class_declaration", "abstract_class_declaration"):
                sub = dict(vars_)
                for body in child.named_children:
                    if body.type == "class_body":
                        for pd in body.named_children:
                            if pd.type in ("public_field_definition", "property_definition"):
                                n = pd.child_by_field_name("name")
                                ty = type_of_annotation(pd)
                                if n is not None and ty:
                                    sub[n.text.decode()] = ty
                walk(child, sub, caller, lex)
            elif t == "lexical_declaration":
                for decl in child.named_children:
                    if decl.type != "variable_declarator":
                        continue
                    n = decl.child_by_field_name("name")
                    if n is None:
                        continue
                    lex.add(_ts_pattern_name(n))
                    ty = type_of_annotation(decl)
                    val = decl.child_by_field_name("value")
                    if ty is None and val is not None and val.type == "new_expression":
                        ctor = val.child_by_field_name("constructor")
                        if ctor is not None:
                            ty = _norm_type(ctor.text.decode())
                    if ty:
                        vars_[_ts_pattern_name(n)] = ty  # 原地传播到兄弟节点
                walk(child, vars_, caller, lex)
            elif t == "call_expression":
                fn = child.child_by_field_name("function")
                recv, name = _ts_call_target(fn)
                if name:
                    sites.append(CallSite(
                        caller_local=caller,
                        name=name,
                        receiver=recv,
                        receiver_type=vars_.get(recv) if recv else None,
                        arg_count=_ts_argc(child),
                        kind="call",
                        line=child.start_point[0] + 1,
                        col=child.start_point[1] + 1,
                        local_def=(recv is None and name in lex),
                    ))
                walk(child, vars_, caller, lex)
            elif t == "new_expression":
                ctor = child.child_by_field_name("constructor")
                if ctor is not None:
                    sites.append(CallSite(
                        caller_local=caller,
                        name=ctor.text.decode().split("<")[0],
                        receiver=None,
                        receiver_type=None,
                        arg_count=_ts_argc(child),
                        kind="new",
                        line=child.start_point[0] + 1,
                        col=child.start_point[1] + 1,
                    ))
                walk(child, vars_, caller, lex)
            else:
                walk(child, vars_, caller, lex)

    def _ts_call_target(fn):
        if fn is None:
            return None, None
        if fn.type == "identifier":
            return None, fn.text.decode()
        if fn.type == "member_expression":
            obj = fn.child_by_field_name("object")
            prop = fn.child_by_field_name("property")
            if prop is None:
                return None, None
            recv = obj.text.decode() if obj is not None else None
            # 链式/嵌套接收器(如 "user.getName()")保留原文,变量表查不到
            # → 类型未知 → resolver 走兜底链,不硬猜
            return recv, prop.text.decode()
        return None, None  # 高阶调用 i()() 等跳过

    def _ts_argc(node) -> int | None:
        args = node.child_by_field_name("arguments")
        if args is None:
            return None
        return sum(1 for a in args.named_children)

    def _ts_pattern_name(n) -> str:
        return n.text.decode()

    walk(tree.root_node, {}, None, set())
    return sites


def _calls_python(tree, source: bytes, fs: FileSymbols) -> list[CallSite]:
    sites: list[CallSite] = []
    caller_at = _caller_index(fs)

    def walk(node, vars_: dict, caller: int | None, in_class: bool):
        for child in node.named_children:
            t = child.type
            if t == "class_definition":
                idx_before = caller
                walk(child, vars_, caller, True)
            elif t == "function_definition":
                sub = dict(vars_)
                params = child.child_by_field_name("parameters")
                if params is not None:
                    for p in params.named_children:
                        if p.type in ("typed_parameter", "typed_default_parameter"):
                            n = p.child_by_field_name("name")
                            ty = p.child_by_field_name("type")
                            if n is not None and ty is not None:
                                sub[n.text.decode()] = _norm_type(ty.text.decode())
                idx = caller_at.get(child.start_point[0] + 1)
                walk(child, sub, idx if idx is not None else caller, in_class)
            elif t == "call":
                fn = child.child_by_field_name("function")
                recv, name = None, None
                if fn is not None:
                    if fn.type == "identifier":
                        name = fn.text.decode()
                    elif fn.type == "attribute":
                        obj = fn.child_by_field_name("object")
                        attr = fn.child_by_field_name("attribute")
                        if attr is not None:
                            name = attr.text.decode()
                            recv = obj.text.decode() if obj is not None else None
                if name:
                    sites.append(CallSite(
                        caller_local=caller,
                        name=name,
                        receiver=recv,
                        receiver_type=(
                            vars_.get(recv) if recv and recv != "self" else None
                        ),
                        arg_count=_py_argc(child),
                        kind="call",
                        line=child.start_point[0] + 1,
                        col=child.start_point[1] + 1,
                    ))
                walk(child, vars_, caller, in_class)
            else:
                walk(child, vars_, caller, in_class)

    def _py_argc(node) -> int | None:
        args = node.child_by_field_name("arguments")
        if args is None:
            return None
        return sum(1 for a in args.named_children
                   if a.type not in (",",))

    walk(tree.root_node, {}, None, False)
    return sites


def _calls_generic(tree, source: bytes, fs: FileSymbols) -> list[CallSite]:
    """go 等其余语言:identifier/selector 调用,不做类型推断。"""
    sites: list[CallSite] = []
    caller_at = _caller_index(fs)

    def walk(node, caller: int | None):
        for child in node.named_children:
            t = child.type
            if t in ("function_declaration", "method_declaration",
                     "function_definition", "method_definition"):
                idx = caller_at.get(child.start_point[0] + 1)
                walk(child, idx if idx is not None else caller)
            elif t in ("call_expression", "call"):
                fn = child.child_by_field_name("function")
                recv, name = None, None
                if fn is not None:
                    if fn.type in ("identifier",):
                        name = fn.text.decode()
                    elif fn.type in ("selector_expression", "member_expression", "attribute"):
                        obj = (fn.child_by_field_name("object")
                               or fn.child_by_field_name("operand"))
                        prop = (fn.child_by_field_name("field")
                                or fn.child_by_field_name("property")
                                or fn.child_by_field_name("attribute"))
                        if prop is not None:
                            name = prop.text.decode()
                            recv = obj.text.decode() if obj is not None else None
                if name:
                    sites.append(CallSite(
                        caller_local=caller, name=name, receiver=recv,
                        arg_count=None, kind="call",
                        line=child.start_point[0] + 1,
                        col=child.start_point[1] + 1,
                    ))
                walk(child, caller)
            else:
                walk(child, caller)

    walk(tree.root_node, None)
    return sites


def extract_call_sites(rel: str, lang: str, source: bytes, fs: FileSymbols) -> list[CallSite]:
    """提取单文件调用点;解析失败返回空列表。"""
    if lang not in SYMBOL_LANGS:
        return []
    try:
        tree = get_parser(lang).parse(source)
    except Exception:
        return []
    if lang == "java":
        return _calls_java(tree, source, fs)
    if lang in ("javascript", "typescript", "tsx"):
        return _calls_ts(tree, source, fs)
    if lang == "python":
        return _calls_python(tree, source, fs)
    return _calls_generic(tree, source, fs)
