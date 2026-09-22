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
    imports: list[str] = field(default_factory=list)  # 原始 import 目标(Pass1 暂存)

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
    ) -> int:
        sr, _, er, _ = _range(node)
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


def _extract_java(tree, lines: list[str]) -> tuple[_Collector, str | None, list[str]]:
    col = _Collector(lines)
    package: str | None = None
    imports: list[str] = []

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
                for c in child.children:
                    if c.type in ("scoped_identifier", "identifier"):
                        imports.append(c.text.decode("utf-8"))
                        break
            elif t in _JAVA_TYPES:
                name = child.child_by_field_name("name").text.decode("utf-8")
                qname = f"{prefix}{name}"
                idx = col.add(child, _JAVA_TYPES[t], name, qname, parent, with_comment=True)
                walk(child, f"{qname}.", idx)
            elif t == "method_declaration" or t == "constructor_declaration":
                name = child.child_by_field_name("name").text.decode("utf-8")
                kind = "method"
                qname = f"{prefix.rstrip('.') or 'root'}#{name}"
                col.add(child, kind, name, qname, parent)
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
    return col, package, imports


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

_PY_FROM_RE = re.compile(r"^from\s+(\.+)?([\w.]*)\s+import\b")
_PY_IMPORT_RE = re.compile(r"^import\s+([\w.,\s]+)")
# block 是 class body(要穿透)也是函数体(函数分支不递归,天然剪枝)
_PY_TRANSPARENT = {"block"}


def _extract_python(tree, lines: list[str], rel: str) -> tuple[_Collector, list[str]]:
    col = _Collector(lines)
    mod_qname = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
    imports: list[str] = []

    def walk(node, prefix: str, parent: int | None, in_class: bool) -> None:
        for child in node.named_children:
            t = child.type
            if t == "import_from_statement":
                m = _PY_FROM_RE.match(child.text.decode("utf-8"))
                if m:
                    level, mod = m.group(1) or "", m.group(2)
                    imports.append(f"{level}{mod}")
            elif t == "import_statement":
                m = _PY_IMPORT_RE.match(child.text.decode("utf-8"))
                if m:
                    for part in m.group(1).split(","):
                        part = part.split(" as ")[0].strip()
                        if part:
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
    return col, imports


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


def _extract_ts(tree, lines: list[str], rel: str) -> tuple[_Collector, list[str]]:
    col = _Collector(lines)
    mod_qname = str(Path(rel).with_suffix(""))
    imports: list[str] = []

    def import_source(node) -> None:
        src_node = node.child_by_field_name("source")
        if src_node is not None:
            imports.append(src_node.text.decode("utf-8").strip("'\""))

    def walk(node, prefix: str, parent: int | None, in_class: bool) -> None:
        for child in node.named_children:
            t = child.type
            if t == "import_statement":
                import_source(child)
            elif t == "export_statement":
                import_source(child)  # export ... from "./x"
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
    return col, imports


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

_GO_RECV_RE = re.compile(r"[\w.\s()*]+\)\s*$")


def _extract_go(tree, lines: list[str], rel: str) -> tuple[_Collector, list[str]]:
    col = _Collector(lines)
    mod_qname = str(Path(rel).with_suffix(""))
    imports: list[str] = []

    def walk(node, prefix: str, parent: int | None) -> None:
        for child in node.named_children:
            t = child.type
            if t == "import_declaration":
                for spec in child.named_children:
                    if spec.type == "import_spec":
                        path = spec.child_by_field_name("path")
                        if path is not None:
                            imports.append(path.text.decode("utf-8").strip("`\"'"))
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
    return col, imports


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
        col, package, imports = _extract_java(tree, lines)
        fs.package = package
        fs.imports = imports
    elif lang == "python":
        col, imports = _extract_python(tree, lines, rel)
        fs.imports = imports
    elif lang in ("javascript", "typescript", "tsx"):
        col, imports = _extract_ts(tree, lines, rel)
        fs.imports = imports
    elif lang == "go":
        col, imports = _extract_go(tree, lines, rel)
        fs.imports = imports
    else:  # pragma: no cover
        return fs
    fs.symbols = col.symbols
    return fs
