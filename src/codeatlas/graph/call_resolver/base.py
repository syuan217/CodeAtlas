"""级联解析框架(code-graph-rag 思路,PLAN §2/§9.5)。

原则:
- AST 精确优先、同名匹配只做兜底;每条边带 resolution(exact/heuristic);
- 歧义意味着没有边,而不是猜测——多候选一律丢弃;
- 护栏:外部导入的名字绝不被同名兜底重新绑定;
  接收器类型已知(库内类)但没有该方法 = "已知非边",直接丢弃。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from codeatlas.graph.schema import (
    RES_EXACT,
    RES_HEURISTIC,
    SYM_CLASS,
    SYM_INTERFACE,
    SYM_MODULE,
    split_method_qname,
)


@dataclass
class CallSite:
    """单个调用点(Pass1 产物;caller_local 是 FileSymbols.symbols 的局部索引)。"""

    caller_local: int | None
    name: str
    receiver: str | None = None       # 接收器文本("obj" / "this" / 类名 / None=裸调用)
    receiver_type: str | None = None  # 提取时保守推断的接收器声明类型名(可为 None)
    arg_count: int | None = None      # 实参数量(重载消歧用)
    kind: str = "call"                # call / new / super
    line: int = 0
    col: int = 0


@dataclass
class ImportBinding:
    """名字级导入绑定(Pass1 产物,调用边解析的 import 探测输入)。"""

    name: str      # 绑定名:java 类短名 / ts·py 命名导入名 / as 别名
    source: str    # 来源:java FQCN / ts 说明符 / py 点路径(可带相对点)
    kind: str      # "class"(java import 类) / "named"(命名导入) / "module"(整模块 as)


@dataclass
class SymInfo:
    """库内符号的解析视图(Pass2 注册表条目)。"""

    id: int
    kind: str
    name: str
    qualified_name: str
    file_id: int
    file_rel: str
    signature: str | None = None


@dataclass
class ResolvedCall:
    caller_sym_id: int
    dst_sym_id: int
    resolution: str  # exact / heuristic
    line: int
    col: int


@dataclass
class ResolveContext:
    """全库符号注册表 + 文件级 import 绑定(Pass2 构建一次)。"""

    repo_id: int
    # 函数/方法名 → 符号(重名/重载为多元素)
    methods_by_name: dict[str, list[SymInfo]] = field(default_factory=dict)
    # 类/接口:qualified_name → 符号;类短名 → 符号列表(java 同短名多包)
    classes_by_qname: dict[str, SymInfo] = field(default_factory=dict)
    classes_by_name: dict[str, list[SymInfo]] = field(default_factory=dict)
    # 类 qname → 类内方法列表
    methods_of_class: dict[str, list[SymInfo]] = field(default_factory=dict)
    # 文件 rel → import 绑定
    bindings_by_file: dict[str, list[ImportBinding]] = field(default_factory=dict)
    # 文件 rel → 该文件全部符号(同文件兜底)
    symbols_of_file: dict[str, list[SymInfo]] = field(default_factory=dict)
    # module 符号:rel → SymInfo(ts/python 相对导入解析用)与 id 索引
    modules_by_rel: dict[str, SymInfo] = field(default_factory=dict)
    by_id: dict[int, SymInfo] = field(default_factory=dict)
    # 供 imports_resolver._resolve_* 消费的 {rel: module_id} 与反向映射
    modules_ids: dict[str, int] = field(default_factory=dict)
    rel_by_module_id: dict[int, str] = field(default_factory=dict)


def build_context(conn: sqlite3.Connection, repo_id: int) -> ResolveContext:
    ctx = ResolveContext(repo_id=repo_id)
    rows = conn.execute(
        "SELECT s.id, s.kind, s.name, s.qualified_name, s.file_id, s.signature, "
        "       f.path AS file_rel "
        "FROM symbols s JOIN files f ON s.file_id = f.id "
        "WHERE s.repo_id = ?",
        (repo_id,),
    ).fetchall()
    for r in rows:
        info = SymInfo(
            id=r["id"], kind=r["kind"], name=r["name"],
            qualified_name=r["qualified_name"], file_id=r["file_id"],
            file_rel=r["file_rel"], signature=r["signature"],
        )
        ctx.by_id[info.id] = info
        ctx.symbols_of_file.setdefault(info.file_rel, []).append(info)
        if info.kind == SYM_MODULE:
            ctx.modules_by_rel[info.file_rel] = info
            ctx.modules_ids[info.file_rel] = info.id
            ctx.rel_by_module_id[info.id] = info.file_rel
            continue
        if info.kind in (SYM_CLASS, SYM_INTERFACE):
            ctx.classes_by_qname[info.qualified_name] = info
            ctx.classes_by_name.setdefault(info.name, []).append(info)
        else:
            ctx.methods_by_name.setdefault(info.name, []).append(info)
            owner, _ = split_method_qname(info.qualified_name)
            if owner:
                ctx.methods_of_class.setdefault(owner, []).append(info)
    return ctx


def pick_unique(cands: list[SymInfo], arg_count: int | None) -> SymInfo | None:
    """唯一候选 → 命中;重载多候选按实参数量匹配唯一 → 命中;否则 None(歧义丢弃)。

    注意:参数数量比较只在 arg_count 可得时使用,且签名参数个数不可靠时
    (可变参数/泛型)放弃比较、直接判歧义——保守优先。
    """
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    if arg_count is None:
        return None
    by_argc = [c for c in cands if _signature_argc(c.signature) == arg_count]
    if len(by_argc) == 1:
        return by_argc[0]
    return None


def _signature_argc(signature: str | None) -> int | None:
    """从签名行估算形参数量;不可靠时返回 None(放弃比较)。"""
    if not signature:
        return None
    start = signature.find("(")
    end = signature.rfind(")")
    if start < 0 or end <= start:
        return None
    inner = signature[start + 1 : end].strip()
    if not inner:
        return 0
    # 拆顶层逗号(粗略:不处理嵌套泛型/注解里的逗号则低估;出错方向是判歧义,安全)
    depth = 0
    argc = 1
    for ch in inner:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        elif ch == "," and depth == 0:
            argc += 1
    if depth != 0:
        return None
    return argc


class CascadeResult:
    """一次级联解析的结果与标签。"""

    def __init__(self, dst: SymInfo | None, resolution: str | None, reason: str = ""):
        self.dst = dst
        self.resolution = resolution
        self.reason = reason

    @property
    def resolved(self) -> bool:
        return self.dst is not None

    @classmethod
    def exact(cls, dst, reason="") -> CascadeResult:
        return cls(dst, RES_EXACT, reason)

    @classmethod
    def heuristic(cls, dst, reason="") -> CascadeResult:
        return cls(dst, RES_HEURISTIC, reason)

    @classmethod
    def miss(cls, reason="") -> CascadeResult:
        return cls(None, None, reason)


def unique_name_fallback(
    ctx: ResolveContext, name: str, arg_count: int | None
) -> CascadeResult:
    """同名兜底(级联最后一步):库内唯一 → heuristic;多候选/无 → miss。

    调用方必须先确认护栏(接收器类型未知)再进入本步。
    """
    cands = ctx.methods_by_name.get(name, [])
    picked = pick_unique(cands, arg_count)
    if picked is not None:
        return CascadeResult.heuristic(picked, "unique-name")
    return CascadeResult.miss("ambiguous-or-missing")
