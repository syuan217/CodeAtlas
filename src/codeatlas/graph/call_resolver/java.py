"""Java 调用级联(PLAN §2 code-graph-rag 级联的 Java 版):

裸/this/super 调用(类内 → 同文件 → 唯一同名兜底)
→ import 类绑定 / 同包唯一类(static 调用)
→ 接收器声明类型(库内唯一类 → 类内方法;库外/歧义 = 已知非边,丢弃)
→ 库内唯一同名兜底(heuristic;仅当接收器类型未知)。

歧义即丢弃;外部类型绝不走同名兜底(护栏)。
"""

from __future__ import annotations

from codeatlas.graph.call_resolver.base import (
    CascadeResult,
    ResolveContext,
    SymInfo,
    pick_unique,
    unique_name_fallback,
)
from codeatlas.graph.schema import split_method_qname


def _class_methods_exact(
    ctx: ResolveContext, class_qname: str, name: str, arg_count: int | None
) -> CascadeResult:
    cands = [m for m in ctx.methods_of_class.get(class_qname, []) if m.name == name]
    picked = pick_unique(cands, arg_count)
    if picked is not None:
        return CascadeResult.exact(picked, "class-method")
    return CascadeResult.miss("no-or-ambiguous-in-class")


def _same_file_exact(
    ctx: ResolveContext, file_rel: str, name: str, arg_count: int | None
) -> CascadeResult:
    cands = [
        s
        for s in ctx.symbols_of_file.get(file_rel, [])
        if s.name == name and s.kind in ("function", "method")
    ]
    picked = pick_unique(cands, arg_count)
    if picked is not None:
        return CascadeResult.exact(picked, "same-file")
    return CascadeResult.miss("no-or-ambiguous-in-file")


def resolve_site(site, caller: SymInfo | None, fs, ctx: ResolveContext) -> CascadeResult:
    """M3 验收收紧后的级联(宁缺毋错):

    - 裸/this/super 调用只认 类内 与 同文件,不再全库同名兜底
      (跨类调用在 Java 必须类名限定,裸调用兜底必错);
    - 带接收器且类型未知 → 丢弃(实例 getter/Lombok 生成方法/链式中间结果
      均无类型可依,曾误绑同名 static 工具方法);
    - new 只认库内类。
    heuristic 兜底仅保留在 generic(python/go)。
    """
    name, argc = site.name, site.arg_count
    file_rel = fs.rel
    caller_class = None
    if caller is not None and caller.kind == "method":
        caller_class = split_method_qname(caller.qualified_name)[0]

    # ---- P1 裸调用 / this / super ----
    if site.receiver in (None, "this", "super"):
        if caller_class:
            r = _class_methods_exact(ctx, caller_class, name, argc)
            if r.resolved:
                return r
        return _same_file_exact(ctx, file_rel, name, argc)

    # ---- P2 接收器是类名(static 调用 / 工厂)----
    binding = next(
        (b for b in fs.import_bindings if b.name == site.receiver), None
    )
    if binding is not None and binding.kind == "class":
        cls = ctx.classes_by_qname.get(binding.source)
        if cls is not None:
            r = _class_methods_exact(ctx, cls.qualified_name, name, argc)
            if r.resolved:
                return r
            if any(m.name == name for m in ctx.methods_of_class.get(cls.qualified_name, [])):
                return r  # 类内有同名但歧义 → 丢弃
            return CascadeResult.miss("imported-class-no-method")
        return CascadeResult.miss("external-class")  # 外部类,不兜底
    # static import 成员:a.b.C.m 绑定 m → 直接库内查 C#m
    if binding is not None and binding.kind == "named":
        cls = ctx.classes_by_qname.get(binding.source)
        if cls is not None:
            return _class_methods_exact(ctx, cls.qualified_name, name, argc)
        return CascadeResult.miss("external-static")

    # 未 import 但接收器像类名(同包类):库内短名唯一才尝试
    cls_cands = ctx.classes_by_name.get(site.receiver, [])
    if len(cls_cands) == 1:
        r = _class_methods_exact(ctx, cls_cands[0].qualified_name, name, argc)
        if r.resolved:
            return r
        return CascadeResult.miss("unique-class-no-method")
    if len(cls_cands) > 1:
        return CascadeResult.miss("ambiguous-class-name")

    # ---- P3 接收器是实例:类型已知才解析;未知 = 丢弃(护栏)----
    if site.receiver_type:
        typed = ctx.classes_by_name.get(site.receiver_type, [])
        if len(typed) == 1:
            r = _class_methods_exact(ctx, typed[0].qualified_name, name, argc)
            if r.resolved:
                return r
            return CascadeResult.miss("typed-receiver-no-method")  # 已知非边
        return CascadeResult.miss("type-unknown-or-external")
    return CascadeResult.miss("untyped-receiver")


def resolve_new(site, caller: SymInfo | None, fs, ctx: ResolveContext) -> CascadeResult:
    """new Foo(...):库内类 → 构造器符号(无则类符号);库外 → miss。"""
    name = site.name
    binding = next((b for b in fs.import_bindings if b.name == name), None)
    cls: SymInfo | None = None
    if binding is not None:
        cls = ctx.classes_by_qname.get(binding.source)
        if cls is None:
            return CascadeResult.miss("external-class")
    else:
        cands = ctx.classes_by_name.get(name, [])
        if len(cands) == 1:
            cls = cands[0]
        elif len(cands) > 1:
            return CascadeResult.miss("ambiguous-class")
        else:
            return CascadeResult.miss("class-not-in-repo")
    ctors = [
        m
        for m in ctx.methods_of_class.get(cls.qualified_name, [])
        if m.name == name
    ]
    picked = pick_unique(ctors, site.arg_count)
    if picked is not None:
        return CascadeResult.exact(picked, "constructor")
    if cls.qualified_name in ctx.methods_of_class and not ctors:
        return CascadeResult.exact(cls, "default-constructor→class")
    if ctors:
        return CascadeResult.miss("ambiguous-ctor")
    return CascadeResult.exact(cls, "class-symbol")
