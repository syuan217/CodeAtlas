"""JS/TS 调用级联(M3 验收收紧后,宁缺毋错):

- 裸调用:import 命名绑定(含 @/ 别名)→ 同文件;词法局部定义(局部函数/
  回调参数)→ 丢弃;不再全库同名兜底(曾误绑无关模块同名导出);
- this.method → 类内;
- 接收器类型已知(标注 / new 推断)→ 类#方法;类型未知(第三方库、链式
  中间结果、解构回调)→ 丢弃;
- new ClassName → 构造器/类符号。

别名:v1 内置 `@/ → src/`(tsconfig paths 标准约定);其余别名不解析。
"""

from __future__ import annotations

import posixpath

from codeatlas.graph.call_resolver.base import (
    CascadeResult,
    ResolveContext,
    SymInfo,
    pick_unique,
)
from codeatlas.graph.schema import split_method_qname
from codeatlas.ingest.imports_resolver import _TS_INDEX, _TS_SUFFIXES

# tsconfig paths 别名 → 仓库内前缀(v1 内置标准约定)
TS_PATH_ALIASES = {"@": "src"}


def _resolve_spec(file_rel: str, spec: str, ctx: ResolveContext) -> str | None:
    """说明符 → 目标文件 rel。支持 ./ ../ 相对与 @/ 别名;裸包名不解析。"""
    candidates: list[str] = []
    if spec.startswith("."):
        src_dir = posixpath.dirname(file_rel)
        joined = (
            posixpath.normpath(posixpath.join(src_dir, spec))
            if src_dir
            else posixpath.normpath(spec)
        )
        if not joined.startswith(".."):
            candidates.append(joined)
    else:
        for alias, prefix in TS_PATH_ALIASES.items():
            head = f"{alias}/"
            if spec.startswith(head):
                candidates.append(posixpath.normpath(prefix + "/" + spec[len(head):]))
    for joined in candidates:
        for suffix in _TS_SUFFIXES:
            if joined + suffix in ctx.modules_by_rel:
                return joined + suffix
        for index in _TS_INDEX:
            if joined + index in ctx.modules_by_rel:
                return joined + index
    return None


def _file_name_exact(
    ctx: ResolveContext, file_rel: str | None, name: str, arg_count: int | None
) -> CascadeResult:
    if file_rel is None:
        return CascadeResult.miss("no-target-file")
    cands = [
        s
        for s in ctx.symbols_of_file.get(file_rel, [])
        if s.name == name and s.kind in ("function", "method", "class", "interface")
    ]
    picked = pick_unique(cands, arg_count)
    if picked is not None:
        return CascadeResult.exact(picked, "imported-name")
    return CascadeResult.miss("no-or-ambiguous-in-target")


def resolve_site(site, caller: SymInfo | None, fs, ctx: ResolveContext) -> CascadeResult:
    name, argc = site.name, site.arg_count
    if site.kind == "new":
        return _resolve_new(site, caller, fs, ctx)

    caller_class = None
    if caller is not None and caller.kind == "method":
        caller_class = split_method_qname(caller.qualified_name)[0]

    # ---- P1 裸调用 ----
    if site.receiver is None:
        if site.local_def:
            # 词法局部:函数内定义的局部函数 / 回调参数同名(提取时标记)
            return CascadeResult.miss("lexical-local")
        binding = next(
            (b for b in fs.import_bindings if b.name == name and b.kind == "named"),
            None,
        )
        if binding is not None:
            target = _resolve_spec(fs.rel, binding.source, ctx)
            r = _file_name_exact(ctx, target, binding.target_name or name, argc)
            if r.resolved:
                return r
            return CascadeResult.miss("binding-target-miss")
        # 同文件
        cands = [
            s
            for s in ctx.symbols_of_file.get(fs.rel, [])
            if s.name == name and s.kind in ("function", "method", "class")
        ]
        picked = pick_unique(cands, argc)
        if picked is not None:
            return CascadeResult.exact(picked, "same-file")
        return CascadeResult.miss("bare-unresolved")  # 不做全库兜底

    # ---- P2 this.method ----
    if site.receiver in ("this", "self"):
        if caller_class:
            cands = [
                m
                for m in ctx.methods_of_class.get(caller_class, [])
                if m.name == name
            ]
            picked = pick_unique(cands, argc)
            if picked is not None:
                return CascadeResult.exact(picked, "this-method")
        return CascadeResult.miss("this-no-method")

    # 模块命名空间调用:Api.method()(@/ 别名修复后走 exact)
    mod_binding = next(
        (b for b in fs.import_bindings if b.name == site.receiver and b.kind == "module"),
        None,
    )
    if mod_binding is not None:
        target = _resolve_spec(fs.rel, mod_binding.source, ctx)
        return _file_name_exact(ctx, target, name, argc)

    # ---- P3 接收器类型已知才解析;未知 = 丢弃(第三方库/链式/解构回调)----
    if site.receiver_type:
        typed = ctx.classes_by_name.get(site.receiver_type, [])
        if len(typed) == 1:
            cands = [
                m
                for m in ctx.methods_of_class.get(typed[0].qualified_name, [])
                if m.name == name
            ]
            picked = pick_unique(cands, argc)
            if picked is not None:
                return CascadeResult.exact(picked, "typed-receiver")
            return CascadeResult.miss("typed-receiver-no-method")
        return CascadeResult.miss("type-unknown-or-external")
    return CascadeResult.miss("untyped-receiver")


def _resolve_new(site, caller: SymInfo | None, fs, ctx: ResolveContext) -> CascadeResult:
    name = site.name.split("<")[0]
    # 同文件类优先
    local = [
        s
        for s in ctx.symbols_of_file.get(fs.rel, [])
        if s.name == name and s.kind == "class"
    ]
    if len(local) == 1:
        cls = local[0]
    else:
        binding = next(
            (b for b in fs.import_bindings if b.name == name and b.kind == "named"), None
        )
        target = (
            _resolve_spec(fs.rel, binding.source, ctx) if binding is not None else None
        )
        pool = ctx.symbols_of_file.get(target, []) if target else []
        if not pool:
            return CascadeResult.miss("new-class-unknown")
        cands = [s for s in pool if s.name == name and s.kind == "class"]
        if len(cands) != 1:
            return CascadeResult.miss("new-class-ambiguous-or-missing")
        cls = cands[0]
    ctors = [
        m
        for m in ctx.methods_of_class.get(cls.qualified_name, [])
        if m.name == "constructor"
    ]
    if ctors:
        picked = pick_unique(ctors, site.arg_count)
        if picked is not None:
            return CascadeResult.exact(picked, "constructor")
        return CascadeResult.exact(cls, "ctor-overload→class")
    return CascadeResult.exact(cls, "class-symbol")
