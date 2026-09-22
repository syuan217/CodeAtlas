"""其余语言(go / python 走相同兜底)的同模块精确 + 唯一同名兜底。

python 的 self.method 也在此处理(其调用点由 _calls_python 提取)。
"""

from __future__ import annotations

from codeatlas.graph.call_resolver.base import (
    CascadeResult,
    ResolveContext,
    pick_unique,
    unique_name_fallback,
)
from codeatlas.graph.schema import split_method_qname


def resolve_site(site, caller, fs, ctx: ResolveContext) -> CascadeResult:
    name, argc = site.name, site.arg_count

    caller_class = None
    if caller is not None and caller.kind == "method":
        caller_class = split_method_qname(caller.qualified_name)[0]

    # self.method():当前类内
    if site.receiver in ("self", "this"):
        if caller_class:
            cands = [
                m
                for m in ctx.methods_of_class.get(caller_class, [])
                if m.name == name
            ]
            picked = pick_unique(cands, argc)
            if picked is not None:
                return CascadeResult.exact(picked, "self-method")
        return CascadeResult.miss("self-no-method")

    # import 命名绑定(python from X import name / go 模块)→ 目标模块文件
    binding = next(
        (b for b in fs.import_bindings if b.name == name and b.kind == "named"), None
    )
    if binding is not None and fs.lang == "python":
        from codeatlas.ingest.imports_resolver import _resolve_python

        mid = _resolve_python(fs.rel, binding.source, ctx.modules_ids)
        target = ctx.rel_by_module_id.get(mid) if mid else None
        if target is not None:
            cands = [
                s
                for s in ctx.symbols_of_file.get(target, [])
                if s.name == name and s.kind in ("function", "class", "method")
            ]
            picked = pick_unique(cands, argc)
            if picked is not None:
                return CascadeResult.exact(picked, "from-import")
        return CascadeResult.miss("from-import-target-miss")

    # 模块命名空间:pm.f()(python import as)→ 模块文件内 f
    mod_binding = next(
        (b for b in fs.import_bindings if b.name == site.receiver and b.kind == "module"),
        None,
    )
    if mod_binding is not None and fs.lang == "python":
        from codeatlas.ingest.imports_resolver import _resolve_python

        mid = _resolve_python(fs.rel, mod_binding.source, ctx.modules_ids)
        target = ctx.rel_by_module_id.get(mid) if mid else None
        if target is not None:
            cands = [
                s
                for s in ctx.symbols_of_file.get(target, [])
                if s.name == name and s.kind in ("function", "method", "class")
            ]
            picked = pick_unique(cands, argc)
            if picked is not None:
                return CascadeResult.exact(picked, "module-attr")
        return CascadeResult.miss("module-attr-miss")

    # 同文件唯一
    cands = [
        s
        for s in ctx.symbols_of_file.get(fs.rel, [])
        if s.name == name and s.kind in ("function", "method")
    ]
    picked = pick_unique(cands, argc)
    if picked is not None:
        return CascadeResult.exact(picked, "same-file")

    # 库内唯一兜底(heuristic)
    return unique_name_fallback(ctx, name, argc)
