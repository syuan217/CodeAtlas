"""IMPORTS 边两阶段写入(PLAN §9.2)。

Pass1(symbols.py,indexer 内收集):逐文件解析 import 语句暂存内存;
Pass2(本模块,全部文件入库后调用):对照库中符号注册表验证目标——
内部目标必须验证命中,歧义即丢弃,未命中(外部依赖)首版不持久化
(§7 schema 无外部节点/字段,不建 ExternalModule)。

注册表直接查 symbols 表(module 符号 qname=相对路径;java 类符号 qname=FQCN),
天然支持增量:改动文件的旧 IMPORTS 边随符号子树级联删除,重跑只写新边。
"""

from __future__ import annotations

import posixpath
import sqlite3
from dataclasses import dataclass, field


@dataclass
class ImportRef:
    """Pass1 暂存的单文件 import 信息。"""

    module_symbol_id: int
    rel: str
    lang: str
    imports: list[str] = field(default_factory=list)


# ts/js 相对导入的候选后缀(按常见优先级)
_TS_SUFFIXES = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]
_TS_INDEX = [f"/index{s}" for s in (".ts", ".tsx", ".js", ".jsx")]


def _resolve_java(import_target: str, java_classes: dict[str, int]) -> int | None:
    """java FQCN → 目标文件 module symbol id;static 成员先试整体再去尾。"""
    target = import_target.removeprefix("static ").strip()
    for candidate in (target, target.rsplit(".", 1)[0]):
        mid = java_classes.get(candidate)
        if mid is not None:
            return mid
    return None


def _resolve_python(rel: str, target: str, modules: dict[str, int]) -> int | None:
    """"pkg.mod" / ".mod" / "..pkg.mod" → module symbol id。"""
    if target.startswith("."):
        level = len(target) - len(target.lstrip("."))
        mod = target.lstrip(".")
        parts = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
        base = parts[:-level] if level <= len(parts) else []
        if not base:
            return None  # 相对层级逃出仓库根
        full_parts = base + ([mod] if mod else [])
    else:
        full_parts = target.split(".")
    joined = "/".join(full_parts)
    for cand in (f"{joined}.py", f"{joined}/__init__.py"):
        mid = modules.get(cand)
        if mid is not None:
            return mid
    return None


def _resolve_ts(rel: str, spec: str, modules: dict[str, int]) -> int | None:
    """相对说明符 ./x ../x → module symbol id;裸包名/别名首版丢弃。"""
    if not spec.startswith("."):
        return None
    src_dir = posixpath.dirname(rel)
    joined = posixpath.normpath(posixpath.join(src_dir, spec)) if src_dir else posixpath.normpath(spec)
    if joined.startswith(".."):
        return None
    for suffix in _TS_SUFFIXES:
        mid = modules.get(joined + suffix)
        if mid is not None:
            return mid
    for index in _TS_INDEX:
        mid = modules.get(joined + index)
        if mid is not None:
            return mid
    return None


def write_import_edges(
    conn: sqlite3.Connection, repo_id: int, refs: list[ImportRef]
) -> int:
    """Pass2:验证并写入 IMPORTS 边,返回写入条数。

    边写入用 WHERE NOT EXISTS 幂等(IMPORTS 边 line/col 为 NULL,
    UNIQUE 约束对 NULL 不去重,故不能只靠 INSERT OR IGNORE)。
    """
    modules: dict[str, int] = {}
    java_classes: dict[str, int] = {}
    for row in conn.execute(
        "SELECT s.id, s.qualified_name, s.kind, "
        "       (SELECT m.id FROM symbols m WHERE m.file_id = s.file_id AND m.kind='module') AS mod_id "
        "FROM symbols s WHERE s.repo_id = ?",
        (repo_id,),
    ):
        if row["kind"] == "module":
            modules[row["qualified_name"]] = row["id"]
        elif row["kind"] in ("class", "interface") and row["mod_id"] is not None:
            java_classes.setdefault(row["qualified_name"], row["mod_id"])
            # 同 FQCN 出现在多个文件 = 歧义,标记为无效
            if java_classes[row["qualified_name"]] != row["mod_id"]:
                java_classes[row["qualified_name"]] = -1

    written = 0
    for ref in refs:
        targets: set[int] = set()
        for imp in ref.imports:
            dst: int | None
            if ref.lang == "java":
                dst = _resolve_java(imp, java_classes)
            elif ref.lang == "python":
                dst = _resolve_python(ref.rel, imp, modules)
            elif ref.lang in ("javascript", "typescript", "tsx"):
                dst = _resolve_ts(ref.rel, imp, modules)
            else:
                dst = None  # go 的模块路径解析依赖 go.mod,首版不建边
            if dst and dst > 0 and dst != ref.module_symbol_id:
                targets.add(dst)
        for dst in targets:
            cur = conn.execute(
                "INSERT INTO edges(repo_id, src_id, dst_id, kind) "
                "SELECT ?, ?, ?, 'IMPORTS' "
                "WHERE NOT EXISTS (SELECT 1 FROM edges "
                "  WHERE repo_id=? AND src_id=? AND dst_id=? AND kind='IMPORTS')",
                (repo_id, ref.module_symbol_id, dst, repo_id, ref.module_symbol_id, dst),
            )
            written += cur.rowcount
    conn.commit()
    return written
