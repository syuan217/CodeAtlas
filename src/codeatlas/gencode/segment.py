"""模块划分(PLAN §9.7):目录聚类起步。

Java 顶层包 / JS 顶层目录 → 模块;过大(文件数超上限)按二级目录拆分,
过小(<MIN_FILES)并入父模块。模块 ID = 目录相对路径;模块哈希 = 成员文件
内容哈希的有序聚合(freshness 的 stale 判据)。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import PurePosixPath

MAX_FILES_PER_MODULE = 60
MIN_FILES_PER_MODULE = 5


@dataclass
class Module:
    id: str
    files: list[str] = field(default_factory=list)
    source_hash: str = ""


def _agg_hash(hashes: list[str]) -> str:
    from codeatlas.config import hash_bytes

    return hash_bytes("\n".join(sorted(hashes)).encode())


def _load_code_files(conn: sqlite3.Connection, repo_id: int) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT f.path, f.hash FROM files f "
            "WHERE f.repo_id=? AND f.parse_status='ok' AND ("
            "  f.path LIKE '%%.java' OR f.path LIKE '%%.ts' OR f.path LIKE '%%.tsx' "
            "  OR f.path LIKE '%%.xml' OR f.path LIKE '%%.js')",
            (repo_id,),
        )
    ]


def _cluster(files: list[dict], depth: int) -> dict[str, list[dict]]:
    """按前 depth 段目录路径聚类。"""
    groups: dict[str, list[dict]] = {}
    for f in files:
        parts = PurePosixPath(f["path"]).parts
        key = "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts)
        groups.setdefault(key or "(root)", []).append(f)
    return groups


def segment(conn: sqlite3.Connection, repo_id: int) -> list[Module]:
    files = _load_code_files(conn, repo_id)
    if not files:
        return []

    def build(depth: int) -> list[Module]:
        groups = _cluster(files, depth)
        modules: list[Module] = []
        small: list[Module] = []
        for key in sorted(groups):
            members = groups[key]
            if len(members) > MAX_FILES_PER_MODULE:
                # 过大:递归加深一层拆分
                sub = _cluster(members, depth + 1)
                for sk in sorted(sub):
                    sub_members = sub[sk]
                    m = Module(id=sk, files=[f["path"] for f in sub_members],
                               source_hash=_agg_hash([f["hash"] or "" for f in sub_members]))
                    if len(sub_members) < MIN_FILES_PER_MODULE:
                        small.append(m)
                    else:
                        modules.append(m)
            elif len(members) < MIN_FILES_PER_MODULE:
                small.append(Module(id=key, files=[f["path"] for f in members],
                                    source_hash=_agg_hash([f["hash"] or "" for f in members])))
            else:
                modules.append(Module(id=key, files=[f["path"] for f in members],
                                      source_hash=_agg_hash([f["hash"] or "" for f in members])))
        # 过小模块并入最近的父级模块(键前缀最长的)
        for m in small:
            parent = None
            best = -1
            for cand in modules:
                if cand.id == m.id:
                    continue
                common = 0
                a, b = m.id.split("/"), cand.id.split("/")
                for x, y in zip(a, b):
                    if x == y:
                        common += 1
                    else:
                        break
                if common > best and common >= 1:
                    best, parent = common, cand
            if parent is not None:
                parent.files += m.files
                parent.source_hash = _agg_hash(
                    [next(f["hash"] or "" for f in files if f["path"] == p)
                     for p in sorted(parent.files)]
                )
            elif m.files:
                modules.append(m)  # 没有父可并,保留
        return sorted(modules, key=lambda m: m.id)

    return build(1)
