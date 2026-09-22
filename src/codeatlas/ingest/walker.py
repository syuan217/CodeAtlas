"""目录遍历:尊重 .gitignore(嵌套)+ repos.yaml exclude。

跳过规则(PLAN §9.1):二进制(首 1KB 控制字符 >10%,AnythingLLM 哨探法)
与 >1MB 单文件由调用方(indexer)读取时标记 parse_status=skipped;
walker 只负责"哪些文件可见"。输出按路径排序,保证遍历确定可重复。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pathspec

from codeatlas.config import RepoCfg

MAX_FILE_BYTES = 1 * 1024 * 1024
BINARY_SNIFF_BYTES = 1024
BINARY_CTRL_RATIO = 0.10

# 控制字符 = NUL 与 <0x20 中除 \t \n \r 外;>=0x80(UTF-8 多字节)不算
_CTRL = {b for b in range(0x20)} - {0x09, 0x0A, 0x0D} | {0x00, 0x7F}


def sniff_binary(head: bytes) -> bool:
    """首块哨探:NUL/控制字符占比 >10% 判二进制。"""
    if not head:
        return False
    ctrl = sum(1 for b in head if b in _CTRL)
    return ctrl / len(head) > BINARY_CTRL_RATIO


@dataclass
class FileEntry:
    abs_path: Path
    rel: str  # 相对仓库根,posix 风格
    size: int
    mtime: float


def _load_dir_spec(d: Path) -> pathspec.GitIgnoreSpec | None:
    gi = d / ".gitignore"
    if not gi.is_file():
        return None
    try:
        lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _ignored(
    stack: list[tuple[str, pathspec.GitIgnoreSpec]],
    rel: str,
    *,
    is_dir: bool = False,
) -> bool:
    """rel 相对每层 .gitignore 所在目录匹配一次(嵌套 gitignore 语义)。

    目录额外用 "dir/" 形式再试:GitIgnoreSpec 对 "gen/" 模式不匹配裸目录名。
    """
    for base, spec in stack:
        if base:
            if not rel.startswith(base + "/"):
                continue
            sub = rel[len(base) + 1:]
        else:
            sub = rel
        if spec.match_file(sub):
            return True
        if is_dir and spec.match_file(sub + "/"):
            return True
    return False


def walk(repo: RepoCfg) -> list[FileEntry]:
    """遍历仓库可见文件;.gitignore(每目录)与 repo.exclude 之外的都返回。"""
    root = repo.path
    if not root.is_dir():
        raise FileNotFoundError(f"仓库路径不存在或不是目录:{root}")
    exclude = pathspec.GitIgnoreSpec.from_lines(repo.exclude or [])
    out: list[FileEntry] = []

    def rec(d: Path, rel_dir: str, stack: list[tuple[str, pathspec.GitIgnoreSpec]]) -> None:
        spec = _load_dir_spec(d)
        if spec is not None:
            stack = stack + [(rel_dir, spec)]
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except (PermissionError, OSError):
            return
        for e in entries:
            if e.name in (".git", ".gitignore"):
                continue
            rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
            if e.is_dir(follow_symlinks=False):
                if _ignored(stack, rel, is_dir=True) or exclude.match_file(rel + "/"):
                    continue
                rec(Path(e.path), rel, stack)
            elif e.is_file(follow_symlinks=False):
                if _ignored(stack, rel) or exclude.match_file(rel):
                    continue
                try:
                    st = e.stat()
                except OSError:
                    continue
                out.append(FileEntry(Path(e.path), rel, st.st_size, st.st_mtime))

    rec(root, "", [])
    return out
