"""git 基线变更集(PLAN §9.4)。

主路径:`git diff --name-status <基线>..HEAD`;
兜底回退 MD5(mtime)+blake2b 全量扫描,三种情况触发:
目录非 git 仓库 / 工作区非干净(未提交改动 diff 看不到)/ 基线 commit 不存在(rebase/浅克隆)。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from codeatlas.config import RepoCfg, hash_bytes
from codeatlas.ingest.walker import walk


@dataclass
class ChangeSet:
    repo: RepoCfg
    mode: str  # "git" | "scan"
    base_commit: str | None
    head_commit: str | None
    changes: dict[str, str] = field(default_factory=dict)  # rel -> A/M/D

    @property
    def is_noop(self) -> bool:
        return not self.changes


def _git(repo_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_path), "-c", "core.quotepath=false", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _head(repo_path: Path) -> str | None:
    r = _git(repo_path, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def _is_clean(repo_path: Path) -> bool:
    return _git(repo_path, "status", "--porcelain").stdout.strip() == ""


def _commit_exists(repo_path: Path, rev: str) -> bool:
    return (
        _git(repo_path, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").returncode
        == 0
    )


def _parse_name_status(out: str) -> dict[str, str]:
    """`diff --name-status` 输出 → {path: A/M/D};重命名拆成 旧D+新A。"""
    changes: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]  # R100/C75 → R/C
        if status == "R":
            changes[parts[1]] = "D"
            changes[parts[2]] = "A"
        elif status == "C":
            changes[parts[2]] = "A"  # copy:旧文件仍在
        else:
            changes[parts[1]] = {"A": "A", "D": "D"}.get(status, "M")
    return changes


def _scan(
    repo: RepoCfg,
    known_files: dict[str, tuple[str, float]],
    force: bool = False,
) -> ChangeSet:
    """兜底全量扫描:mtime 快速跳过,变了再算 blake2b 判 A/M/D。

    force=True(--full)不做任何跳过,全部文件强制进变更集(修复性重索引)。
    """
    changes: dict[str, str] = {}
    seen: set[str] = set()
    for entry in walk(repo):
        seen.add(entry.rel)
        known = known_files.get(entry.rel)
        if known is None:
            changes[entry.rel] = "A"
        elif force:
            changes[entry.rel] = "M"
        elif known[1] != entry.mtime:
            if hash_bytes(entry.abs_path.read_bytes()) != known[0]:
                changes[entry.rel] = "M"
    for rel in known_files:
        if rel not in seen:
            changes[rel] = "D"
    return ChangeSet(repo, "scan", None, None, changes)


def compute_change_set(
    repo: RepoCfg,
    indexed_commit: str | None,
    known_files: dict[str, tuple[str, float]],
    force_scan: bool = False,
) -> ChangeSet:
    """known_files = 库中该 repo 现有 {rel: (hash, mtime)},scan 兜底比对用。

    force_scan=True 跳过 git 快路径,直接全量 hash/mtime 扫描(--full)。
    """
    path = repo.path
    if not force_scan and (path / ".git").exists():
        head = _head(path)
        if head and _is_clean(path):
            if indexed_commit is None:
                files = _git(path, "ls-files").stdout.splitlines()
                return ChangeSet(
                    repo, "git", None, head, {f: "A" for f in files if f.strip()}
                )
            if _commit_exists(path, indexed_commit):
                r = _git(path, "diff", "--name-status", f"{indexed_commit}..HEAD")
                return ChangeSet(
                    repo, "git", indexed_commit, head, _parse_name_status(r.stdout)
                )
    return _scan(repo, known_files, force=force_scan)
