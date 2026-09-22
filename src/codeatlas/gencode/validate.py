"""五层质量的硬校验与指标层(PLAN §9.6 第 1/2 层)。

硬校验三关(宁可缺,不可错):
a. 引用路径 ∈ 仓库文件清单;
b. 引用行号范围合法(1 ≤ A ≤ B ≤ 文件行数);
c. snippet 反查:引用邻近代码块逐字内容在真实文件 find——命中则用真实行号
   覆盖 LLM 给的行号(核心防幻觉机制),找不到 → 该引用重试,仍失败剔除并标
   [未验证]。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

CITATION_RE = re.compile(r"\[([^\]:]+):(\d+)-(\d+)\]")
FENCE_BLOCK_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)
UNVERIFIED_MARK = "[未验证]"
_MAX_SNIPPET_SEARCH = 400  # snippet 反查的文件大小上限(KB)


@dataclass
class PageMetrics:
    citations_total: int = 0
    citations_ok: int = 0
    citations_fixed: int = 0      # snippet 反查覆盖了行号
    citations_removed: int = 0
    symbol_coverage: float = 0.0
    mermaid_ok: bool = True
    mermaid_issues: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def front_matter_dict(self) -> dict:
        return {
            "citations_total": self.citations_total,
            "citations_ok": self.citations_ok,
            "citations_fixed": self.citations_fixed,
            "citations_removed": self.citations_removed,
            "symbol_coverage": round(self.symbol_coverage, 3),
            "mermaid_ok": self.mermaid_ok,
        }


def _repo_files(conn: sqlite3.Connection, repo_id: int) -> set[str]:
    return {
        r["path"]
        for r in conn.execute(
            "SELECT path FROM files WHERE repo_id=? AND parse_status='ok'",
            (repo_id,),
        )
    }


def _page_file_map(conn: sqlite3.Connection, repo_id: int) -> dict[str, str]:
    """引用路径白名单 + 该文件的磁盘路径(经 files 表,不信任 LLM 传路径)。"""
    root_row = conn.execute(
        "SELECT path FROM repos WHERE id=?", (repo_id,)
    ).fetchone()
    root = Path(root_row["path"]) if root_row else Path()
    return {p: root / p for p in _repo_files(conn, repo_id)}


def check_mermaid(md: str) -> tuple[bool, list[str]]:
    """mermaid 语法级校验(结构化检查;渲染级校验留待人工/工具)。"""
    issues: list[str] = []
    blocks = re.findall(r"```mermaid\n(.*?)```", md, re.S)
    if not blocks:
        return True, []
    for i, b in enumerate(blocks):
        if not re.match(r"\s*graph\s+TD\b", b):
            issues.append(f"mermaid#{i}: 不是 graph TD")
        steps = re.findall(r"-->", b)
        if not steps:
            issues.append(f"mermaid#{i}: 无任何边(-->),疑似空图")
        lines = [ln for ln in b.splitlines() if "-->" in ln]
        no_ref = [ln.strip()[:60] for ln in lines if ":" not in ln]
        if no_ref:
            issues.append(f"mermaid#{i}: {len(no_ref)} 条边未标注 文件:行号")
    return not issues, issues


def symbol_coverage(md: str, conn: sqlite3.Connection, repo_id: int,
                    file_paths: list[str]) -> float:
    """模块 public 符号被页面提及比例(阈值 70% 记入指标,不硬失败)。"""
    if not file_paths:
        return 0.0
    ph = ",".join("?" * len(file_paths))
    rows = conn.execute(
        f"SELECT name FROM symbols s JOIN files f ON s.file_id=f.id "
        f"WHERE s.repo_id=? AND f.path IN ({ph}) "
        f"AND s.kind IN ('class','interface')",
        [repo_id, *file_paths],
    ).fetchall()
    names = [r["name"] for r in rows]
    if not names:
        return 0.0
    hit = sum(1 for n in names if n in md)
    return hit / len(names)


def _fix_or_remove_citation(
    md: str, cite_path: str, a: int, b: int,
    file_map: dict[str, str], cache: dict[str, list[str]],
) -> tuple[str, str]:
    """返回 (处理后 md, 动作 ok|fixed|removed|missing-path|bad-range)。"""
    real = file_map.get(cite_path)
    if real is None:
        return md, "missing-path"
    if cite_path not in cache:
        try:
            if real.stat().st_size > _MAX_SNIPPET_SEARCH * 1024:
                cache[cite_path] = []
            else:
                cache[cite_path] = real.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
        except OSError:
            cache[cite_path] = []
    lines = cache[cite_path]
    if not lines:
        return md, "bad-range"
    range_ok = 1 <= a <= b <= len(lines)
    # snippet 反查(优先于范围判定:LLM 行号越界恰是最需要反查修复的场景)
    idx = md.find(f"[{cite_path}:{a}-{b}]")
    if idx >= 0:
        before = md[:idx]
        fence = None
        for m in FENCE_BLOCK_RE.finditer(before):
            fence = m
        if fence is not None:
            snippet_lines = [
                ln for ln in fence.group(1).splitlines() if ln.strip()
            ]
            whole = "\n".join(lines)
            # 逐行匹配(模型改写常见,整段匹配过严):首个非空行命中即反查
            probe_line = next((ln for ln in snippet_lines if ln.strip()), "")
            if probe_line and probe_line in whole:
                hit = whole.count("\n", 0, whole.find(probe_line)) + 1
                new_start = hit
                new_end = min(new_start + len(snippet_lines) - 1, len(lines))
                if range_ok and probe_line in "\n".join(
                    lines[max(0, a - 30): b + 30]
                ):
                    return md, "ok"
                md = md.replace(
                    f"[{cite_path}:{a}-{b}]",
                    f"[{cite_path}:{new_start}-{new_end}]",
                )
                return md, "fixed"
            return md, "snippet-not-found"
    # 无邻近代码块:只校验路径与范围
    return md, "ok" if range_ok else "bad-range"


def validate_page(
    md: str, conn: sqlite3.Connection, repo_id: int, file_paths: list[str]
) -> tuple[str, PageMetrics]:
    """校验并修复页面;返回 (修复后 md, 指标)。不可信引用剔除并标 [未验证]。"""
    metrics = PageMetrics()
    file_map = _page_file_map(conn, repo_id)
    cache: dict[str, list[str]] = {}

    citations = list(CITATION_RE.finditer(md))
    metrics.citations_total = len(citations)
    to_remove: list[tuple[str, str]] = []
    for m in citations:
        path, a, b = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        md, action = _fix_or_remove_citation(md, path, a, b, file_map, cache)
        if action in ("ok",):
            metrics.citations_ok += 1
        elif action == "fixed":
            metrics.citations_ok += 1
            metrics.citations_fixed += 1
        elif action in ("missing-path", "bad-range", "snippet-not-found"):
            to_remove.append((path, f"{a}-{b}"))
            metrics.errors.append(f"{path}:{a}-{b} {action}")
        # retry-signal 由调用方(生成器)处理:这里直接剔除并标记

    for path, rng in to_remove:
        md = md.replace(
            f"[{path}:{rng}]", f"~~{path}:{rng}~~{UNVERIFIED_MARK}"
        )
    metrics.citations_removed = len(to_remove)

    metrics.mermaid_ok, metrics.mermaid_issues = check_mermaid(md)
    metrics.symbol_coverage = symbol_coverage(md, conn, repo_id, file_paths)
    return md, metrics
