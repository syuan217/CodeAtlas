"""freshness 与人工保护(PLAN §9.6 第 3/4 层)。

- stale:front-matter 的 source_hash 与页面源文件(source_files)当前哈希不一致
  → 页面头部标"已过期";
- 人工保护:页面正文哈希 ≠ 生成时记录的 content_hash(被人改过)、
  front-matter 标 reviewed: true、或根本不是生成页(无 content_hash)→ 永不覆盖;
  重生成写 <同名>.suggested.md 供 diff。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from codeatlas.config import content_hash

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
STALE_BANNER = "> ⚠️ **已过期**:本页基于较早代码生成(source_hash 不一致)。内容仅供参考。\n"


def _parse_front_matter(text: str) -> dict[str, str]:
    m = FM_RE.match(text)
    out: dict[str, str] = {}
    if not m:
        return out
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def read_front_matter(path: Path) -> dict:
    return _parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))


def page_body(text: str) -> str:
    """页面正文:去 front-matter 与工具插入的过期 banner,口径与写入时的 content_hash 一致。"""
    body = FM_RE.sub("", text, count=1).lstrip("\n")
    if body.startswith(STALE_BANNER):
        body = body[len(STALE_BANNER):].lstrip("\n")
    return body


def is_human_owned(path: Path) -> bool:
    """已存在的页面是否归人所有(不得覆盖)。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = _parse_front_matter(text)
    recorded = fm.get("content_hash")
    if not recorded:
        return True  # 不是本工具生成的页(或 front-matter 被人删了)
    if fm.get("reviewed", "").lower() == "true":
        return True
    return content_hash(page_body(text)) != recorded


def write_page(path: Path, front: dict, body_md: str) -> Path:
    """写页面;目标归人所有(见 is_human_owned)→ 改写 <stem>.suggested.md,原文件永不动。"""
    body = body_md.lstrip("\n")
    front = dict(front)
    front["content_hash"] = content_hash(body)
    target = path
    suggested = path.with_suffix(".suggested.md")
    if path.exists() and is_human_owned(path):
        target = suggested
    elif suggested.exists():
        suggested.unlink()  # 原页可覆盖了,旧建议稿已无对照意义
    text = "---\n" + "\n".join(f"{k}: {v}" for k, v in front.items()) + "\n---\n\n" + body
    target.parent.mkdir(parents=True, exist_ok=True)  # 章节子目录
    target.write_text(text, encoding="utf-8")
    return target


def is_stale(path: Path, current_source_hash: str) -> bool:
    fm = read_front_matter(path)
    if not fm:
        return False
    return fm.get("source_hash", "") != current_source_hash


def mark_stale_pages(
    wiki_repo_dir: Path, hash_of_files: Callable[[list[str]], str]
) -> list[str]:
    """递归扫描已生成页面(页面在章节子目录里),按 front-matter 的 source_files
    重算源文件哈希;过期的在 front-matter 后插 banner,恢复新鲜的撤掉 banner(幂等)。"""
    stale: list[str] = []
    if not wiki_repo_dir.is_dir():
        return stale
    for page in sorted(wiki_repo_dir.rglob("*.md")):
        if page.name.endswith(".suggested.md"):
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        fm = _parse_front_matter(text)
        files = [f for f in fm.get("source_files", "").split(",") if f]
        if not files:
            continue
        m = FM_RE.match(text)
        if m is None:
            continue
        rest = text[m.end():]
        has_banner = rest.lstrip("\n").startswith(STALE_BANNER)
        if fm.get("source_hash", "") != hash_of_files(files):
            stale.append(page.stem)
            if not has_banner:
                page.write_text(m.group(0) + STALE_BANNER + rest, encoding="utf-8")
        elif has_banner:
            body = rest.lstrip("\n")[len(STALE_BANNER):]
            page.write_text(m.group(0) + body, encoding="utf-8")
    return stale
