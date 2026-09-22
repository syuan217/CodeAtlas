"""freshness 与人工保护(PLAN §9.6 第 3/4 层)。

- stale:front-matter 的 source_hash 与模块当前哈希不一致 → 页面头部标"已过期";
- 人工保护:页面当前内容哈希 ≠ 生成时记录的 content_hash → 被人改过,
  永不覆盖;重生成写 <同名>.suggested.md 供 diff。
"""

from __future__ import annotations

import re
from pathlib import Path

from codeatlas.config import content_hash

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
STALE_BANNER = "> ⚠️ **已过期**:本页基于较早代码生成(source_hash 不一致)。内容仅供参考。\n"


def read_front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = FM_RE.match(text)
    out: dict[str, str] = {}
    if not m:
        return out
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def write_page(path: Path, front: dict, body_md: str) -> Path:
    """写页面;目标已存在且 body 哈希 ≠ 记录的 content_hash(人工修改过)→
    改写 <stem>.suggested.md,原文件永不动。"""
    body = body_md.lstrip("\n")
    front = dict(front)
    front["content_hash"] = content_hash(body)
    target = path
    if path.exists():
        old_fm = read_front_matter(path)
        current_body_hash = content_hash(
            FM_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        )
        if old_fm.get("content_hash") and current_body_hash != old_fm.get("content_hash"):
            target = path.with_suffix(".suggested.md")
    text = "---\n" + "\n".join(f"{k}: {v}" for k, v in front.items()) + "\n---\n\n" + body
    target.parent.mkdir(parents=True, exist_ok=True)  # 模块 id 含 "/" 时为子目录
    target.write_text(text, encoding="utf-8")
    return target


def is_stale(path: Path, current_source_hash: str) -> bool:
    fm = read_front_matter(path)
    if not fm:
        return False
    return fm.get("source_hash", "") != current_source_hash


def mark_stale_pages(wiki_repo_dir: Path, current_hashes: dict[str, str]) -> list[str]:
    """扫描已生成页面,过期的在 front-matter 后插 banner(幂等:已有不重复插)。"""
    stale: list[str] = []
    if not wiki_repo_dir.is_dir():
        return stale
    for page in sorted(wiki_repo_dir.glob("*.md")):
        if page.name.endswith(".suggested.md"):
            continue
        fm = read_front_matter(page)
        module_id = fm.get("module_id", "")
        if not module_id or module_id not in current_hashes:
            continue
        if is_stale(page, current_hashes[module_id]):
            text = page.read_text(encoding="utf-8", errors="replace")
            if STALE_BANNER not in text:
                m = FM_RE.match(text)
                if m:
                    text = m.group(0) + STALE_BANNER + text[m.end():]
                else:
                    text = STALE_BANNER + text
                page.write_text(text, encoding="utf-8")
            stale.append(page.stem)
    return stale
