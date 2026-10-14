"""wiki 结构规划(PLAN §9.7):XML 结构规划 + 四级容错解析。

- 结构规划输出页面候选来自模块划分(模型只做选择与排序);
- 逐页生成、校验重试、单页失败隔离与落盘在 gencode/wiki.py。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from codeatlas.providers.llm import LLMProvider

PROMPTS_DIR = Path(__file__).parent / "prompts"
GENERATOR_VERSION = "v2"  # 与在用的 *.v2.md prompt 一致


@dataclass
class Page:
    id: str
    title: str = ""
    importance: str = "normal"
    sections: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    chapter_id: str = ""       # 所属章(v2)
    chapter_title: str = ""


_FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)
_PAGE_RE = re.compile(
    r'<page\b([^>]*)>(.*?)</page>', re.S | re.I)
_PAGE_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def parse_structure(text: str) -> list[Page]:
    """四级容错(DeepWiki 思路):剥 fence → ElementTree → regex 逐块 → 截断抢救。"""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("<wiki_structure")
    if start >= 0:
        text = text[start:]
    # L2:严格 XML
    try:
        root = ET.fromstring(text)
        pages = _pages_from_element(root)
        if pages:
            return pages
    except ET.ParseError:
        pass
    # L3:regex 逐 <page> 块;部分命中(<page 出现数多于闭合数)时继续 L4 抢救
    pages = _pages_from_regex(text)
    n_open = text.count("<page")
    if pages and len(pages) >= n_open:
        return pages
    # L4:截断抢救——补齐未闭合的尾部块
    rescued = text
    if "<page" in rescued and "</page>" not in rescued.rsplit("<page", 1)[-1]:
        rescued = rescued + "</page></wiki_structure>"
    elif rescued.count("<page") > rescued.count("</page>"):
        rescued += "</page>" * (rescued.count("<page") - rescued.count("</page>"))
        rescued += "</wiki_structure>"
    rescued_pages = _pages_from_regex(rescued)
    return rescued_pages if len(rescued_pages) > len(pages) else pages


def _pages_from_element(root) -> list[Page]:
    pages = []
    for ch in root.iter("chapter"):
        cid = (ch.get("id") or "").strip()
        ctitle = (ch.get("title") or "").strip()
        for el in ch.iter("page"):
            pages.append(_page_from_attrs(el, cid, ctitle))
    if not pages:  # v1 平铺结构兼容
        for el in root.iter("page"):
            pages.append(_page_from_attrs(el, "", ""))
    return [p for p in pages if p.id]


def _page_from_attrs(el, cid: str, ctitle: str) -> Page:
    return Page(
        id=(el.get("id") or "").strip(),
        title=(el.get("title") or "").strip(),
        importance=(el.get("importance") or "normal").strip(),
        sections=[(s.text or "").strip() for s in el.iter("section") if s.text],
        file_paths=[
            p.strip()
            for p in (el.findtext("filePaths") or "").split(",")
            if p.strip()
        ],
        related=[
            r.strip()
            for r in (el.findtext("relatedPages") or "").split(",")
            if r.strip()
        ],
        chapter_id=cid,
        chapter_title=ctitle,
    )


def _pages_from_regex(text: str) -> list[Page]:
    pages = []
    chapters = list(re.finditer(r'<chapter\b([^>]*)>', text))
    for m in _PAGE_RE.finditer(text):
        cid, ctitle = "", ""
        for cm in chapters:
            if cm.start() < m.start():
                attrs = dict(_PAGE_ATTR_RE.findall(cm.group(1)))
                cid = attrs.get("id", "")
                ctitle = attrs.get("title", "")
            else:
                break
        attrs = dict(_PAGE_ATTR_RE.findall(m.group(1)))
        body = m.group(2)

        def tag_content(tag: str) -> str:
            tm = re.search(rf"<{tag}>(.*?)</{tag}>", body, re.S | re.I)
            return tm.group(1).strip() if tm else ""

        pages.append(
            Page(
                id=attrs.get("id", "").strip(),
                title=attrs.get("title", "").strip(),
                importance=attrs.get("importance", "normal"),
                sections=[
                    s.strip() for s in re.findall(
                        r"<section>(.*?)</section>", body, re.S | re.I
                    ) if s.strip()
                ],
                file_paths=[p.strip() for p in tag_content("filePaths").split(",") if p.strip()],
                related=[r.strip() for r in tag_content("relatedPages").split(",") if r.strip()],
                chapter_id=cid,
                chapter_title=ctitle,
            )
        )
    return [p for p in pages if p.id]


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


async def plan_pages(
    llm: LLMProvider,
    repo_summary: str,
    modules_text: str,
    size_hint: str,
) -> list[Page]:
    prompt = (
        load_prompt("structure_planning.v2")
        .replace("{repo_summary}", repo_summary)
        .replace("{modules}", modules_text)
        .replace("{size_hint}", size_hint)
    )
    r = await llm.chat(
        [{"role": "user", "content": prompt}], stage="wiki",
        temperature=0.2, max_tokens=8192, thinking_disabled=True,
    )
    return parse_structure(r.content)
