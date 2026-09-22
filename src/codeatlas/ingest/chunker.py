"""切块器(PLAN §9.3)。

- code chunk:function/method 一个 chunk(签名行到函数尾);class 声明+字段+
  类 javadoc 单独一个 chunk;无符号文件(xml/sql 等)整体词级回退。
- doc chunk:markdown 按标题层级切(H2 为界),代码块不切断。
- 超过 chunk_max_tokens(默认 512)→ 词级回退:空格切词、按 token 上限累积,
  每块回算真实 1-based 起止行号(DeepWiki LineTracking 思路)。
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from codeatlas.config import content_hash
from codeatlas.ingest.symbols import FileSymbols

_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """近似 token 数(1 token ≈ 3 字符,代码场景偏保守,宁可早切)。"""
    return max(1, len(text) // 3)


@dataclass
class ChunkOut:
    kind: str  # code / doc
    title: str | None
    content: str
    line_start: int  # 1-based,真实行号
    line_end: int
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = content_hash(self.content)


# ---------------------------------------------------------------------------
# 词级回退(真实行号回算)
# ---------------------------------------------------------------------------

def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def word_fallback(
    text: str,
    max_tokens: int,
    *,
    kind: str = "code",
    title: str | None = None,
    base_line: int = 1,
) -> list[ChunkOut]:
    """词级切块:每块 ≤ max_tokens(近似),行号由字符偏移回算。

    base_line: text 在原文件中的起始行号(切片场景用)。
    """
    if not text.strip():
        return []
    offsets = _line_offsets(text)

    def line_at(char_offset: int) -> int:
        return base_line - 1 + bisect_right(offsets, char_offset)

    chunks: list[ChunkOut] = []
    words = list(_WORD_RE.finditer(text))
    if not words:
        return []
    start = words[0].start()
    prev_end = start
    for w in words:
        if estimate_tokens(text[start:w.end()]) > max_tokens and w.start() > start:
            chunks.append(
                ChunkOut(
                    kind=kind,
                    title=title,
                    content=text[start:prev_end],
                    line_start=line_at(start),
                    line_end=line_at(max(prev_end - 1, start)),
                )
            )
            start = w.start()
        prev_end = w.end()
    chunks.append(
        ChunkOut(
            kind=kind,
            title=title,
            content=text[start:prev_end],
            line_start=line_at(start),
            line_end=line_at(max(prev_end - 1, start)),
        )
    )
    return chunks


def _slice_lines(lines: list[str], start_1based: int, end_1based: int) -> str:
    return "\n".join(lines[start_1based - 1 : end_1based])


# ---------------------------------------------------------------------------
# code 切块
# ---------------------------------------------------------------------------

def chunk_code(
    text: str,
    fs: FileSymbols,
    max_tokens: int,
) -> list[ChunkOut]:
    """按符号切块;符号文本本身超限则对该符号做词级回退。"""
    lines = text.splitlines()
    out: list[ChunkOut] = []

    # 每个类找到第一个子符号(方法/嵌套类)的起始行,类块 = javadoc/声明..其前一行
    first_child_line: dict[int, int] = {}
    for i, s in enumerate(fs.symbols):
        if s.parent is not None:
            cur = first_child_line.get(s.parent)
            if cur is None or s.line_start < cur:
                first_child_line[s.parent] = s.line_start

    for idx, s in enumerate(fs.symbols):
        if s.kind in ("function", "method"):
            body = _slice_lines(lines, s.line_start, s.line_end)
            out.extend(_emit(body, s, s.line_start, max_tokens, kind="code"))
        elif s.kind in ("class", "interface"):
            start = s.javadoc_start or s.line_start
            end = (first_child_line.get(idx, s.line_end + 1)) - 1
            if end < start:
                end = s.line_end  # 无子符号:整个类范围(纯声明类)
            head = _slice_lines(lines, start, end)
            out.extend(_emit(head, s, start, max_tokens, kind="code"))

    if not out and text.strip():  # 无符号文件(xml/sql/空壳)整体词级回退
        out = word_fallback(text, max_tokens, kind="code", title=fs.rel)
    return out


def _emit(body: str, sym, start_line: int, max_tokens: int, *, kind: str) -> list[ChunkOut]:
    if not body.strip():
        return []
    if estimate_tokens(body) <= max_tokens:
        line_end = start_line + body.count("\n")
        return [ChunkOut(kind=kind, title=sym.qualified_name, content=body,
                         line_start=start_line, line_end=line_end)]
    return word_fallback(body, max_tokens, kind=kind, title=sym.qualified_name,
                         base_line=start_line)


# ---------------------------------------------------------------------------
# markdown 切块
# ---------------------------------------------------------------------------

def chunk_markdown(text: str, max_tokens: int) -> list[ChunkOut]:
    """按 H2 切块;```/``` 代码块内的标题行不切块;超限块词级回退。"""
    lines = text.splitlines()
    if not lines:
        return []
    bounds: list[tuple[int, str | None]] = []  # (起始行 1-based, 标题)
    fence = False
    title: str | None = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = not fence
            continue
        if not fence and line.startswith("## "):
            if bounds:
                bounds.append((i + 1, line[3:].strip()))
            else:
                bounds = [(1, None), (i + 1, line[3:].strip())]
    if not bounds:
        bounds = [(1, None)]

    out: list[ChunkOut] = []
    for bi, (start, heading) in enumerate(bounds):
        end = bounds[bi + 1][0] - 1 if bi + 1 < len(bounds) else len(lines)
        body = _slice_lines(lines, start, end)
        if not body.strip():
            continue
        doc_title = heading if heading else (lines[0].lstrip("# ").strip() or None)
        if estimate_tokens(body) <= max_tokens:
            out.append(ChunkOut(kind="doc", title=doc_title, content=body,
                                line_start=start, line_end=end))
        else:
            out.extend(word_fallback(body, max_tokens, kind="doc", title=doc_title,
                                     base_line=start))
    return out


def chunk_text(text: str, lang: str, fs: FileSymbols, max_tokens: int) -> list[ChunkOut]:
    """入口:按语言分流。"""
    if lang in ("markdown",):
        return chunk_markdown(text, max_tokens)
    return chunk_code(text, fs, max_tokens)
