"""chunker.py:符号切块、词级回退(真实行号)、markdown H2 切块。"""

from pathlib import Path

from codeatlas.config import content_hash
from codeatlas.ingest.chunker import (
    ChunkOut,
    chunk_code,
    chunk_markdown,
    chunk_text,
    estimate_tokens,
    word_fallback,
)
from codeatlas.ingest.symbols import detect_language, extract_symbols

FIXTURES = Path(__file__).parent / "fixtures"


def make_fs(rel: str, root: str):
    p = FIXTURES / root / rel
    src = p.read_bytes()
    return src.decode("utf-8"), extract_symbols(rel, detect_language(rel), src)


# ---------------------------------------------------------------------------
# code:函数/类切块
# ---------------------------------------------------------------------------

def test_java_method_chunks_have_real_line_numbers():
    text, fs = make_fs("com/example/service/UserService.java", "java_mini")
    chunks = chunk_code(text, fs, max_tokens=512)
    by_title = {c.title: c for c in chunks}
    validate = by_title["com.example.service.UserService#validate"]
    assert validate.line_start == 10
    assert validate.line_end == 19
    assert "public boolean validate" in validate.content.splitlines()[0]
    summarize = by_title["com.example.service.UserService#summarize"]
    assert summarize.line_start == 21
    assert summarize.line_end == 44


def test_java_class_chunk_is_decl_plus_fields_with_javadoc():
    text, fs = make_fs("com/example/model/User.java", "java_mini")
    chunks = chunk_code(text, fs, max_tokens=512)
    cls = next(c for c in chunks if c.title == "com.example.model.User")
    # 起点 = javadoc 第 3 行,终点 = 第一个方法(getId, L11)前一行 L10
    assert cls.line_start == 3
    assert cls.line_end == 10
    body_lines = cls.content.splitlines()
    assert body_lines[0].strip().startswith("/**")
    assert any("private Long id;" in ln for ln in body_lines)
    assert "getId" not in cls.content  # 方法体不进类块


def test_oversized_symbol_falls_back_to_word_chunks():
    text, fs = make_fs("com/example/service/UserService.java", "java_mini")
    # summarize 方法 L21-44,故意把上限压到 30 token 触发词级回退
    chunks = chunk_code(text, fs, max_tokens=30)
    summ = [c for c in chunks if c.title == "com.example.service.UserService#summarize"]
    assert len(summ) >= 2
    # 所有块行号落在方法范围内且首块起点为真实方法起始行
    assert summ[0].line_start == 21
    assert all(21 <= c.line_start <= c.line_end <= 44 for c in summ)
    # 块内容拼接 ≈ 原方法文本(词级,空白略有差)
    joined = "\n".join(c.content for c in summ)
    original = "\n".join(text.splitlines()[20:44])
    assert "".join(joined.split()) == "".join(original.split())


def test_word_fallback_line_numbers_exact():
    text = "alpha beta\ngamma delta\n\nepsilon"
    chunks = word_fallback(text, 1)  # 每词一块
    assert [(c.line_start, c.line_end) for c in chunks] == [
        (1, 1),  # alpha
        (1, 1),  # beta
        (2, 2),  # gamma
        (2, 2),  # delta
        (4, 4),  # epsilon(空行跳过)
    ]


def test_word_fallback_respects_base_line():
    # "aaaa bbbb" 整体 estimate=3 > 1 → 切两块;均在 100 行(单行文本)
    chunks = word_fallback("aaaa bbbb", 1, base_line=100)
    assert len(chunks) == 2
    assert chunks[0].line_start == 100
    assert chunks[1].line_start == 100


def test_symbolless_file_whole_fallback():
    text = "SELECT 1 FROM t;\nSELECT 2 FROM u;\n"
    from codeatlas.ingest.symbols import FileSymbols, Symbol

    fs = FileSymbols(rel="q.sql", lang="sql", module=Symbol("module", "q.sql", "q.sql", 1, 1, None, None))
    chunks = chunk_code(text, fs, max_tokens=512)
    assert len(chunks) == 1
    assert chunks[0].title == "q.sql"
    assert chunks[0].line_start == 1 and chunks[0].line_end == 2


def test_chunk_content_hash_filled():
    text, fs = make_fs("pkg/mod.py", "py_mini")
    chunks = chunk_code(text, fs, max_tokens=512)
    assert all(c.content_hash == content_hash(c.content) for c in chunks)


def test_python_chunks():
    text, fs = make_fs("pkg/mod.py", "py_mini")
    chunks = chunk_code(text, fs, max_tokens=512)
    titles = {c.title for c in chunks}
    assert "pkg.mod.clamp" in titles
    assert "pkg.mod.Counter" in titles
    assert "pkg.mod.Counter#bump" in titles


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

MD_SAMPLE = """# 项目

前言段落。

## 安装

npm i。

```bash
## 这不是标题(在代码块里)
echo hi
```

## 用法

见上。
"""


def test_markdown_h2_split_and_fence_protection():
    chunks = chunk_markdown(MD_SAMPLE, max_tokens=512)
    titles = [c.title for c in chunks]
    assert titles == ["项目", "安装", "用法"]  # 第一块 title 取 H1 文本
    install = chunks[1]
    assert install.line_start == 5
    assert install.line_end == 13  # 到 "## 用法"(L14)前一行,含 fence 闭合行 L12 与空行 L13
    assert "## 这不是标题" in install.content  # fence 内标题不切块
    usage = chunks[2]
    assert usage.line_start == 14


def test_markdown_oversize_section_falls_back():
    long_md = "# t\n\n## big\n\n" + ("word " * 500)
    chunks = chunk_markdown(long_md, max_tokens=50)
    big = [c for c in chunks if c.title == "big"]
    assert len(big) >= 2
    assert all(c.line_start >= 3 for c in big)


def test_fixture_readme_chunks():
    text = (FIXTURES / "java_mini" / "README.md").read_text()
    chunks = chunk_markdown(text, max_tokens=512)
    titles = [c.title for c in chunks]
    assert "Model Layer" in titles
    assert all(c.kind == "doc" for c in chunks)


def test_chunk_text_dispatch():
    text = "# h\n\n## a\n\nxx\n"
    from codeatlas.ingest.symbols import FileSymbols, Symbol

    fs = FileSymbols(rel="r.md", lang="markdown",
                     module=Symbol("module", "r.md", "r.md", 1, 1, None, None))
    assert chunk_text(text, "markdown", fs, 512)[0].kind == "doc"


def test_estimate_tokens():
    assert estimate_tokens("a" * 300) == 100
    assert estimate_tokens("") == 1
