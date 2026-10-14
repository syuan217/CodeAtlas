"""M5 wiki 生成测试:XML 四级容错、硬校验(snippet 反查)、人工保护、端到端。"""

import asyncio
import json
import re

import httpx
import pytest

from codeatlas.config import RepoCfg
from codeatlas.gencode.freshness import (
    mark_stale_pages,
    read_front_matter,
    write_page,
)
from codeatlas.gencode.generate import parse_structure
from codeatlas.gencode.segment import segment
from codeatlas.gencode.validate import check_mermaid, validate_page
from conftest import Env, FakeEmbedServer, copy_fixture

GOOD_XML = """<wiki_structure>
  <page id="overview" title="总览" importance="high">
    <section>架构</section>
    <filePaths>com/example/model/User.java</filePaths>
    <relatedPages>com</relatedPages>
  </page>
  <page id="com" title="工具" importance="normal">
    <section>字符串工具</section>
    <filePaths>com/example/util/Strings.java</filePaths>
  </page>
</wiki_structure>"""

NL = chr(10)
MERMAID_TD = f"```mermaid{NL}graph TD{NL} A[foo.java:1] --> B[bar.java:2]{NL}```"
MERMAID_LR = f"```mermaid{NL}graph LR{NL} A --> B{NL}```"
MERMAID_NOREF = f"```mermaid{NL}graph TD{NL} A --> B{NL}```"


def test_parse_structure_strict():
    pages = parse_structure(GOOD_XML)
    assert [p.id for p in pages] == ["overview", "com"]
    assert pages[0].file_paths == ["com/example/model/User.java"]
    assert pages[0].related == ["com"]


def test_parse_structure_fence_wrapped():
    wrapped = f"前缀{NL}```xml{NL}{GOOD_XML}{NL}```"
    assert [p.id for p in parse_structure(wrapped)] == ["overview", "com"]


def test_parse_structure_truncated_rescue():
    truncated = GOOD_XML[: GOOD_XML.rfind("</page>")]
    pages = parse_structure(truncated)
    ids = [p.id for p in pages]
    assert "overview" in ids and "com" in ids


def test_parse_structure_regex_fallback():
    broken = f"<wiki_structure>{NL}" + GOOD_XML[
        GOOD_XML.index("<page"):
    ].replace("</wiki_structure>", "")
    pages = parse_structure(broken)
    assert len(pages) == 2


def test_segment_modules(env, tmp_path):
    root = copy_fixture("java_mini", tmp_path)
    env.run(RepoCfg(name="jmini", path=root))
    repo_id = env.one("SELECT id FROM repos WHERE name='jmini'")["id"]
    modules = segment(env.conn, repo_id)
    assert modules
    all_files = {f for m in modules for f in m.files}
    assert "com/example/util/Strings.java" in all_files
    for m in modules:
        assert m.source_hash and m.files


def test_validate_page_snippet_lookup(env, tmp_path):
    root = copy_fixture("java_mini", tmp_path)
    env.run(RepoCfg(name="jmini", path=root))
    repo_id = env.one("SELECT id FROM repos WHERE name='jmini'")["id"]

    real_snippet = "    public static String truncate(String s, int max) {"
    md_ok = (
        f"# 页面{NL}{NL}```java{NL}{real_snippet}{NL}```{NL}{NL}"
        f"引用 [com/example/util/Strings.java:99-99](…) 与 "
        f"[com/example/util/Strings.java:1-5](…){NL}"
    )
    fixed, metrics = validate_page(md_ok, env.conn, repo_id, [])
    assert "[com/example/util/Strings.java:99-99]" not in fixed
    assert metrics.citations_fixed >= 1
    assert metrics.citations_ok >= 1

    md_fake = (
        f"# 假引用页{NL}{NL}```java{NL}"
        f"this snippet does not exist anywhere in repo 000{NL}"
        f"```{NL}{NL}引用 [com/example/util/Strings.java:1-5](…){NL}"
    )
    fixed2, m2 = validate_page(md_fake, env.conn, repo_id, [])
    assert m2.citations_removed == 1
    assert "[未验证]" in fixed2

    _, m3 = validate_page("引用 [not/exist/File.java:1-9](…)", env.conn, repo_id, [])
    assert m3.citations_removed == 1


def test_check_mermaid():
    ok, _ = check_mermaid(MERMAID_TD)
    assert ok
    ok_lr, issues = check_mermaid(MERMAID_LR)
    assert not ok_lr and any("TD" in i for i in issues)
    ok_nr, issues2 = check_mermaid(MERMAID_NOREF)
    assert not ok_nr and any("行号" in i for i in issues2)


def test_freshness_human_protection(tmp_path):
    page = tmp_path / "01-ch" / "mod.md"
    target = write_page(
        page, {"repo": "r", "source_files": "a.java", "source_hash": "h1"},
        f"# 正文{NL}内容",
    )
    assert target == page
    page.write_text(page.read_text(encoding="utf-8").replace("# 正文", "# 人工改过的正文"), encoding="utf-8")
    target2 = write_page(
        page, {"repo": "r", "source_files": "a.java", "source_hash": "h2"}, "# 新版正文"
    )
    assert target2 == page.with_suffix(".suggested.md")
    assert "人工改过的正文" in page.read_text(encoding="utf-8")
    fm = read_front_matter(page)
    assert fm["source_hash"] == "h1"
    # 页面在章节子目录里:过期扫描必须递归
    stale = mark_stale_pages(tmp_path, lambda files: "h2" if files == ["a.java"] else "")
    assert "mod" in stale
    assert "已过期" in page.read_text(encoding="utf-8")


def test_freshness_unedited_page_is_overwritten(tmp_path):
    """未被人改过的页:重生成直接覆盖原页,不写 .suggested.md(曾因换行口径不一致恒判人工修改)。"""
    page = tmp_path / "01-ch" / "p.md"
    write_page(page, {"repo": "r"}, f"# v1{NL}正文")
    target = write_page(page, {"repo": "r"}, f"# v2{NL}正文")
    assert target == page
    assert "# v2" in page.read_text(encoding="utf-8")
    assert not page.with_suffix(".suggested.md").exists()


def test_freshness_stale_banner_is_not_a_human_edit(tmp_path):
    page = tmp_path / "p.md"
    write_page(page, {"repo": "r", "source_files": "a.java", "source_hash": "h1"}, "# v1")
    assert mark_stale_pages(tmp_path, lambda files: "h2") == ["p"]
    assert "已过期" in page.read_text(encoding="utf-8")
    assert mark_stale_pages(tmp_path, lambda files: "h2") == ["p"]  # 幂等
    assert page.read_text(encoding="utf-8").count("已过期") == 1
    target = write_page(page, {"repo": "r", "source_files": "a.java", "source_hash": "h2"}, "# v2")
    assert target == page                     # 工具自己插的 banner 不算人工修改
    assert "已过期" not in page.read_text(encoding="utf-8")
    assert "# v2" in page.read_text(encoding="utf-8")


def test_freshness_reviewed_or_foreign_page_protected(tmp_path):
    reviewed = tmp_path / "r.md"
    write_page(reviewed, {"repo": "r", "reviewed": "false"}, "# v1")
    reviewed.write_text(reviewed.read_text(encoding="utf-8").replace("reviewed: false", "reviewed: true"), encoding="utf-8")
    assert write_page(reviewed, {"repo": "r", "reviewed": "false"}, "# v2") == \
        reviewed.with_suffix(".suggested.md")
    assert "# v1" in reviewed.read_text(encoding="utf-8")

    foreign = tmp_path / "f.md"                # 无 front-matter:人写的同名文件
    foreign.write_text("# 我自己的笔记", encoding="utf-8")
    assert write_page(foreign, {"repo": "r"}, "# 生成") == foreign.with_suffix(".suggested.md")
    assert foreign.read_text(encoding="utf-8") == "# 我自己的笔记"


def test_freshness_leftover_suggestion_removed_on_overwrite(tmp_path):
    page = tmp_path / "p.md"
    write_page(page, {"repo": "r"}, "# v1")
    page.with_suffix(".suggested.md").write_text("旧建议稿", encoding="utf-8")
    write_page(page, {"repo": "r"}, "# v2")
    assert not page.with_suffix(".suggested.md").exists()


def _wiki_llm_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    user = body["messages"][-1]["content"]
    if "结构规划师" in user or "主编" in user:
        content = f"```xml{NL}{GOOD_XML}{NL}```"
    else:
        content = (
            f"# 工具模块{NL}{NL}"
            f"<details><summary>Relevant source files</summary>{NL}"
            f"- com/example/util/Strings.java{NL}</details>{NL}{NL}"
            f"```java{NL}"
            f"    public static String truncate(String s, int max) {{"
            f"{NL}```{NL}{NL}"
            f"truncate 实现见 [com/example/util/Strings.java:1-5]()。{NL}{NL}"
            f"```mermaid{NL}graph TD{NL}"
            f" A[Strings.java:5] --> B[Strings.java:6]{NL}```{NL}"
        )
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "model": "test-model",
    })


def test_generate_wiki_end_to_end(tmp_path, monkeypatch):
    from codeatlas import config as config_mod

    monkeypatch.setattr(config_mod, "WIKI_DIR", tmp_path / "wiki")
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    root = copy_fixture("java_mini", tmp_path)
    env_repo = RepoCfg(name="jmini", path=root)
    e.run(env_repo)

    from codeatlas.gencode.wiki import generate_wiki
    from codeatlas.providers.llm import LLMProvider

    llm = LLMProvider(
        e.settings, e.conn, transport=httpx.MockTransport(_wiki_llm_handler),
        backoff_base=0,
    )

    stats = asyncio.run(
        generate_wiki(
            env_repo, e.conn, llm, e.settings, e.lance(), e.provider(),
            concise=True, budget_tokens=50_000,
        )
    )
    assert stats.modules >= 1
    assert stats.pages_planned >= 2  # 规划 + 按需兜底
    assert stats.pages_generated == stats.pages_planned

    wiki_dir = tmp_path / "wiki" / "jmini"
    mds = list(wiki_dir.glob("**/*.md"))
    assert (wiki_dir / "README.md").exists()  # 目录页
    assert len(mds) >= 3  # README + 2 内容页
    content_pages = [p for p in mds if p.name != "README.md"]
    fm = read_front_matter(content_pages[0])
    assert fm["repo"] == "jmini"
    # v2 导航:面包屑 + 上下页注入
    body = content_pages[0].read_text(encoding="utf-8")
    assert "📖 目录" in body
    assert "上一页" in body or "下一页" in body
    assert "file://" in body or "](…" not in body  # 源码引用已转绝对链接(或无引用)

    n = e.one("SELECT COUNT(*) AS c FROM chunks WHERE kind='wiki'")["c"]
    assert n == stats.indexed_chunks > 0
    n_fts = e.one(
        "SELECT COUNT(*) AS c FROM chunks_fts f JOIN chunks c ON c.id=f.rowid "
        "WHERE chunks_fts MATCH 'truncate' AND c.kind='wiki'"
    )["c"]
    assert n_fts >= 1  # wiki chunk 本身可被关键词检索(此前只写 chunks 未写 FTS)


def _wiki_env(tmp_path, monkeypatch):
    from codeatlas import config as config_mod

    monkeypatch.setattr(config_mod, "WIKI_DIR", tmp_path / "wiki")
    e = Env(tmp_path, monkeypatch, server=FakeEmbedServer())
    repo = RepoCfg(name="jmini", path=copy_fixture("java_mini", tmp_path))
    e.run(repo)
    return e, repo, tmp_path / "wiki" / "jmini"


def _run_wiki(e, repo, handler, tracker=None, **kw):
    from codeatlas.gencode.wiki import generate_wiki
    from codeatlas.providers.llm import LLMProvider

    llm = LLMProvider(e.settings, e.conn, transport=httpx.MockTransport(handler),
                      backoff_base=0, tracker=tracker)
    return asyncio.run(generate_wiki(repo, e.conn, llm, e.settings, e.lance(), e.provider(),
                                     concise=True, budget_tokens=50_000, **kw))


def test_wiki_one_failed_page_does_not_discard_others(tmp_path, monkeypatch):
    e, repo, wiki_dir = _wiki_env(tmp_path, monkeypatch)

    def handler(request):
        user = json.loads(request.content)["messages"][-1]["content"]
        if re.search(r"页面 id[:\uff1a]com[(\uff08]", user):
            return httpx.Response(400, json={"error": "bad request"})
        return _wiki_llm_handler(request)

    stats = _run_wiki(e, repo, handler)
    assert stats.pages_failed >= 1 and any("com" in err for err in stats.errors)
    pages = [p for p in wiki_dir.rglob("*.md") if p.name != "README.md"]
    assert pages and all(p.stem != "com" for p in pages)
    toc = (wiki_dir / "README.md").read_text(encoding="utf-8")
    assert "com.md" not in toc  # 目录不链接未生成的页


def test_wiki_cost_limit_keeps_finished_pages(tmp_path, monkeypatch):
    from codeatlas.providers._retry import CostLimitExceeded, CostTracker

    e, repo, wiki_dir = _wiki_env(tmp_path, monkeypatch)
    e.settings = e.settings.model_copy(update={"llm_price_prompt": 1e4,
                                               "llm_price_completion": 1e4})
    tracker = CostTracker(limit=3.5)  # 每次调用 1.5 元:规划 + 1 页后的下一次即超
    with pytest.raises(CostLimitExceeded):
        _run_wiki(e, repo, _wiki_llm_handler, tracker=tracker)
    pages = [p for p in wiki_dir.rglob("*.md") if p.name != "README.md"]
    assert len(pages) >= 1  # 已付费生成的页落盘,不随异常丢失


def test_wiki_orphan_pages_cleaned_human_ones_kept(tmp_path, monkeypatch):
    e, repo, wiki_dir = _wiki_env(tmp_path, monkeypatch)
    old = wiki_dir / "09-old" / "old.md"
    write_page(old, {"repo": "jmini"}, "# 旧规划遗留页")
    mine = wiki_dir / "09-old" / "mine.md"
    write_page(mine, {"repo": "jmini"}, "# 生成页")
    mine.write_text(mine.read_text(encoding="utf-8").replace("# 生成页", "# 人工改写过"), encoding="utf-8")
    _run_wiki(e, repo, _wiki_llm_handler)
    assert not old.exists()
    assert mine.exists()
    titles = {r["title"] for r in e.q("SELECT title FROM chunks WHERE kind='wiki'")}
    assert not any(":old:" in t for t in titles)


def test_wiki_cli_exit_code_reflects_failures(monkeypatch):
    import typer

    import codeatlas.cli as cli_mod
    from codeatlas.config import RepoCfg as _R

    monkeypatch.setattr(cli_mod, "load_repos",
                        lambda: [_R(name="a", path="/tmp"), _R(name="b", path="/tmp")])
    monkeypatch.setattr("codeatlas.config.load_repos",
                        lambda: [_R(name="a", path="/tmp"), _R(name="b", path="/tmp")])

    async def fake_one(cfg, concise, max_pages, tracker):
        return "ok" if cfg.name == "a" else "failed"

    monkeypatch.setattr(cli_mod, "_wiki_one", fake_one)
    with pytest.raises(typer.Exit) as ei:
        asyncio.run(cli_mod._wiki(None, False, None))
    assert ei.value.exit_code == 1
