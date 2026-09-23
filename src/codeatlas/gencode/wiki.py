"""wiki 生成编排 v2(zread 风格):章节书结构 + 叙述页 + 导航注入 + 入索引。

结构:wiki/<repo>/README.md(目录页)+ <NN>-<chapter>/<page>.md;
每页确定性注入面包屑(› 目录/› 章)与上一页/下一页;源码行内引用
后处理为 file:// 绝对链接(本地 markdown 阅读器可点击跳转)。
"""

from __future__ import annotations

import asyncio
import posixpath
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from codeatlas.config import Settings, WIKI_DIR
from codeatlas.gencode.freshness import mark_stale_pages, write_page
from codeatlas.gencode.generate import (
    GENERATOR_VERSION,
    Page,
    load_prompt,
    parse_structure,
)
from codeatlas.gencode.pack import DEFAULT_BUDGET_TOKENS, interface_summary, pack_files
from codeatlas.gencode.segment import Module, segment
from codeatlas.gencode.validate import PageMetrics, validate_page
from codeatlas.ingest.chunker import chunk_markdown
from codeatlas.ingest.indexer import upsert_repo
from codeatlas.providers.embedding import EmbeddingProvider
from codeatlas.providers.llm import LLMProvider

MAX_VALIDATE_RETRIES = 2


@dataclass
class WikiStats:
    repo: str
    modules: int = 0
    pages_planned: int = 0
    pages_generated: int = 0
    pages_failed: int = 0
    citations_total: int = 0
    citations_ok: int = 0
    citations_removed: int = 0
    output_dir: str = ""
    indexed_chunks: int = 0
    errors: list[str] = field(default_factory=list)


def _repo_summary(repo_root: Path) -> str:
    parts = []
    for name in ("README.md", "CONTEXT.md"):
        p = repo_root / name
        if p.is_file():
            try:
                parts.append(
                    f"[{name}]\n"
                    + p.read_text(encoding="utf-8", errors="replace")[:2000]
                )
            except OSError:
                pass
    return "\n\n".join(parts) or "(无 README/CONTEXT)"


def _modules_text(modules: list[Module]) -> str:
    lines = []
    for m in modules:
        if len(m.files) < 3:
            continue  # 单文件杂项不成模块
        preview = ", ".join(m.files[:6]) + ("…" if len(m.files) > 6 else "")
        lines.append(f"- {m.id}({len(m.files)} 文件):{preview}")
    return "\n".join(lines)


def module_import_graph(conn: sqlite3.Connection, repo_id: int, modules: list[Module]) -> str:
    mod_of: dict[str, str] = {}
    for m in modules:
        for f in m.files:
            mod_of[f] = m.id
    rows = conn.execute(
        "SELECT fa.path AS src, fb.path AS dst, COUNT(*) AS c FROM edges e "
        "JOIN symbols a ON e.src_id=a.id JOIN files fa ON a.file_id=fa.id "
        "JOIN symbols b ON e.dst_id=b.id JOIN files fb ON b.file_id=fb.id "
        "WHERE e.repo_id=? AND e.kind='IMPORTS' GROUP BY fa.path, fb.path",
        (repo_id,),
    ).fetchall()
    agg: dict[tuple[str, str], int] = {}
    for r in rows:
        ms, md = mod_of.get(r["src"]), mod_of.get(r["dst"])
        if ms and md and ms != md:
            agg[(ms, md)] = agg.get((ms, md), 0) + r["c"]
    lines = ["graph TD"]
    for (a, b), c in sorted(agg.items(), key=lambda x: -x[1])[:20]:
        lines.append(
            f'  {a.replace("/", "_")}["{a}"] -->|{c} 依赖| {b.replace("/", "_")}["{b}"]'
        )
    return "\n".join(lines)


def _rel_link(target: Path, from_file: Path) -> str:
    """目标页相对当前文件的 markdown 链接路径(posix 风格)。"""
    rel = posixpath.relpath(
        str(target), str(from_file.parent) if str(from_file.parent) != "" else "."
    )
    return rel.replace("\\", "/")


async def _chat_page(llm: LLMProvider, prompt: str) -> str:
    r = await llm.chat(
        [{"role": "user", "content": prompt}], stage="wiki",
        temperature=0.2, max_tokens=16384, thinking_disabled=True,
    )
    return r.content


async def _generate_one(
    llm: LLMProvider,
    page: Page,
    prompt: str,
    conn: sqlite3.Connection,
    repo_id: int,
) -> tuple[str, PageMetrics]:
    last_md, metrics = "", PageMetrics()
    feedback = ""
    for _attempt in range(MAX_VALIDATE_RETRIES + 1):
        p = prompt + (
            f"\n\n上一版硬校验失败清单(逐条修复后重新输出完整页面):\n{feedback}"
            if feedback
            else ""
        )
        md = await _chat_page(llm, p)
        md, metrics = validate_page(md, conn, repo_id, page.file_paths)
        if metrics.citations_removed == 0:
            return md, metrics
        feedback = "\n".join(metrics.errors[:10])
        last_md = md
    return last_md, metrics


async def generate_wiki(
    repo_cfg,
    conn: sqlite3.Connection,
    llm: LLMProvider,
    settings: Settings,
    lance,
    embedder: EmbeddingProvider | None,
    *,
    concise: bool = False,
    max_pages: int | None = None,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    concurrency: int = 2,
) -> WikiStats:
    repo_id = upsert_repo(conn, repo_cfg)
    repo_root = Path(repo_cfg.path)
    stats = WikiStats(repo=repo_cfg.name)
    modules = segment(conn, repo_id)
    stats.modules = len(modules)
    if not modules:
        stats.errors.append("没有可划分的模块(先索引仓库)")
        return stats
    mod_by_id = {m.id: m for m in modules}
    mod_hash = {m.id: m.source_hash for m in modules}

    from codeatlas.gencode.generate import plan_pages

    size_hint = "全书 4~6 页、2~3 章" if concise else "全书 8~12 页、4~6 章"
    pages = await plan_pages(
        llm, _repo_summary(repo_root), _modules_text(modules), size_hint
    )
    all_files = {f for m in modules for f in m.files}
    aligned: list[Page] = []
    for p_ in pages:
        p_.file_paths = [f for f in p_.file_paths if f in all_files]
        if p_.file_paths:
            aligned.append(p_)
    if not aligned:
        aligned = [
            Page(id="overview--index", title="架构总览", importance="high",
                 chapter_id="overview", chapter_title="总览")
        ]
    min_pages = 4 if concise else min(8, max(4, len(modules) // 2))
    if len(aligned) < min_pages:
        covered = {p_.chapter_id for p_ in aligned}
        for m in sorted(modules, key=lambda m: -len(m.files)):
            if len(aligned) >= min_pages:
                break
            if m.id in covered or len(m.files) < 3:
                continue
            slug = m.id.replace("/", "-")
            aligned.append(
                Page(
                    id=f"{slug}--index",
                    title=m.id.rsplit("/", 1)[-1],
                    chapter_id=slug,
                    chapter_title=m.id.rsplit("/", 1)[-1],
                    file_paths=m.files[:10],
                )
            )
        stats.errors.append("结构规划页数不足,已按模块规模兜底补足")
    if max_pages:
        aligned = aligned[:max_pages]
    stats.pages_planned = len(aligned)

    wiki_dir = WIKI_DIR / repo_cfg.name
    wiki_dir.mkdir(parents=True, exist_ok=True)
    stats.output_dir = str(wiki_dir)

    chapter_order: list[str] = []
    for p_ in aligned:
        cid = p_.chapter_id or "misc"
        if cid not in chapter_order:
            chapter_order.append(cid)
    chapter_dirname = {
        cid: f"{i + 1:02d}-{cid.replace('/', '-')}"
        for i, cid in enumerate(chapter_order)
    }
    page_targets = [
        (p_, wiki_dir / chapter_dirname[p_.chapter_id or "misc"] / f"{p_.id}.md")
        for p_ in aligned
    ]

    sem = asyncio.Semaphore(max(1, concurrency))
    repo_abs = repo_root

    def to_file_links(md: str) -> str:
        import re as _re

        def _sub(m):
            target = repo_abs / m.group(1)
            return f"(file://{target})" if target.exists() else m.group(0)

        return _re.sub(r"\(([^)\s:]+):\d+-\d+\)", _sub, md)

    generated: dict[str, tuple[str, PageMetrics]] = {}

    async def run_page(page: Page) -> None:
        packed, _used = pack_files(
            conn, repo_id, repo_root, page.file_paths, budget_tokens
        )
        neighbors = (
            "\n\n".join(
                interface_summary(conn, repo_id, mod_by_id[rid])
                for rid in page.related
                if rid in mod_by_id
            )
            or "(无)"
        )
        linkable = ", ".join(f"{p_.id}.md" for p_ in aligned if p_.id != page.id)[:800]
        spec = (
            f"书:{repo_cfg.name}\n章:{page.chapter_title or page.chapter_id}\n"
            f"页面 id:{page.id}(文件名,延伸阅读链接用 .md)\n"
            f"标题:{page.title}\n重点:{page.importance}\n"
            f"建议小节:{' ; '.join(page.sections) or '自行组织'}\n"
            f"可链接的同书页面:{linkable}"
        )
        extra = ""
        if page.chapter_id == "overview":
            extra = (
                "\n模块依赖关系(确定性生成,mermaid 可润色但不得删边):\n"
                f"```mermaid\n{module_import_graph(conn, repo_id, modules)}\n```\n"
            )
        prompt = (
            load_prompt("page_generation.v2")
            .replace("{page_spec}", spec)
            .replace("{neighbor_summaries}", neighbors)
            .replace("{packed_source}", packed)
            + extra
        )
        async with sem:
            md, metrics = await _generate_one(llm, page, prompt, conn, repo_id)
        stats.pages_generated += 1
        if metrics.citations_total and metrics.citations_ok == 0:
            stats.pages_failed += 1
        stats.citations_total += metrics.citations_total
        stats.citations_ok += metrics.citations_ok
        stats.citations_removed += metrics.citations_removed
        generated[page.id] = (to_file_links(md), metrics)

    await asyncio.gather(*[run_page(p_) for p_ in aligned])

    # 统一落盘:面包屑 + 上下页导航(确定性注入,不参与校验)
    for i, (page, path) in enumerate(page_targets):
        if page.id not in generated:
            continue
        body, metrics = generated[page.id]
        breadcrumb = f"[📖 目录]({_rel_link(wiki_dir / 'README.md', path)})"
        if page.chapter_id:
            breadcrumb += f" › {page.chapter_title or page.chapter_id}"
        nav = "\n\n---\n"
        if i > 0:
            prev = page_targets[i - 1]
            nav += f"⬅️ 上一页:[{prev[0].title or prev[0].id}]({_rel_link(prev[1], path)})"
        if i + 1 < len(page_targets):
            nxt = page_targets[i + 1]
            nav += (" · " if len(nav) > 5 else "") + (
                f"➡️ 下一页:[{nxt[0].title or nxt[0].id}]({_rel_link(nxt[1], path)})"
            )
        fm = {
            "repo": repo_cfg.name,
            "page_id": page.id,
            "title": page.title or page.id,
            "chapter": page.chapter_id,
            "module_id": page.chapter_id,
            "source_hash": mod_hash.get(page.chapter_id, ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
            "reviewed": "false",
        }
        fm.update({f"metrics.{k}": v for k, v in metrics.front_matter_dict().items()})
        write_page(path, fm, breadcrumb + "\n\n" + body + nav)

    # README 目录页(TOC)
    toc = [
        f"# {repo_cfg.name} · 代码导读",
        "",
        f"> codeatlas {GENERATOR_VERSION} 生成;行内引用可点击跳转源码,页脚可上下翻页。",
        "",
    ]
    for cid in chapter_order:
        ch_pages = [
            (p_, f_) for p_, f_ in page_targets if (p_.chapter_id or "misc") == cid
        ]
        if not ch_pages:
            continue
        first = ch_pages[0][0]
        toc.append(f"## {first.chapter_title or cid}")
        toc.append("")
        for p_, f_ in ch_pages:
            toc.append(
                f"- [{p_.title or p_.id}]({_rel_link(f_, wiki_dir / 'README.md')})"
            )
        toc.append("")
    (wiki_dir / "README.md").write_text("\n".join(toc), encoding="utf-8")

    mark_stale_pages(wiki_dir, mod_hash)
    stats.indexed_chunks = await _index_wiki(
        conn, lance, embedder, repo_id, wiki_dir, settings
    )
    return stats


async def _index_wiki(conn, lance, embedder, repo_id, wiki_dir: Path, settings) -> int:
    from codeatlas.db.fts import fts_delete

    old = conn.execute(
        "SELECT id, content FROM chunks WHERE repo_id=? AND kind='wiki'", (repo_id,)
    ).fetchall()
    for r in old:
        fts_delete(conn, r["id"], r["content"])
    conn.execute("DELETE FROM chunks WHERE repo_id=? AND kind='wiki'", (repo_id,))
    conn.commit()

    n = 0
    embed_pending = []
    for page in sorted(wiki_dir.glob("**/*.md")):
        if page.name.endswith(".suggested.md"):
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(text, settings.chunk_max_tokens)
        for ch in chunks:
            cur = conn.execute(
                "INSERT INTO chunks(repo_id, kind, title, content, content_hash, "
                "line_start, line_end) VALUES(?,?,?,?,?,?,?)",
                (repo_id, "wiki", f"{repo_id}:{page.stem}:{ch.title or ''}",
                 ch.content, ch.content_hash, ch.line_start, ch.line_end),
            )
            embed_pending.append((cur.lastrowid, ch.content))
            n += 1
    conn.commit()
    if embedder is not None and embed_pending:
        try:
            vectors = await embedder.embed(
                [c for _, c in embed_pending], stage="wiki", repo_id=repo_id
            )
            lance_rows = [
                (cid, repo_id, "wiki", blob)
                for (cid, _), blob in zip(embed_pending, vectors)
            ]
            for lance_id, chunk_id in lance.add_vectors(lance_rows):
                conn.execute(
                    "INSERT INTO vector_refs(chunk_id, lance_id, model) VALUES(?,?,?)",
                    (chunk_id, lance_id, embedder.s.embed_model),
                )
            conn.commit()
        except Exception:
            pass  # embedding 失败不阻塞 wiki(FTS 检索路径仍可用)
    return n
