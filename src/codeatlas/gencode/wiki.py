"""wiki 生成编排(PLAN §9.7):segment → 规划 → 逐页生成+校验 → 落盘 → 入索引。"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from codeatlas.config import Settings, WIKI_DIR
from codeatlas.gencode.freshness import mark_stale_pages, write_page
from codeatlas.gencode.generate import (
    GENERATOR_VERSION,
    Page,
    generate_page,
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
                parts.append(f"[{name}]\n" + p.read_text(encoding="utf-8",
                                                         errors="replace")[:2000])
            except OSError:
                pass
    return "\n\n".join(parts) or "(无 README/CONTEXT)"


def _modules_text(modules: list[Module]) -> str:
    lines = []
    for m in modules:
        preview = ", ".join(m.files[:6]) + ("…" if len(m.files) > 6 else "")
        lines.append(f"- {m.id}({len(m.files)} 文件):{preview}")
    return "\n".join(lines)


def module_import_graph(conn: sqlite3.Connection, repo_id: int, modules: list[Module]) -> str:
    """模块间 IMPORTS 边 → mermaid 骨架(overview 页的确定性底图)。"""
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
        lines.append(f'  {a.replace("/", "_")}["{a}"] -->|{c} 依赖| {b.replace("/", "_")}["{b}"]')
    return "\n".join(lines)


async def _generate_one(
    llm: LLMProvider,
    page: Page,
    prompt: str,
    conn: sqlite3.Connection,
    repo_id: int,
) -> tuple[str, PageMetrics]:
    """单页:生成 → 硬校验 → 失败带错误清单重试(≤2)→ 仍失败用已剔除版本。"""
    last_md, metrics = "", PageMetrics()
    feedback = ""
    for attempt in range(MAX_VALIDATE_RETRIES + 1):
        p = prompt + (f"\n\n上一版硬校验失败清单(逐条修复后重新输出完整页面):\n{feedback}"
                      if feedback else "")
        md = await generate_page(llm, p)
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

    size_hint = "4~6 页(concise 模式)" if concise else "8~12 页(comprehensive 模式)"
    from codeatlas.gencode.generate import plan_pages

    pages = await plan_pages(
        llm, _repo_summary(repo_root), _modules_text(modules), "", size_hint
    )
    # 页面与模块对齐:overview 或已存在模块 id;filePaths 限定在模块文件清单内
    all_files = {f for m in modules for f in m.files}
    mod_by_id = {m.id: m for m in modules}
    aligned: list[Page] = []
    for p in pages:
        if p.id != "overview" and p.id not in mod_by_id:
            continue
        p.file_paths = [f for f in p.file_paths if f in all_files]
        if p.id != "overview" and not p.file_paths:
            p.file_paths = mod_by_id[p.id].files[:10]
        aligned.append(p)
    if aligned and aligned[0].id != "overview":
        aligned.insert(0, Page(id="overview", title="架构总览", importance="high"))
    # 对齐兜底:规划不配合(输出 id 与模块不符)时,按文件数补足前几个大模块
    min_pages = min(4, len(modules))
    if len(aligned) < min_pages:
        covered = {p_.id for p_ in aligned}
        for m in sorted(modules, key=lambda m: -len(m.files)):
            if len(aligned) >= max(min_pages, 3):
                break
            if m.id in covered:
                continue
            aligned.append(Page(id=m.id, title=m.id.rsplit("/", 1)[-1],
                                importance="normal"))
        stats.errors.append(
            f"结构规划仅对齐 {len([p_ for p_ in aligned if p_.id != 'overview'])} 页,"
            f"已按模块规模兜底补足"
        )
    if max_pages:
        aligned = aligned[:max_pages]
    stats.pages_planned = len(aligned)

    wiki_dir = WIKI_DIR / repo_cfg.name
    wiki_dir.mkdir(parents=True, exist_ok=True)
    stats.output_dir = str(wiki_dir)
    mod_hash = {m.id: m.source_hash for m in modules}

    sem = asyncio.Semaphore(max(1, concurrency))

    async def run_page(page: Page) -> None:
        files = page.file_paths or (
            mod_by_id[page.id].files[:10] if page.id in mod_by_id else []
        )
        packed, used = pack_files(conn, repo_id, repo_root, files, budget_tokens)
        neighbors = "\n\n".join(
            interface_summary(conn, repo_id, mod_by_id[rid])
            for rid in page.related if rid in mod_by_id
        ) or "(无)"
        extra = ""
        if page.id == "overview":
            extra = (
                "\n模块依赖关系(确定性生成,mermaid 骨架,可润色但不得删边):\n"
                f"```mermaid\n{module_import_graph(conn, repo_id, modules)}\n```\n"
            )
        prompt = (
            load_prompt("page_generation.v1")
            .replace("{page_spec}",
                     f"id={page.id}\ntitle={page.title}\n"
                     f"sections={' ; '.join(page.sections)}\nimportance={page.importance}")
            .replace("{neighbor_summaries}", neighbors)
            .replace("{packed_source}", packed)
            + extra
        )
        async with sem:
            md, metrics = await _generate_one(llm, page, prompt, conn, repo_id)
        stats.pages_generated += 1
        if metrics.citations_removed and metrics.citations_ok == 0:
            stats.pages_failed += 1
        stats.citations_total += metrics.citations_total
        stats.citations_ok += metrics.citations_ok
        stats.citations_removed += metrics.citations_removed
        fm = {
            "repo": repo_cfg.name,
            "page_id": page.id,
            "title": page.title or page.id,
            "module_id": page.id,
            "source_hash": mod_hash.get(page.id, ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
            "reviewed": "false",
        }
        fm.update({f"metrics.{k}": v for k, v in metrics.front_matter_dict().items()})
        target = write_page(wiki_dir / f"{page.id}.md", fm, md)

    await asyncio.gather(*[run_page(p) for p in aligned])

    # 其余既有页面的 stale 标记
    mark_stale_pages(wiki_dir, mod_hash)

    # wiki 入索引(FTS 必做;embedding 有 provider 则做)
    stats.indexed_chunks = await _index_wiki(
        conn, lance, embedder, repo_id, wiki_dir, settings
    )
    return stats


async def _index_wiki(conn, lance, embedder, repo_id, wiki_dir: Path, settings) -> int:
    from codeatlas.db.fts import fts_delete

    # 清旧 wiki chunks(外部内容 FTS 表须带原 content 发 delete 命令)
    old = conn.execute(
        "SELECT id, content FROM chunks WHERE repo_id=? AND kind='wiki'", (repo_id,)
    ).fetchall()
    for r in old:
        fts_delete(conn, r["id"], r["content"])
    conn.execute(
        "DELETE FROM chunks WHERE repo_id=? AND kind='wiki'", (repo_id,)
    )
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
