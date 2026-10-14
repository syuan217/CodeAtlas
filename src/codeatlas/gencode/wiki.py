"""wiki 生成编排 v2(zread 风格):章节书结构 + 叙述页 + 导航注入 + 入索引。

结构:wiki/<repo>/README.md(目录页)+ <NN>-<chapter>/<page>.md;
每页确定性注入面包屑(› 目录/› 章)与上一页/下一页;源码行内引用
后处理为 file:// 绝对链接(本地 markdown 阅读器可点击跳转)。
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath

from codeatlas import config
from codeatlas.config import Settings
from codeatlas.gencode.freshness import is_human_owned, mark_stale_pages, write_page
from codeatlas.gencode.generate import (
    GENERATOR_VERSION,
    Page,
    load_prompt,
)
from codeatlas.gencode.pack import DEFAULT_BUDGET_TOKENS, interface_summary, pack_files
from codeatlas.gencode.segment import Module, _agg_hash, segment
from codeatlas.gencode.validate import PageMetrics, validate_page
from codeatlas.ingest.chunker import chunk_markdown
from codeatlas.ingest.indexer import upsert_repo
from codeatlas.providers.embedding import EmbeddingProvider
from codeatlas.providers._retry import CostLimitExceeded, ProviderError
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


def _safe_slug(text: str, fallback: str) -> str:
    """LLM 给的 id/章节名 → 单段安全文件名(保留中英文、数字、-、_)。"""
    slug = re.sub(r"[^\w\-]+", "-", text or "").strip("-")
    return slug[:80] or fallback


def _files_hash(conn: sqlite3.Connection, repo_id: int, paths: list[str]) -> str:
    """页面源文件的聚合内容哈希(freshness 判据;文件被删时自然变化)。"""
    if not paths:
        return ""
    ph = ",".join("?" * len(paths))
    found = {
        r["path"]: r["hash"] or ""
        for r in conn.execute(
            f"SELECT path, hash FROM files WHERE repo_id=? AND path IN ({ph})",
            [repo_id, *paths],
        )
    }
    return _agg_hash([f"{p}:{found.get(p, '-')}" for p in paths])


def _rel_link(target: PurePath, from_file: PurePath) -> str:
    """目标页相对当前文件的 markdown 链接路径(始终 '/' 分隔)。

    按路径对象计算(不拼字符串):posixpath.relpath 喂 Windows 路径会得到 ../C:/…。
    """
    return target.relative_to(from_file.parent, walk_up=True).as_posix()


def _file_uri(path: PurePath) -> str:
    """源码跳转链接:Windows 上是 file:///C:/…,手拼 f"file://{path}" 会得到无效的 file://C:\…。"""
    if not path.is_absolute():
        path = Path(path).resolve()
    return path.as_uri()


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
    # 页面 id 来自 LLM,用作文件名前必须收敛到安全字符(防 ../ 与绝对路径写出 wiki 目录)并去重
    seen_ids: set[str] = set()
    for n, p_ in enumerate(aligned, 1):
        pid = _safe_slug(p_.id, f"page-{n}")
        while pid in seen_ids:
            pid = f"{pid}-{n}"
        seen_ids.add(pid)
        p_.id = pid
    stats.pages_planned = len(aligned)

    wiki_dir = config.WIKI_DIR / repo_cfg.name  # 动态读,便于测试重定向
    wiki_dir.mkdir(parents=True, exist_ok=True)
    stats.output_dir = str(wiki_dir)

    chapter_order: list[str] = []
    for p_ in aligned:
        cid = p_.chapter_id or "misc"
        if cid not in chapter_order:
            chapter_order.append(cid)
    chapter_dirname = {
        cid: f"{i + 1:02d}-{_safe_slug(cid, 'misc')}"
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
            return f"({_file_uri(target)})" if target.exists() else m.group(0)

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

    # 单页失败只记错、不连坐整书;费用超限则撤掉其余页,已生成的照常落盘后再上抛
    halt: list[CostLimitExceeded] = []
    tasks: list[asyncio.Future] = []

    async def guarded(page: Page) -> None:
        try:
            await run_page(page)
        except CostLimitExceeded as e:
            if not halt:
                halt.append(e)
                me = asyncio.current_task()
                for t in tasks:
                    if t is not me:
                        t.cancel()
        except ProviderError as e:
            stats.pages_failed += 1
            stats.errors.append(f"页面 {page.id} 生成失败(已有旧版则保留):{e}")

    tasks.extend(asyncio.ensure_future(guarded(p_)) for p_ in aligned)
    for r in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
            raise r

    # 导航/目录只指向实际存在的页(本次生成的 + 生成失败但保留的旧版)
    live = [(p_, f_) for p_, f_ in page_targets if p_.id in generated or f_.exists()]

    # 统一落盘:面包屑 + 上下页导航(确定性注入,不参与校验)
    for i, (page, path) in enumerate(live):
        if page.id not in generated:
            continue
        body, metrics = generated[page.id]
        breadcrumb = f"[📖 目录]({_rel_link(wiki_dir / 'README.md', path)})"
        if page.chapter_id:
            breadcrumb += f" › {page.chapter_title or page.chapter_id}"
        nav = "\n\n---\n"
        if i > 0:
            prev = live[i - 1]
            nav += f"⬅️ 上一页:[{prev[0].title or prev[0].id}]({_rel_link(prev[1], path)})"
        if i + 1 < len(live):
            nxt = live[i + 1]
            nav += (" · " if len(nav) > 5 else "") + (
                f"➡️ 下一页:[{nxt[0].title or nxt[0].id}]({_rel_link(nxt[1], path)})"
            )
        fm = {
            "repo": repo_cfg.name,
            "page_id": page.id,
            "title": page.title or page.id,
            "chapter": page.chapter_id,
            # 过期判据按本页实际引用的源文件(chapter_id 是 LLM 起的 slug,不是模块 id)
            "source_files": ",".join(page.file_paths),
            "source_hash": _files_hash(conn, repo_id, page.file_paths),
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
        ch_pages = [(p_, f_) for p_, f_ in live if (p_.chapter_id or "misc") == cid]
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

    if not max_pages:  # 试跑(--max-pages)不清理,免得删掉完整版的其余页
        _remove_orphan_pages(wiki_dir, {f_ for _, f_ in page_targets})
    mark_stale_pages(wiki_dir, lambda files: _files_hash(conn, repo_id, files))
    stats.indexed_chunks = await _index_wiki(
        conn, lance, embedder, repo_id, wiki_dir, settings
    )
    if halt:
        raise halt[0]
    return stats


def _remove_orphan_pages(wiki_dir: Path, keep: set[Path]) -> None:
    """删除旧规划遗留、且未被人改过的页(不在本次目录里的生成页会继续被检索到);
    人工修改/非生成的页保留。"""
    for page in sorted(wiki_dir.rglob("*.md")):
        if page.name == "README.md" or page.name.endswith(".suggested.md") or page in keep:
            continue
        if is_human_owned(page):
            continue
        page.unlink()
        suggested = page.with_suffix(".suggested.md")
        if suggested.exists():
            suggested.unlink()
    for d in sorted(wiki_dir.iterdir()):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


async def _index_wiki(conn, lance, embedder, repo_id, wiki_dir: Path, settings) -> int:
    from codeatlas.db.chunkstore import embed_chunks, insert_text_chunk, purge_chunks

    purge_chunks(conn, lance, repo_id=repo_id, kind="wiki")
    conn.commit()

    embed_pending = []
    for page in sorted(wiki_dir.glob("**/*.md")):
        if page.name.endswith(".suggested.md"):
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        for ch in chunk_markdown(text, settings.chunk_max_tokens):
            cid = insert_text_chunk(
                conn, repo_id=repo_id, kind="wiki",
                title=f"{repo_id}:{page.stem}:{ch.title or ''}", chunk=ch,
            )
            embed_pending.append((cid, ch.content))
    conn.commit()
    try:
        await embed_chunks(conn, lance, embedder, embed_pending,
                           repo_id=repo_id, kind="wiki", stage="wiki")
    except Exception:
        pass  # embedding 失败不阻塞 wiki(FTS 检索路径仍可用)
    return len(embed_pending)
