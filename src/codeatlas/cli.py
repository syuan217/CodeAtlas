"""atlas CLI 入口(typer)。M0:doctor / status / cost。"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codeatlas import __version__
from codeatlas.config import (
    DB_PATH,
    LANCEDB_DIR,
    REPOS_YAML,
    ensure_dirs,
    get_settings,
    load_repos,
)
from codeatlas.cost import cost_summary, model_summary, total_cost
from codeatlas.db.models import connect, init_db
from codeatlas.graph.queries import (
    call_edges_stats,
    callees_of,
    callers_of,
    find_symbols,
    impact_of,
    sample_call_edges,
)
from codeatlas.ingest.indexer import index_repo
from codeatlas.providers._retry import CostLimitExceeded, ProviderError
from codeatlas.providers.embedding import EmbeddingProvider, unpack_vector
from codeatlas.providers.llm import LLMProvider
from codeatlas.retrieve.context import build_context
from codeatlas.retrieve.rerank import maybe_rerank
from codeatlas.retrieve.search import retrieve

app = typer.Typer(help="codeatlas · 本地代码知识库", no_args_is_help=True)
console = Console()


@app.command()
def status() -> None:
    """库状态:schema 版本、repos、各类对象计数、数据目录。"""
    ensure_dirs()
    conn = connect()
    try:
        init_db(conn)
        repos = conn.execute("SELECT COUNT(*) AS c FROM repos").fetchone()["c"]
        counts = {}
        for table in ("files", "symbols", "edges", "chunks"):
            counts[table] = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        schema_version = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()["value"]

        table_out = Table(title="codeatlas status", show_header=False)
        table_out.add_column(style="bold cyan")
        table_out.add_column()
        table_out.add_row("数据库", str(DB_PATH))
        table_out.add_row("schema_version", schema_version)
        table_out.add_row("repos(已注册)", str(repos))
        for k, v in counts.items():
            table_out.add_row(k, str(v))
        table_out.add_row("向量库目录", str(LANCEDB_DIR))
        table_out.add_row("仓库清单", str(REPOS_YAML) + f"({len(load_repos())} 条)")
        table_out.add_row("累计费用", f"{total_cost(conn):.4f} 元")
        console.print(table_out)

        repo_rows = conn.execute(
            "SELECT name, indexed_commit, last_indexed_at FROM repos ORDER BY id"
        ).fetchall()
        if repo_rows:
            rt = Table(title="repos")
            for col in ("name", "indexed_commit", "last_indexed_at"):
                rt.add_column(col)
            for r in repo_rows:
                rt.add_row(r["name"], r["indexed_commit"] or "-", r["last_indexed_at"] or "-")
            console.print(rt)
    finally:
        conn.close()


@app.command()
def cost(
    by_model: bool = typer.Option(False, "--by-model", help="按模型而非 stage 汇总"),
) -> None:
    """费用统计(usage_log 汇总)。"""
    conn = connect()
    try:
        init_db(conn)
        rows = model_summary(conn) if by_model else cost_summary(conn)
        t = Table(title="费用统计")
        key = "model" if by_model else "stage"
        for col in (key, "calls", "prompt_tokens", "completion_tokens", "cost(元)"):
            t.add_column(col, justify="right" if col != key else "left")
        for r in rows:
            t.add_row(
                r[key], str(r["calls"]), str(r["prompt_tokens"]),
                str(r["completion_tokens"]), f"{r['cost']:.6f}",
            )
        console.print(t)
        if not rows:
            console.print("[dim]尚无费用流水(usage_log 为空)[/dim]")
        else:
            console.print(f"合计:{total_cost(conn):.6f} 元")
    finally:
        conn.close()


async def _doctor() -> None:
    s = get_settings()

    env_table = Table(title="环境", show_header=False)
    env_table.add_column(style="bold cyan")
    env_table.add_column()
    env_table.add_row("codeatlas", f"v{__version__}")
    env_table.add_row("数据库", str(DB_PATH))
    conn = connect()
    init_db(conn)

    llm_configured = bool(s.llm_base_url and s.llm_api_key and s.llm_model)
    embed_configured = bool(s.embed_base_url and s.embed_api_key and s.embed_model)
    if not (llm_configured and embed_configured):
        env_table.add_row("LLM 配置", "[green]OK[/green]" if llm_configured else "[red]缺失[/red]")
        env_table.add_row("Embedding 配置", "[green]OK[/green]" if embed_configured else "[red]缺失[/red]")
        console.print(env_table)
        console.print(
            Panel(
                "服务商未配置。请 `cp .env.example .env` 并填写 LLM/EMBED 三项"
                "(BASE_URL / API_KEY / MODEL),再运行 atlas doctor。"
            )
        )
        conn.close()
        raise typer.Exit(code=1)

    # ---- LLM 端点最小调用 ----
    t0 = time.perf_counter()
    llm = LLMProvider(s, conn)
    try:
        r = await llm.chat(
            [{"role": "user", "content": "ping"}], stage="doctor", max_tokens=8
        )
        llm_ms = (time.perf_counter() - t0) * 1000
        llm_line = (
            f"[green]OK[/green]  模型 {r.model}  耗时 {llm_ms:.0f}ms  "
            f"tokens {r.prompt_tokens}+{r.completion_tokens}  "
            f"费用 {r.cost:.6f} 元" + ("" if r.priced else "  [yellow](单价未知,cost=0)[/yellow]")
        )
    except (ProviderError, CostLimitExceeded) as e:
        llm_line = f"[red]FAIL[/red]  {e}"
    finally:
        await llm.aclose()

    # ---- Embedding 端点最小调用 ----
    t0 = time.perf_counter()
    emb = EmbeddingProvider(s, conn)
    try:
        blobs = await emb.embed(["ping"], input_type="query", stage="doctor")
        emb_ms = (time.perf_counter() - t0) * 1000
        dim = len(blobs[0]) // 4
        vec0 = unpack_vector(blobs[0])
        finite = all(x == x and abs(x) != float("inf") for x in vec0)
        emb_cost = conn.execute(
            "SELECT COALESCE(SUM(cost),0) AS c FROM usage_log WHERE stage='doctor' AND model=?",
            (s.embed_model,),
        ).fetchone()["c"]
        emb_line = (
            f"[green]OK[/green]  模型 {s.embed_model}  维度 {dim}"
            f"(EMBED_DIM={s.embed_dim} {'一致' if dim == s.embed_dim else '[red]不一致[/red]'})  "
            f"耗时 {emb_ms:.0f}ms  数值 {'正常' if finite else '[red]含非有限值[/red]'}  "
            f"累计费用 {emb_cost:.6f} 元"
        )
    except (ProviderError, CostLimitExceeded) as e:
        emb_line = f"[red]FAIL[/red]  {e}"
    finally:
        await emb.aclose()

    ok = llm_line.startswith("[green]") and emb_line.startswith("[green]")
    env_table.add_row("LLM 端点", llm_line)
    env_table.add_row("Embedding 端点", emb_line)
    console.print(env_table)
    console.print(f"本次 doctor 最小调用费用合计:{total_cost(conn):.6f} 元")
    conn.close()
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def index(
    repo: str = typer.Option(None, "--repo", help="只索引指定仓库(缺省 = repos.yaml 全部)"),
    full: bool = typer.Option(False, "--full", help="忽略 git 基线,强制全量重解析"),
) -> None:
    """索引 repos.yaml 中的仓库:遍历→符号→切块→FTS→embedding→LanceDB。"""
    repos = load_repos()
    if not repos:
        console.print("[yellow]repos.yaml 里没有仓库;请编辑 repos.yaml 填入本地路径[/yellow]")
        raise typer.Exit(code=1)
    if repo:
        repos = [r for r in repos if r.name == repo]
        if not repos:
            console.print(f"[red]repos.yaml 里没有名为 {repo!r} 的仓库[/red]")
            raise typer.Exit(code=1)
    missing = [str(r.path) for r in repos if not r.path.is_dir()]
    if missing:
        console.print(f"[red]仓库路径不存在:{', '.join(missing)}[/red]")
        raise typer.Exit(code=1)

    s = get_settings()
    if not (s.embed_base_url and s.embed_api_key and s.embed_model):
        console.print(
            "[red]Embedding 未配置:atlas index 需要 .env 里的 EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL[/red]"
        )
        raise typer.Exit(code=1)

    for r in repos:
        console.rule(f"index {r.name}")
        try:
            stats = index_repo(r, settings=s, full=full)
        except (ProviderError, CostLimitExceeded) as e:
            console.print(f"[red]{r.name} 索引中断:{e}[/red]")
            raise typer.Exit(code=1)
        console.print(stats.summary_line())
        if stats.mode == "git" and stats.head_commit:
            console.print(f"[dim]indexed_commit → {stats.head_commit[:12]}[/dim]")


@app.command()
def repair_vectors() -> None:
    """一次性修复向量表:按 vector_refs + embed_cache 重写,消除重复/孤儿行。"""
    conn = connect()
    try:
        init_db(conn)
        s = get_settings()
        from codeatlas.db.lance import LanceStore

        store = LanceStore(s)
        before = store.count()
        rebuilt, missing = store.rebuild_from_sql(conn)
        conn.commit()
        store.maybe_create_index()
        console.print(
            f"向量表重建完成:{before} 行(含重复/孤儿)→ {rebuilt} 行有效;"
            f"缺缓存向量 {missing} 条(下次 atlas index 会自动重嵌)"
        )
    finally:
        conn.close()


QA_SYSTEM_PROMPT = (
    "你是代码库问答助手。只基于 <context> 标签内提供的代码片段回答问题;"
    "引用代码时使用格式 [相对路径:起始行-结束行](行号必须来自片段开头的 [lines A-B] 标注,"
    "不得自行推算);上下文不足以回答时明确说\"知识库中未找到\",禁止编造。"
    "回答使用与问题相同的语言。"
)


async def _ask(question: str, repo_name: str | None) -> None:
    s = get_settings()
    conn = connect()
    init_db(conn)
    repo_id = None
    if repo_name:
        row = conn.execute("SELECT id FROM repos WHERE name=?", (repo_name,)).fetchone()
        if row is None:
            console.print(f"[red]仓库 {repo_name!r} 未索引(先运行 atlas index --repo {repo_name})[/red]")
            conn.close()
            raise typer.Exit(code=1)
        repo_id = row["id"]

    from codeatlas.db.lance import LanceStore

    lance = LanceStore(s)
    embedder = EmbeddingProvider(s, conn)
    llm = LLMProvider(s, conn)
    try:
        cands = await retrieve(question, s, conn, embedder, lance, repo_id=repo_id)
        corpus = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        cands = maybe_rerank(question, cands, s, corpus)
        if not cands:
            console.print("[yellow]知识库中未找到相关内容(候选为空)。[/yellow]")
            return
        context, used = build_context(
            conn, cands, top_n=s.context_top_n,
            budget_tokens=s.llm_context_window - 2048,
        )
        if not context:
            console.print("[yellow]知识库中未找到相关内容(上下文组装为空)。[/yellow]")
            return

        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"<context>\n{context}\n</context>\n\n问题:{question}"},
        ]
        r = await llm.chat(messages, stage="ask", repo_id=repo_id)
        console.print(Panel(r.content, title="回答"))
        # 引用清单(来自实际装填的候选)
        console.print("[bold]引用来源(检索命中,供核对):[/bold]")
        ph = ",".join("?" * len(used))
        rows = conn.execute(
            f"SELECT c.id AS cid, c.title, c.line_start, c.line_end, f.path AS fpath "
            f"FROM chunks c LEFT JOIN files f ON c.file_id=f.id WHERE c.id IN ({ph})",
            [c.chunk_id for c in used],
        ).fetchall()
        by_id = {r_["cid"]: r_ for r_ in rows}
        for cand in used:
            r_ = by_id.get(cand.chunk_id)
            if r_ is None:
                continue
            sim = f" sim={cand.vec_sim:.2f}" if cand.vec_sim is not None else ""
            console.print(
                f"  [{r_['fpath']}:{r_['line_start']}-{r_['line_end']}] "
                f"{r_['title'] or ''}{sim} [dim]({'+'.join(sorted(cand.sources))})[/dim]"
            )
        price_note = "" if r.priced else "  [yellow](单价未知,cost=0)[/yellow]"
        console.print(
            f"[dim]召回 {len(cands)} 条 → 装填 {len(used)} 条;"
            f"本次费用 {r.cost:.6f} 元(tokens {r.prompt_tokens}+{r.completion_tokens})"
            f"{price_note}[/dim]"
        )
    finally:
        await embedder.aclose()
        await llm.aclose()
        conn.close()


@app.command()
def ask(
    question: str = typer.Argument(..., help="自然语言问题"),
    repo: str = typer.Option(None, "--repo", help="限定检索的仓库(缺省=全部)"),
) -> None:
    """语义问答:融合检索 → 上下文组装 → LLM 单轮作答(带 文件:行号 引用)。"""
    asyncio.run(_ask(question, repo))


def _resolve_symbol_or_exit(conn, pattern: str, repo: str | None):
    repo_id = None
    if repo:
        row = conn.execute("SELECT id FROM repos WHERE name=?", (repo,)).fetchone()
        if row is None:
            console.print(f"[red]仓库 {repo!r} 未索引[/red]")
            raise typer.Exit(code=1)
        repo_id = row["id"]
    syms = find_symbols(conn, pattern, repo_id)
    if not syms:
        console.print(f"[yellow]未找到符号:{pattern}[/yellow]")
        raise typer.Exit(code=1)
    if len(syms) > 1 and syms[0]["name"] != pattern and syms[0]["qualified_name"] != pattern:
        console.print(f"[yellow]{pattern!r} 匹配到 {len(syms)} 个符号,请用完整 qualified_name:[/yellow]")
        for s in syms[:15]:
            console.print(f"  {s['qualified_name']}  ({s['fpath']}:{s['line_start']})")
        raise typer.Exit(code=1)
    return syms[0], repo_id


def _print_call_edges(edges, title: str) -> None:
    t = Table(title=title)
    for col in ("调用方", "被调方", "resolution", "调用点"):
        t.add_column(col)
    for e in edges:
        t.add_row(
            e.src_qname, e.dst_qname, e.resolution or "-",
            f"{e.src_file}:{e.line}" if e.src_file and e.line else "-",
        )
    console.print(t)
    exact = sum(1 for e in edges if e.resolution == "exact")
    console.print(f"[dim]共 {len(edges)} 条(exact {exact} / heuristic {len(edges) - exact})[/dim]")


@app.command()
def definition(
    symbol: str = typer.Argument(..., help="符号名或 qualified_name"),
    repo: str = typer.Option(None, "--repo"),
) -> None:
    """查看符号定义位置(文件:行)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        console.print(
            f"[green]{sym['qualified_name']}[/green]  {sym['kind']}\n"
            f"  位置:{sym['fpath']}:{sym['line_start']}-{sym['line_end']}\n"
            f"  签名:{sym['signature'] or '-'}"
        )
    finally:
        conn.close()


@app.command("callers")
def callers_cmd(
    symbol: str = typer.Argument(...),
    repo: str = typer.Option(None, "--repo"),
) -> None:
    """谁调用了这个符号(沿 CALLS 入边)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        _print_call_edges(callers_of(conn, sym), f"callers of {sym['qualified_name']}")
    finally:
        conn.close()


@app.command("callees")
def callees_cmd(
    symbol: str = typer.Argument(...),
    repo: str = typer.Option(None, "--repo"),
) -> None:
    """这个符号调用了谁(沿 CALLS 出边)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        _print_call_edges(callees_of(conn, sym), f"callees of {sym['qualified_name']}")
    finally:
        conn.close()


@app.command()
def impact(
    symbol: str = typer.Argument(...),
    repo: str = typer.Option(None, "--repo"),
    depth: int = typer.Option(10, "--depth", help="最大传播深度"),
) -> None:
    """影响面:改这个符号会波及哪些方法(沿 CALLS 入边传播到不动点)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        rows = impact_of(conn, sym, max_depth=depth)
        t = Table(title=f"impact of {sym['qualified_name']}")
        for col in ("深度", "受影响符号", "文件"):
            t.add_column(col)
        for r in rows:
            t.add_row(str(r["depth"]), r["qname"], r["file"])
        console.print(t)
        files = {r["file"] for r in rows}
        console.print(
            f"[dim]共 {len(rows)} 个方法 / {len(files)} 个文件受影响;"
            f"depth=1 为直接调用方[/dim]"
        )
    finally:
        conn.close()


@app.command("calls-sample")
def calls_sample(
    n: int = typer.Option(30, "-n", help="抽样条数"),
    resolution: str = typer.Option(None, "--resolution", help="exact / heuristic"),
    repo: str = typer.Option(None, "--repo"),
) -> None:
    """M3 验收抽样:随机取 CALLS 边供人工核对。"""
    conn = connect()
    try:
        init_db(conn)
        repo_id = None
        if repo:
            row = conn.execute("SELECT id FROM repos WHERE name=?", (repo,)).fetchone()
            if row is None:
                console.print(f"[red]仓库 {repo!r} 未索引[/red]")
                raise typer.Exit(code=1)
            repo_id = row["id"]
        console.print(f"全库 CALLS 边统计:{call_edges_stats(conn, repo_id)}")
        edges = sample_call_edges(conn, n=n, resolution=resolution, repo_id=repo_id)
        _print_call_edges(edges, f"随机抽样 {len(edges)} 条(resolution={resolution or '全部'})")
        console.print(
            "[dim]核对方法:打开调用点文件(调用点列),确认该行确实调用了被调方;"
            "再打开被调方定义确认语义正确。[/dim]"
        )
    finally:
        conn.close()


@app.command("rebuild-calls")
def rebuild_calls(
    repo: str = typer.Option(None, "--repo", help="只处理指定仓库(缺省=全部已索引)"),
) -> None:
    """重算全库 CALLS 边(不动 chunks/向量;M3 补建或规则升级后使用)。"""
    from codeatlas.config import RepoCfg as _RepoCfg
    from codeatlas.graph.call_resolver.base import build_context
    from codeatlas.graph.call_resolver.orchestrate import (
        light_call_file,
        resolve_and_write_calls,
    )

    conn = connect()
    try:
        init_db(conn)
        rows = conn.execute(
            "SELECT r.id, r.name, r.path, f.path AS rel FROM repos r "
            "JOIN files f ON f.repo_id = r.id "
            "WHERE f.parse_status = 'ok'" + (" AND r.name = ?" if repo else ""),
            [repo] if repo else [],
        ).fetchall()
        if repo and not rows:
            console.print(f"[red]仓库 {repo!r} 未索引或没有可解析文件[/red]")
            raise typer.Exit(code=1)
        by_repo: dict[int, dict] = {}
        for r in rows:
            by_repo.setdefault(
                r["id"], {"name": r["name"], "path": r["path"], "rels": []}
            )["rels"].append(r["rel"])
        for repo_id, info in by_repo.items():
            cfg = _RepoCfg(name=info["name"], path=Path(info["path"]))
            with console.status(f"[bold]{info['name']}[/bold] 重算 CALLS 边…"):
                ctx = build_context(conn, repo_id)
                call_files = []
                for rel in info["rels"]:
                    cf = light_call_file(conn, cfg, rel)
                    if cf is not None:
                        call_files.append(cf)
                stats = resolve_and_write_calls(conn, repo_id, call_files, ctx)
            console.print(
                f"{info['name']}: 调用点 {stats.sites} → exact {stats.exact} / "
                f"heuristic {stats.heuristic} / 丢弃 {stats.dropped}"
                f"(写入 {stats.edges_written} 条)"
            )
    finally:
        conn.close()


@app.command()
def doctor() -> None:
    """连通性体检:两个端点各一次最小调用,打印模型/维度/费用。"""
    asyncio.run(_doctor())


if __name__ == "__main__":
    app()
