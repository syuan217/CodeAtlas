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
def status(as_json: bool = typer.Option(False, "--json", help="输出 JSON(Agent 消费)")) -> None:
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
        if as_json:
            repos_rows = [dict(r) for r in conn.execute(
                r"SELECT name, indexed_commit, last_indexed_at FROM repos "
                r"WHERE name NOT LIKE '\_\_%' ESCAPE '\' ORDER BY id")]
            _emit_json({
                "db": str(DB_PATH), "schema_version": schema_version,
                "counts": counts, "repos": repos_rows,
                "total_cost": total_cost(conn),
            })
            conn.close()
            return

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
            r"SELECT name, indexed_commit, last_indexed_at FROM repos "
            r"WHERE name NOT LIKE '\_\_%' ESCAPE '\' ORDER BY id"
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
            [{"role": "user", "content": "ping"}], stage="doctor",
            max_tokens=256, thinking_disabled=True,
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
    no_embed: bool = typer.Option(
        False, "--no-embed",
        help="纯本地索引(FTS/符号/调用边,跳过向量;零 API 费用;后补 --full 可重嵌)",
    ),
    rebuild_calls: bool = typer.Option(
        False, "--rebuild-calls",
        help="索引后全量重算调用边(升级版本/解析规则变化后用;零 API 费用)",
    ),
    repair_vectors: bool = typer.Option(
        False, "--repair-vectors",
        help="索引后按缓存重建向量表(lancedb 损坏/误删时用;零 API 费用)",
    ),
) -> None:
    """索引 repos.yaml 中的仓库:遍历→符号→切块→FTS→(可选)embedding→LanceDB。"""
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
    embed_ready = bool(s.embed_base_url and s.embed_api_key and s.embed_model)
    if not embed_ready and not no_embed:
        console.print(
            "[red]Embedding 未配置:atlas index 需要配置 EMBED_*,或使用 --no-embed 纯本地索引[/red]"
        )
        raise typer.Exit(code=1)
    if no_embed:
        console.print("[yellow]--no-embed:纯本地索引,向量检索不可用(atlas search/ask 的 FTS 路正常)。[/yellow]")

    for r in repos:
        console.rule(f"index {r.name}")
        try:
            stats = index_repo(r, settings=s, full=full, no_embed=no_embed or not embed_ready)
        except (ProviderError, CostLimitExceeded) as e:
            console.print(f"[red]{r.name} 索引中断:{e}[/red]")
            raise typer.Exit(code=1)
        console.print(stats.summary_line())
        if stats.mode == "git" and stats.head_commit:
            console.print(f"[dim]indexed_commit → {stats.head_commit[:12]}[/dim]")

    import subprocess
    import sys

    def tail_cmd(args: list[str]) -> None:
        r = subprocess.run([sys.executable, "-m", "codeatlas", *args])
        if r.returncode != 0:
            raise typer.Exit(code=r.returncode)

    if rebuild_calls:
        console.rule("index · rebuild-calls")
        tail_cmd(["rebuild-calls"] + (["--repo", repo] if repo else []))
    if repair_vectors:
        console.rule("index · repair-vectors")
        tail_cmd(["repair-vectors"])


@app.command("repair-vectors", hidden=True)
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


def _emit_json(data) -> None:
    import json as _json

    console.print_json(_json.dumps(data, ensure_ascii=False, default=str))


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
            loc = r_['fpath'] or f"wiki/{r_['title'] or ''}"  # wiki chunk 无 file_id
            console.print(
                f"  [{loc}:{r_['line_start']}-{r_['line_end']}] "
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


def _pretty_qname(qname: str) -> str:
    """展示用 qname:去掉重载消歧后缀 @line(v2 验收 #8/#37/#48 记法问题)。"""
    import re

    return re.sub(r"@\d+$", "", qname)


def _print_call_edges(edges, title: str) -> None:
    t = Table(title=title)
    for col in ("调用方", "被调方", "resolution", "调用点"):
        t.add_column(col)
    for e in edges:
        t.add_row(
            _pretty_qname(e.src_qname), _pretty_qname(e.dst_qname), e.resolution or "-",
            f"{e.src_file}:{e.line}" if e.src_file and e.line else "-",
        )
    console.print(t)
    exact = sum(1 for e in edges if e.resolution == "exact")
    console.print(f"[dim]共 {len(edges)} 条(exact {exact} / heuristic {len(edges) - exact})[/dim]")


@app.command()
def definition(
    symbol: str = typer.Argument(..., help="符号名或 qualified_name"),
    repo: str = typer.Option(None, "--repo"),
    as_json: bool = typer.Option(False, "--json", help="输出 JSON(Agent 消费)"),
) -> None:
    """查看符号定义位置(文件:行)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        if as_json:
            _emit_json({k: sym[k] for k in
                        ("qualified_name", "kind", "fpath", "line_start", "line_end", "signature")})
            return
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
    as_json: bool = typer.Option(False, "--json", help="输出 JSON(Agent 消费)"),
) -> None:
    """谁调用了这个符号(沿 CALLS 入边)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        edges = callers_of(conn, sym)
        if as_json:
            _emit_json({"symbol": sym["qualified_name"],
                        "edges": [e.__dict__ for e in edges]})
            return
        _print_call_edges(edges, f"callers of {sym['qualified_name']}")
    finally:
        conn.close()


@app.command("callees")
def callees_cmd(
    symbol: str = typer.Argument(...),
    repo: str = typer.Option(None, "--repo"),
    as_json: bool = typer.Option(False, "--json", help="输出 JSON(Agent 消费)"),
) -> None:
    """这个符号调用了谁(沿 CALLS 出边)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        edges = callees_of(conn, sym)
        if as_json:
            _emit_json({"symbol": sym["qualified_name"],
                        "edges": [e.__dict__ for e in edges]})
            return
        _print_call_edges(edges, f"callees of {sym['qualified_name']}")
    finally:
        conn.close()


@app.command()
def impact(
    symbol: str = typer.Argument(...),
    repo: str = typer.Option(None, "--repo"),
    depth: int = typer.Option(10, "--depth", help="最大传播深度"),
    as_json: bool = typer.Option(False, "--json", help="输出 JSON(Agent 消费)"),
) -> None:
    """影响面:改这个符号会波及哪些方法(沿 CALLS 入边传播到不动点)。"""
    conn = connect()
    try:
        init_db(conn)
        sym, _ = _resolve_symbol_or_exit(conn, symbol, repo)
        rows = impact_of(conn, sym, max_depth=depth)
        if as_json:
            _emit_json({"symbol": sym["qualified_name"], "impacts": rows})
            return
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


@app.command("rebuild-calls", hidden=True)
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
def collect(
    schema: str = typer.Option(None, "--schema", help="只处理指定库(缺省=data/ddl 全部)"),
    import_sheet: Path = typer.Option(
        None, "--import", help="导入已填写的 fill_sheet.md(表规模画像),转为各库 *_manual.json"
    ),
    import_slow: Path = typer.Option(
        None, "--import-slow",
        help="导入慢查询文件(CSV/JSON/纯文本 SQL;可多次执行合并)",
    ),
) -> None:
    """画像采集(人工模式):填报表/只读脚本/慢查询文件三通道。"""
    from codeatlas.config import DDL_DIR, PROFILES_DIR
    from codeatlas.schema.collect_ob import (
        generate_fill_sheet,
        generate_scripts,
        import_fill_sheet,
        import_slow_queries,
        manual_template,
    )
    from codeatlas.schema.ddl import load_ddl_dir

    parsed_all = load_ddl_dir(DDL_DIR)
    if not parsed_all:
        console.print(f"[red]{DDL_DIR} 下没有 DDL 文件(.sql/.md)[/red]")
        raise typer.Exit(code=1)
    targets = {schema: parsed_all[schema]} if schema else parsed_all

    if import_slow is not None:
        if not import_slow.exists():
            console.print(f"[red]文件不存在:{import_slow}[/red]")
            raise typer.Exit(code=1)
        counts = import_slow_queries(
            import_slow, targets, out_dir=PROFILES_DIR, default_schema=schema
        )
        if counts:
            for s, n in counts.items():
                console.print(f"慢查询导入:[green]{s}[/green] {n} 条 → {PROFILES_DIR / (s + '_manual.json')}")
            console.print("[dim]重新运行 atlas audit 即生效(QRY101 + 访问路径证据)。[/dim]")
        else:
            console.print("[yellow]没有条目被归属到任何库:检查文件格式,或用 --schema 指定库名。[/yellow]")
        return

    if import_sheet is not None:
        written = import_fill_sheet(import_sheet, targets, out_dir=PROFILES_DIR)
        for w in written:
            console.print(f"已导入画像:[green]{w}[/green]")
        console.print("[dim]重新运行 atlas audit 即生效(IDX101 等统计规则)。[/dim]")
        return

    sheet = generate_fill_sheet(targets)
    console.print(f"汇总填报表(推荐人工填写):[green]{sheet}[/green]")
    scripts = generate_scripts(targets)
    templates = [manual_template(s, p_) for s, p_ in targets.items()]
    for s in scripts:
        console.print(f"采集脚本(可选,DBA 执行):[green]{s}[/green]")
    console.print(f"[dim]另有 JSON 模板 {len(templates)} 个(索引基数/列区分度等更细粒度可选)。[/dim]")
    console.print(
        "[dim]填完 fill_sheet.md 后:uv run atlas collect --import <该文件> → atlas audit。[/dim]"
    )


@app.command()
def audit(
    repos: str = typer.Option(None, "--repos", help="限定扫描的仓库,逗号分隔(缺省=repos.yaml 全部)"),
) -> None:
    """OceanBase 索引体检:DDL 解析 + 代码 SQL 访问路径 + 规则引擎 + 报告。"""
    from codeatlas.config import DDL_DIR, PROFILES_DIR, load_repos
    from codeatlas.ingest.indexer import upsert_repo
    from codeatlas.schema.audit import (
        persist_findings,
        render_report,
        run_audit,
    )
    from codeatlas.schema.collect_ob import apply_profile, load_manual
    from codeatlas.schema.ddl import load_ddl_dir, persist_ddl
    from codeatlas.schema.sql_extract import extract_repo, persist_query_map

    conn = connect()
    try:
        init_db(conn)
        parsed_all = load_ddl_dir(DDL_DIR)
        if not parsed_all:
            console.print(f"[red]{DDL_DIR} 下没有 DDL 文件[/red]")
            raise typer.Exit(code=1)

        # 1) DDL 入库
        for schema, p_ in parsed_all.items():
            persist_ddl(conn, p_)
        total_fail = sum(len(p_.failures) for p_ in parsed_all.values())
        console.print(
            f"DDL:{len(parsed_all)} 库 / {sum(len(p_.tables) for p_ in parsed_all.values())} 表"
            + (f"(解析失败 {total_fail} 段)" if total_fail else "")
        )

        # 2) 画像回填(存在即应用)
        for schema in parsed_all:
            mp = PROFILES_DIR / f"{schema}_manual.json"
            if mp.exists():
                try:
                    n = apply_profile(conn, schema, load_manual(mp))
                    console.print(f"画像回填:{schema}({n} 表)")
                except Exception as e:
                    console.print(f"[yellow]画像 {schema} 回填失败:{e}[/yellow]")

        # 3) 代码 SQL 提取(全部登记仓库;未涉及库的表自然不进规则)
        cfgs = load_repos()
        if repos:
            only = {r.strip() for r in repos.split(",")}
            cfgs = [c for c in cfgs if c.name in only]
        anti_patterns: list[dict] = []
        for cfg in cfgs:
            if not cfg.path.is_dir():
                continue
            repo_id = upsert_repo(conn, cfg)
            qs = extract_repo(cfg, cfg.path)
            persist_query_map(conn, repo_id, qs)
            for q in qs:
                for ap in q.anti_patterns:
                    anti_patterns.append({**ap, "fingerprint": q.fingerprint,
                                          "source_file": q.source_file,
                                          "source_line": q.source_line})
            console.print(f"SQL 提取:{cfg.name}({len(qs)} 条语句)")
        qmap = [
            dict(r)
            for r in conn.execute(
                "SELECT DISTINCT query_fingerprint, source_file, source_line, "
                "table_name, column_name, usage, freq FROM query_column_map"
            )
        ]

        # 3.5) 慢查询:入库表画像之外的事件数据(独立文件导入);SQL 的
        # where/join 列并入访问路径证据(线上真实查询,IDX001 最硬证据)
        from codeatlas.schema.collect_ob import load_manual
        from codeatlas.schema.sql_extract import (
            ExtractedQuery,
            _clean_dynamic,
            _extract_accesses,
            _try_parse,
            fingerprint_sql,
        )

        slow_by_schema: dict[str, list[dict]] = {}
        slow_extra_qmap: list[dict] = []
        for schema in parsed_all:
            mp = PROFILES_DIR / f"{schema}_manual.json"
            if not mp.exists():
                continue
            data = load_manual(mp)
            slow = [q for q in (data.get("slow_queries") or []) if q.get("sql")]
            slow_by_schema[schema] = slow
            for q in slow:
                sql = _clean_dynamic(str(q["sql"]))
                stmt = _try_parse(sql)
                if stmt is None:
                    continue
                tbls, accesses = _extract_accesses(stmt)
                if not tbls:
                    continue
                eq = ExtractedQuery(
                    source_file=f"slow_query:{schema}", source_line=0, sql=sql[:2000],
                    fingerprint=fingerprint_sql(sql), tables=tbls, accesses=accesses,
                )
                freq = int(q.get("freq") or 1)
                for a in eq.accesses:
                    slow_extra_qmap.append({
                        "query_fingerprint": eq.fingerprint,
                        "source_file": eq.source_file, "source_line": 0,
                        "table_name": a.table, "column_name": a.column,
                        "usage": a.usage, "freq": freq,
                    })
        if slow_extra_qmap:
            conn.execute(
                "INSERT INTO repos(name, path, languages) "
                "VALUES('__codeatlas_reports__', '', '[]') "
                "ON CONFLICT(name) DO UPDATE SET path=''"
            )
            sid = conn.execute(
                "SELECT id FROM repos WHERE name='__codeatlas_reports__'"
            ).fetchone()["id"]
            # key 含 schema(source_file 体现归属);同键取最大频次
            merged: dict[tuple, int] = {}
            for r in slow_extra_qmap:
                schema_k = r["source_file"].split(":", 1)[1]
                key = (schema_k, r["query_fingerprint"], r["table_name"],
                       r["column_name"], r["usage"])
                merged[key] = max(merged.get(key, 0), r["freq"])
            conn.execute(
                "DELETE FROM query_column_map WHERE repo_id=? AND source_file LIKE 'slow_query:%'",
                (sid,),
            )
            rows = [
                (sid, fp, f"slow_query:{schema_k}", 0, tbl, col, usage, freq)
                for (schema_k, fp, tbl, col, usage), freq in merged.items()
            ]
            conn.executemany(
                "INSERT INTO query_column_map(repo_id, query_fingerprint, source_file, "
                "source_line, table_name, column_name, usage, freq) VALUES(?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            console.print(f"慢查询证据:{sum(len(v) for v in slow_by_schema.values())} 条,"
                          f"访问路径 {len(merged)} 条并入")
        qmap += slow_extra_qmap

        # 4) 规则引擎 + 报告
        results = [run_audit(conn, schema, qmap, anti_patterns,
                             slow_queries=slow_by_schema.get(schema))
                   for schema in parsed_all]
        report = render_report(results)
        persist_findings(conn, results)
        # 报告入索引(kind=report;FTS+向量,供 atlas ask 引用)
        from codeatlas.db.lance import LanceStore as _LS
        from codeatlas.providers.embedding import EmbeddingProvider as _EP
        from codeatlas.schema.audit import index_report as _index_report

        _lance = _LS(get_settings())
        _embed = _EP(get_settings(), conn)

        async def _index_and_close():
            n = await _index_report(conn, _lance, _embed, report, get_settings())
            await _embed.aclose()
            return n

        n_rc = asyncio.run(_index_and_close())
        total = sum(len(r.findings) for r in results)
        console.print(
            f"\nfindings 共 [bold]{total}[/bold] 条;报告:[green]{report}[/green]"
            f"(入库 {n_rc} chunks,可被 atlas ask 引用)"
        )
        for r in results:
            console.print(
                f"  {r.schema}: {len(r.findings)} findings"
                + (f"(unavailable: {','.join(r.unavailable)})" if r.unavailable else "")
            )
    finally:
        conn.close()


async def _wiki_one(cfg, concise: bool, max_pages: int | None) -> bool:
    """生成单个仓库的 wiki;返回是否成功。"""
    from codeatlas.db.lance import LanceStore
    from codeatlas.gencode.wiki import generate_wiki

    s = get_settings()
    conn = connect()
    init_db(conn)
    repo_id = conn.execute(
        "SELECT id FROM repos WHERE name=?", (cfg.name,)
    ).fetchone()
    if repo_id is None or not conn.execute(
        "SELECT COUNT(*) AS c FROM files WHERE repo_id=? AND parse_status='ok'",
        (repo_id["id"],),
    ).fetchone()["c"]:
        console.print(f"[yellow]{cfg.name} 尚未索引,跳过(先 atlas index --repo {cfg.name})[/yellow]")
        conn.close()
        return False

    lance = LanceStore(s)
    llm = LLMProvider(s, conn)
    embedder = EmbeddingProvider(s, conn)
    try:
        with console.status(f"[bold]wiki 生成中:{cfg.name}…[/bold]"):
            stats = await generate_wiki(
                cfg, conn, llm, s, lance, embedder,
                concise=concise, max_pages=max_pages,
            )
        console.print(
            f"{cfg.name}:模块 {stats.modules} → 页面 {stats.pages_planned}"
            f"(成功 {stats.pages_generated},失败 {stats.pages_failed}) · "
            f"引用 {stats.citations_ok}/{stats.citations_total} 通过,"
            f"剔除 {stats.citations_removed} · 入库 {stats.indexed_chunks} chunks"
            f" · 输出 {stats.output_dir}"
        )
        for e in stats.errors:
            console.print(f"[yellow]{e}[/yellow]")
        return True
    except Exception as e:
        console.print(f"[red]{cfg.name} wiki 生成失败:{e}[/red]")
        return False
    finally:
        await llm.aclose()
        await embedder.aclose()
        conn.close()


async def _wiki(repo_name: str | None, concise: bool, max_pages: int | None, force: bool) -> None:
    from codeatlas.config import load_repos

    all_repos = load_repos()
    if repo_name:
        repos = [r for r in all_repos if r.name == repo_name]
        if not repos:
            console.print(f"[red]repos.yaml 里没有 {repo_name!r}[/red]")
            raise typer.Exit(code=1)
    else:
        repos = all_repos
        console.print(f"未指定 --repo,遍历 repos.yaml({len(repos)} 个)逐个生成")
    ok = 0
    for cfg in repos:
        if await _wiki_one(cfg, concise, max_pages):
            ok += 1
    console.print(f"\n完成:{ok}/{len(repos)} 个仓库。冷启动评审:阅读各页 front-matter 的 metrics。")


@app.command()
def wiki(
    repo: str = typer.Argument(None, help="仓库名(缺省 = repos.yaml 全部,已索引的逐个生成)"),
    concise: bool = typer.Option(False, "--concise", help="4~6 页精简模式(冷启动推荐)"),
    max_pages: int = typer.Option(None, "--max-pages", help="限制生成页数(冷启动试跑)"),
    force: bool = typer.Option(False, "--force", help="强制重生成(人工保护仍然生效)"),
) -> None:
    """生成仓库 wiki(结构规划 → 逐页 grounding → 五层校验 → 入索引)。"""
    asyncio.run(_wiki(repo, concise, max_pages, force))


@app.command("index-docs")
def index_docs_cmd() -> None:
    """data/docs 人工文档入库(kind=doc,重跑自动替换;之后 atlas ask 可引用)。"""
    from codeatlas.config import DOCS_DIR
    from codeatlas.db.lance import LanceStore
    from codeatlas.schema.audit import index_docs

    s = get_settings()
    conn = connect()
    init_db(conn)
    embedder = EmbeddingProvider(s, conn)
    lance = LanceStore(s)

    async def _run():
        n = await index_docs(conn, lance, embedder, DOCS_DIR, s)
        await embedder.aclose()
        return n

    try:
        n = asyncio.run(_run())
        console.print(f"docs 入库 {n} chunks(目录:{DOCS_DIR};README.md 已跳过)")
        if n == 0:
            console.print("[yellow]目录为空:把 markdown 文档放进去后再跑。[/yellow]")
    finally:
        conn.close()


async def _agent(question: str, repo_name: str | None, max_turns: int) -> None:
    from codeatlas.agent.loop import DEFAULT_MAX_TURNS, run_agent
    from codeatlas.agent.tools import ToolBox
    from codeatlas.db.lance import LanceStore
    from codeatlas.retrieve.search import retrieve

    s = get_settings()
    conn = connect()
    init_db(conn)
    repo_id = None
    repo_root = None
    if repo_name:
        row = conn.execute("SELECT id, path FROM repos WHERE name=?", (repo_name,)).fetchone()
        if row is None:
            console.print(f"[red]仓库 {repo_name!r} 未索引[/red]")
            conn.close()
            raise typer.Exit(code=1)
        repo_id, repo_root = row["id"], Path(row["path"])

    lance = LanceStore(s)
    embedder = EmbeddingProvider(s, conn)
    llm = LLMProvider(s, conn)

    async def _retrieve(query: str):
        return await retrieve(
            query, s, conn, embedder, lance, repo_id=repo_id
        )

    toolbox = ToolBox(conn, repo_root, repo_id, retriever=_retrieve)
    try:
        with console.status("[bold]agent 思考中…[/bold]"):
            result = await run_agent(
                question, llm, conn, toolbox, repo_name=repo_name,
                max_turns=max_turns or DEFAULT_MAX_TURNS,
            )
        console.print(Panel(result.answer, title="回答"))
        if result.tool_trace:
            console.print("[bold]工具调用轨迹:[/bold]")
            for i, c in enumerate(result.tool_trace, 1):
                args_short = str(c["args"])[:60]
                console.print(
                    f"  {i}. {c['tool']}({args_short}) → {c['result_chars']} 字符"
                )
        console.print(
            f"[dim]轮数 {result.turns} · 工具调用 {len(result.tool_trace)} 次 · "
            f"tokens {result.prompt_tokens}+{result.completion_tokens} · "
            f"费用 {result.cost:.6f} 元[/dim]"
        )
    finally:
        await embedder.aclose()
        await llm.aclose()
        conn.close()


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="关键词/标识符/自然语言"),
    repo: str = typer.Option(None, "--repo", help="限定仓库"),
    top_k: int = typer.Option(10, "-k", help="返回条数"),
    no_vector: bool = typer.Option(False, "--no-vector", help="跳过向量路(纯 FTS+符号)"),
    as_json: bool = typer.Option(False, "--json", help="输出 JSON(Agent 消费)"),
) -> None:
    """检索(零 LLM):返回相关代码/文档片段与符号位置,不生成回答。"""
    from codeatlas.db.lance import LanceStore
    from codeatlas.retrieve.search import retrieve

    s = get_settings()
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

        embedder = None
        lance = None
        if not no_vector and (s.embed_base_url and s.embed_api_key and s.embed_model):
            embedder = EmbeddingProvider(s, conn)
            lance = LanceStore(s)

        async def _run():
            try:
                return await retrieve(query, s, conn, embedder, lance, repo_id=repo_id)
            finally:
                if embedder is not None:
                    await embedder.aclose()

        cands = asyncio.run(_run())
        if as_json:
            ph = ",".join("?" * min(len(cands), top_k))
            info = {
                r["id"]: r for r in conn.execute(
                    f"SELECT c.id, c.title, c.line_start, c.line_end, c.kind, "
                    f"       f.path AS fpath FROM chunks c "
                    f"LEFT JOIN files f ON c.file_id=f.id WHERE c.id IN ({ph})",
                    [c.chunk_id for c in cands[:top_k]],
                )
            }
            out = []
            for i, cand in enumerate(cands[:top_k], 1):
                r = info.get(cand.chunk_id)
                if r is None:
                    continue
                out.append({
                    "rank": i,
                    "path": r["fpath"] or f"wiki:{r['title']}",
                    "line_start": r["line_start"], "line_end": r["line_end"],
                    "title": r["title"], "kind": r["kind"],
                    "score": round(cand.score, 4),
                    "vec_sim": round(cand.vec_sim, 4) if cand.vec_sim is not None else None,
                    "sources": sorted(cand.sources),
                })
            _emit_json({"query": query, "total": len(cands), "results": out})
            return
        if not cands:
            console.print("[yellow]无命中。[/yellow]")
            return
        ph = ",".join("?" * min(len(cands), top_k))
        info = {
            r["id"]: r for r in conn.execute(
                f"SELECT c.id, c.title, c.line_start, c.line_end, c.kind, "
                f"       f.path AS fpath FROM chunks c "
                f"LEFT JOIN files f ON c.file_id=f.id WHERE c.id IN ({ph})",
                [c.chunk_id for c in cands[:top_k]],
            )
        }
        table = Table(title=f"检索: {query}")
        for col in ("#", "位置", "标题/符号", "来源", "得分"):
            table.add_column(col)
        for i, cand in enumerate(cands[:top_k], 1):
            r = info.get(cand.chunk_id)
            if r is None:
                continue
            loc = r["fpath"] or f"wiki:{r['title']}"
            table.add_row(
                str(i), f"{loc}:{r['line_start']}-{r['line_end']}",
                (r["title"] or "")[:60], r["kind"],
                f"{cand.score:.4f}" + (f"/{cand.vec_sim:.2f}" if cand.vec_sim else ""),
            )
        console.print(table)
        console.print(
            f"[dim]召回 {len(cands)} 条,显示 {min(top_k, len(cands))};"
            f"{'向量+FTS+符号' if embedder else 'FTS+符号(无向量配置或 --no-vector)'}[/dim]"
        )
    finally:
        conn.close()


@app.command()
def agent(
    question: str = typer.Argument(..., help="问题(支持多跳分析)"),
    repo: str = typer.Option(None, "--repo", help="限定仓库"),
    max_turns: int = typer.Option(6, "--max-turns", help="工具调用轮数上限"),
) -> None:
    """agent 问答:LLM 自主调用检索/调用链/读文件工具,多轮收集证据后作答。"""
    asyncio.run(_agent(question, repo, max_turns))


@app.command()
def update(
    concise: bool = typer.Option(False, "--concise", help="wiki 用 4~6 页精简模式"),
    no_wiki: bool = typer.Option(False, "--no-wiki", help="只做 index,不生成 wiki"),
) -> None:
    """代码更新后的一条龙:index(全部仓库增量)→ wiki(全部已索引仓库)。

    串联执行(index 成功才跑 wiki);两者写同一库,不可并行,本命令内部顺序处理。
    """
    import subprocess
    import sys

    def run_stage(args: list[str]) -> None:
        r = subprocess.run([sys.executable, "-m", "codeatlas", *args])
        if r.returncode != 0:
            raise typer.Exit(code=r.returncode)

    console.rule("update · index")
    run_stage(["index"])
    if no_wiki:
        console.print("[dim]--no-wiki:跳过 wiki。[/dim]")
        return
    console.rule("update · wiki")
    run_stage(["wiki"] + (["--concise"] if concise else []))


@app.command()
def doctor() -> None:
    """连通性体检:两个端点各一次最小调用,打印模型/维度/费用。"""
    asyncio.run(_doctor())


if __name__ == "__main__":
    app()
