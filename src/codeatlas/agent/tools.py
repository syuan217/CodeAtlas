"""agent 工具集(M6):包装现成查询函数为 OpenAI function calling 工具。

六个工具(用户拍板):search / definition / callers / callees / impact /
read_file。全部只读;read_file 的路径必须 ∈ 索引文件白名单(防任意读);
每个工具结果做 token 截断(默认 32k)。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codeatlas.ingest.chunker import estimate_tokens

TOOL_RESULT_BUDGET = 32_000
READ_FILE_MAX_LINES = 500

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "语义+关键词+符号融合检索代码库/wiki/报告。用于:找概念对应的代码、"
            "定位某功能的实现位置。返回候选列表(路径:行号+标题+得分)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言或关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "definition",
            "description": "查符号(类/方法/接口)的定义位置与签名。输入符号名或完整限定名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "符号名,如 validate 或 com.x.S#m"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "callers",
            "description": "谁调用了这个符号(直接调用方,含调用点行号与 exact/heuristic 置信)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "callees",
            "description": "这个符号调用了谁。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "impact",
            "description": "影响面:修改该符号会波及哪些方法/文件(沿调用链向上传播)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "depth": {"type": "integer", "description": "最大传播深度,默认 10"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取索引内源码文件的指定行区间(用于核实细节;一次最多 500 行)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "仓库内相对路径"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
]


def _truncate(text: str, budget: int = TOOL_RESULT_BUDGET) -> str:
    if estimate_tokens(text) <= budget:
        return text
    # 按行截断(保守:估算 token ~ len/3)
    max_chars = budget * 3
    out = text[:max_chars]
    cut = out.rfind("\n")
    return (out[:cut] if cut > 0 else out) + f"\n…(截断,完整结果 {len(text)} 字符)"


class ToolBox:
    """工具执行器:持有库连接与检索依赖,方法与 TOOL_SPECS 一一对应。"""

    def __init__(self, conn: sqlite3.Connection, repo_root: Path | None,
                 repo_id: int | None, retriever=None):
        self.conn = conn
        self.repo_root = repo_root
        self.repo_id = repo_id
        self._retriever = retriever  # async retrieve(query, ...) 可调用
        self.calls: list[dict] = []  # 轨迹

    async def execute(self, name: str, args: dict) -> str:
        try:
            if name == "search":
                result = await self._search(args["query"])
            elif name == "definition":
                result = self._definition(args["symbol"])
            elif name == "callers":
                result = self._call_edges(args["symbol"], "callers")
            elif name == "callees":
                result = self._call_edges(args["symbol"], "callees")
            elif name == "impact":
                result = self._impact(args["symbol"], args.get("depth") or 10)
            elif name == "read_file":
                result = self._read_file(args["path"], args.get("start_line") or 1,
                                         args.get("end_line") or 200)
            else:
                result = f"未知工具:{name}"
        except Exception as e:
            result = f"工具执行出错:{e}"
        self.calls.append({"tool": name, "args": args, "result_chars": len(result)})
        return _truncate(result)

    # ---- 各工具实现 ----

    async def _search(self, query: str) -> str:
        if self._retriever is None:
            return "检索器不可用(未配置 embedding)"
        cands = await self._retriever(query)
        if not cands:
            return "无候选"
        rows = []
        ph = ",".join("?" * min(len(cands), 20))
        info = {
            r["id"]: r for r in self.conn.execute(
                f"SELECT c.id, c.title, c.line_start, c.line_end, f.path AS fpath "
                f"FROM chunks c LEFT JOIN files f ON c.file_id=f.id WHERE c.id IN ({ph})",
                [c.chunk_id for c in cands[:20]],
            )
        }
        for rank, cand in enumerate(cands[:20], 1):
            r = info.get(cand.chunk_id)
            if r is None:
                continue
            loc = r["fpath"] or (r["title"] or "")
            rows.append(
                f"{rank}. [{loc}:{r['line_start']}-{r['line_end']}] {r['title'] or ''} "
                f"(score={cand.score:.4f}, src={'+'.join(sorted(cand.sources))})"
            )
        return "\n".join(rows)

    def _definition(self, symbol: str) -> str:
        from codeatlas.graph.queries import find_symbols

        syms = find_symbols(self.conn, symbol, self.repo_id)
        if not syms:
            return f"未找到符号:{symbol}"
        if len(syms) > 1:
            head = f"匹配 {len(syms)} 个(前 10):"
            lines = [
                f"- {s['qualified_name']}  {s['fpath']}:{s['line_start']}  {s['kind']}"
                for s in syms[:10]
            ]
            return head + "\n" + "\n".join(lines)
        s = syms[0]
        return (
            f"{s['qualified_name']}  [{s['kind']}]\n"
            f"位置:{s['fpath']}:{s['line_start']}-{s['line_end']}\n"
            f"签名:{s['signature'] or '-'}"
        )

    def _call_edges(self, symbol: str, direction: str) -> str:
        from codeatlas.graph.queries import callees_of, callers_of, find_symbols

        syms = find_symbols(self.conn, symbol, self.repo_id)
        if not syms:
            return f"未找到符号:{symbol}"
        if len(syms) > 1:
            return f"符号歧义({len(syms)} 个),请用完整限定名,如 {syms[0]['qualified_name']}"
        edges = (callers_of if direction == "callers" else callees_of)(self.conn, syms[0])
        if not edges:
            return f"无{direction} 边"
        lines = [
            f"- {e.src_qname} → {e.dst_qname}  [{e.resolution}] @ {e.src_file}:{e.line}"
            for e in edges[:50]
        ]
        return f"{len(edges)} 条(显示 {len(lines)}):\n" + "\n".join(lines)

    def _impact(self, symbol: str, depth: int) -> str:
        from codeatlas.graph.queries import find_symbols, impact_of

        syms = find_symbols(self.conn, symbol, self.repo_id)
        if not syms:
            return f"未找到符号:{symbol}"
        if len(syms) > 1:
            return f"符号歧义({len(syms)} 个),请用完整限定名,如 {syms[0]['qualified_name']}"
        rows = impact_of(self.conn, syms[0], max_depth=int(depth))
        if not rows:
            return "无受影响方法"
        files = sorted({r["file"] for r in rows})
        lines = [
            f"- [d{r['depth']}] {r['qname']}  ({r['file']})" for r in rows[:60]
        ]
        return (
            f"受影响:{len(rows)} 方法 / {len(files)} 文件(显示 {len(lines)}):\n"
            + "\n".join(lines)
            + "\n文件清单:" + ", ".join(files[:20])
        )

    def _read_file(self, path: str, start: int, end: int) -> str:
        row = self.conn.execute(
            "SELECT f.path FROM files f WHERE f.path=? "
            + ("AND f.repo_id=?" if self.repo_id is not None else ""),
            [path] + ([self.repo_id] if self.repo_id is not None else []),
        ).fetchone()
        if row is None:
            return f"路径不在索引内:{path}(只允许读已索引文件)"
        real = (self.repo_root / path) if self.repo_root else Path(path)
        try:
            lines = real.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return f"读取失败:{path}"
        start = max(1, int(start))
        end = min(int(end), start + READ_FILE_MAX_LINES - 1, len(lines))
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
        return f"{path} [lines {start}-{end}/{len(lines)}]\n{body}"
