"""agent 工具循环(M6):LLM ↔ 工具,上限 N 轮,耗尽强制综答。

防护(PLAN §4 code-graph-rag 结论):全部只读工具、结果 32k 截断、
轮数上限、费用护栏复用 COST_LIMIT_PER_RUN。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from codeatlas.agent.tools import TOOL_SPECS, ToolBox, approx_tokens
from codeatlas.providers.llm import LLMProvider

DEFAULT_MAX_TURNS = 6
MAX_OUTPUT_TOKENS = 8192
_ELIDED = "(早前的工具结果已省略以控制上下文;如仍需要,请重新调用工具)"


def _fit_context(messages: list[dict], budget: int) -> None:
    """总量超预算时,从最早的工具结果起逐条替换为省略说明(原地修改工作副本);
    system / 问题 / 会话记忆不动。"""
    def size(m: dict) -> int:
        return approx_tokens(m.get("content") or "") + approx_tokens(
            json.dumps(m.get("tool_calls") or "", ensure_ascii=False))

    total = sum(size(m) for m in messages)
    for m in messages:
        if total <= budget:
            return
        if m["role"] == "tool" and m["content"] != _ELIDED:
            total -= size(m)
            m["content"] = _ELIDED
            total += size(m)

AGENT_SYSTEM_PROMPT = (
    "你是代码库分析 agent,通过调用工具收集证据后回答问题。准则:\n"
    "1. 先用 search 定位,再用 definition/callers/callees/impact 深入结构关系,"
    "必要时 read_file 核实细节;每次工具调用应有明确目的,不重复同样查询;\n"
    "2. 证据足够时立即停止调用工具,直接输出最终回答;\n"
    "3. 回答必须基于工具返回的证据,引用格式 [相对路径:起-止行];"
    "证据不足的部分明确说明,禁止编造;\n"
    "4. 工具结果被截断时,用更小的范围(如更精确的 query/更窄的行区间)重查。"
)


@dataclass
class AgentResult:
    answer: str
    turns: int = 0
    tool_trace: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    # 会话记忆:system + 历次问答对(不含工具过程消息),供下一轮 run_agent 复用
    messages: list[dict] = field(default_factory=list)


async def run_agent(
    question: str,
    llm: LLMProvider,
    conn: sqlite3.Connection,
    toolbox: ToolBox,
    repo_name: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    history: list[dict] | None = None,
) -> AgentResult:
    result = AgentResult(answer="")
    # 记忆跨问题延续;工具调用消息只在本次提问的工作副本里流转
    memory = list(history) if history else [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]
    user_msg = {
        "role": "user",
        "content": question + (
            f"\n(分析范围限定仓库:{repo_name})" if repo_name else ""
        ),
    }
    messages: list[dict] = memory + [user_msg]
    # 输入预算 = 模型窗口 - 输出预留;工具定义本身也占输入,按其 JSON 体量扣除
    budget = (llm.s.llm_context_window - MAX_OUTPUT_TOKENS
              - approx_tokens(json.dumps(TOOL_SPECS, ensure_ascii=False)))
    final = ""
    for turn in range(1, max_turns + 1):
        result.turns = turn
        _fit_context(messages, budget)
        r = await llm.chat(
            messages, stage="agent", temperature=0.2, max_tokens=MAX_OUTPUT_TOKENS,
            thinking_disabled=True, tools=TOOL_SPECS,
        )
        result.prompt_tokens += r.prompt_tokens
        result.completion_tokens += r.completion_tokens
        result.cost += r.cost
        if not r.tool_calls:
            final = r.content
            break
        # 回传 assistant 的 tool_calls,再逐个执行回填
        messages.append(
            {
                "role": "assistant",
                "content": r.content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in r.tool_calls
                ],
            }
        )
        for tc in r.tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            output = await toolbox.execute(tc["name"], args)
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": output}
            )
    else:
        # 轮数耗尽:强制综合作答
        messages.append(
            {
                "role": "user",
                "content": "已达到工具调用上限。请基于已获取的信息立即作答,"
                "信息不足之处明确列出。",
            }
        )
        _fit_context(messages, budget)
        r = await llm.chat(
            messages, stage="agent", temperature=0.2, max_tokens=MAX_OUTPUT_TOKENS,
            thinking_disabled=True,
        )
        result.prompt_tokens += r.prompt_tokens
        result.completion_tokens += r.completion_tokens
        result.cost += r.cost
        final = r.content

    result.answer = final
    result.tool_trace = toolbox.calls
    result.messages = memory + [user_msg, {"role": "assistant", "content": final}]
    return result
