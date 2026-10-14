---
name: atlas
description: Use when the user wants to locate, verify, or check code in a codebase indexed by codeatlas — 查找代码在哪、谁在用 X、改 X 影响哪里、X 在哪定义、确认"是不是/有没有 XX"、这块代码现在怎么实现、某机制/流程如何工作(综合问答)、代码导读. Prefer over grep on indexed repos; exact-line citations; may combine multiple commands and cross-check results.
---

只读查询本地已索引代码库(生成/维护类命令如 wiki/index/update 不在本 skill 范围)。已索引仓库见 `~/.codeatlas/repos.yaml`(或 `atlas status --json`)。

## 命令选择(按问题类型)

| 问题 | 命令 |
|---|---|
| 找某功能/概念的代码位置 | `atlas search "<关键词>" --json -k 8` |
| 谁调用了 X | `atlas callers <符号> --json` |
| X 调用了什么 | `atlas callees <符号> --json` |
| X 的定义(签名/位置) | `atlas definition <符号> --json` |
| 改 X 会影响哪里 | `atlas impact <符号> --json` |
| 代码导读 | 读 `~/.codeatlas/data/wiki/<repo>/README.md`(已生成产物,零费用) |
| 综合叙述/多跳问题 | `atlas ask "<问题>"`(一次性;LLM 工具循环多轮取证后综答,带 文件:行号 引用) |

加 `--repo <name>` 限定单仓库,减少符号歧义。

## 组合与交叉比对(不限于单命令)

- 按需串联多个命令,常见链路:`search` 定位 → `definition` 消歧/看签名 → `callers` / `callees` / `impact` 展开结构关系;
- 交叉比对:atlas 查询给候选位置,再用文件阅读核实关键行;多个命令的结果之间、或与文件内容矛盾时,以文件实际内容为准;
- 口语化的"确认一下 / 是不是 / 有没有 / 现在怎么样"类问题:先定位再读源码下结论,不要只凭检索摘要断言。

## 规则

1. 结构查询命令(search/callers/callees/definition/impact)一律加 `--json` 输出;`ask` 无此选项;
2. callers/callees/impact/definition 是确定性图查询,不经过 LLM;引用
   `resolution: heuristic` 的边时注明置信度,`exact` 可直接陈述;
3. 符号多匹配时命令以非零退出并返回 `{"error": "ambiguous", "candidates": [...]}`,
   用其中的完整限定名(如 `com.x.S#create`,或 `Class#method` 短写)或 `--repo` 重试;
   拿不准短名先 `definition` 消歧;
4. 回答中的代码引用用 `[相对路径:起始行-结束行]`,字段直接取自 JSON 的
   `path`/`line_start`/`line_end`;
5. 只对 `indexed_commit` 非空的仓库用 atlas,其余仓库直接读文件;
6. 索引可能滞后于工作区:定位与引用用 atlas,最终结论以文件实际内容为准。
