---
name: atlas
description: Use when the user asks a structural question about a codebase indexed by codeatlas — 谁调用了 X、改 X 影响哪里、X 在哪定义、找某功能的实现、代码导读. Deterministic graph queries with exact-line citations; prefer over grep on indexed repos.
---

查询本地已索引代码库的 CLI。已索引仓库见 `~/.codeatlas/repos.yaml`(或 `atlas status --json`)。

## 命令选择(按问题类型)

| 问题 | 命令 |
|---|---|
| 找某功能/概念的代码位置 | `atlas search "<关键词>" --json -k 8` |
| 谁调用了 X | `atlas callers <符号> --json` |
| X 调用了什么 | `atlas callees <符号> --json` |
| X 的定义(签名/位置) | `atlas definition <符号> --json` |
| 改 X 会影响哪里 | `atlas impact <符号> --json` |
| 综合叙述/多跳问题 | `atlas ask "<问题>"` / `atlas agent "<问题>"`(LLM 作答,带 文件:行号 引用) |

加 `--repo <name>` 限定单仓库,减少符号歧义。

## 规则

1. 查询命令一律加 `--json` 输出;
2. callers/callees/impact/definition 是确定性图查询,不经过 LLM;引用
   `resolution: heuristic` 的边时注明置信度,`exact` 可直接陈述;
3. 符号多匹配报错时,用报错列出的完整限定名(如 `com.x.S#create`)重试;
   拿不准短名先 `definition` 消歧;
4. 回答中的代码引用用 `[相对路径:起始行-结束行]`,字段直接取自 JSON 的
   `path`/`line_start`/`line_end`;
5. 只对 `indexed_commit` 非空的仓库用 atlas,其余仓库直接读文件;
6. 索引可能滞后于工作区:定位与引用用 atlas,最终结论以文件实际内容为准。
