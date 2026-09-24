---
name: atlas
description: Use when the user asks about a local codebase indexed by codeatlas (语义问答/谁调用了X/改X影响哪里/调用链/代码检索/影响面/代码导读). Prefer atlas commands over grep for structural questions — callers/callees/impact are deterministic graph queries with exact-line citations.
---

# atlas — 本地代码知识库查询

atlas 是命令行工具(无后台服务,直接执行)。数据来自已索引的本地仓库
(配置:`~/.codeatlas/repos.yaml`)。**所有查询命令支持 `--json` 输出,优先用它。**

## 命令选择(按问题类型)

| 用户问题类型 | 命令 |
|---|---|
| 找某功能/概念的代码位置 | `atlas search "<关键词或自然语言>" --json -k 8` |
| "谁调用了 X" / "X 被哪里引用" | `atlas callers <符号名> --json` |
| "X 调用了什么" | `atlas callees <符号名> --json` |
| X 的定义在哪(签名/位置) | `atlas definition <符号名> --json` |
| "改 X 会影响哪里" / 重构风险评估 | `atlas impact <符号名> --json` |
| 需要综合叙述的代码问题 | `atlas ask "<问题>" [--repo <名>]`(LLM 作答,带 文件:行号 引用) |
| 多跳分析(链路/对比/跨模块) | `atlas agent "<问题>"`(LLM 多轮调用上述工具) |

仓库范围:加 `--repo <name>` 限定单仓库(推荐,减少歧义);不加则查全部已索引仓库。

## 使用要点

1. **符号歧义**:callers/callees/impact 遇到多匹配会报错并列出候选——换用列出的
   完整限定名(如 `com.x.OrderService#create`)重试;先用 `definition` 查短名也是
   消歧的好办法;
2. **resolution 字段**:`exact`=精确解析(100% 可信),`heuristic`=唯一同名兜底
   (可信但建议抽查);引用调用边结论时注明置信度;
3. **引用格式**:回答中引用代码统一用 `[相对路径:起始行-结束行]`;
   search 结果的 `path`/`line_start`/`line_end` 字段直接拼装即可;
4. **需要看具体代码**时,用文件的绝对路径前缀 `<仓库根>/<相对路径>`
   (仓库根见 `~/.codeatlas/repos.yaml` 或 `atlas status --json` 后再读文件);
5. wiki(生成的代码导读)在 `<DATA_DIR>/wiki/<repo>/` 下,`search`/`ask` 已能检索到;
6. 索引可能滞后于工作区:用户刚改的文件未必在索引里,结论以文件实际内容为准,
   索引结果作定位与引用辅助。

## 环境自检

- `atlas status --json`:确认哪些仓库已索引(indexed_commit 非空);
- 未索引的仓库不要用 atlas 查,直接读文件;
- 所有命令幂等只读(除 index/audit 等写命令,本 skill 不涉及)。
