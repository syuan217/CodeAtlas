# codeatlas

**本地代码知识库 RAG**:把你本地的多个代码仓库(Java 后端 + JS/TS 前端等)变成一个可问答、可查询、可生成文档的知识库。全部解析与检索在本地完成,LLM/Embedding 走 OpenAI 兼容 API。

## 它能做什么

| 能力 | 命令 | 说明 |
|---|---|---|
| **语义问答** | `atlas ask "问题"` | 带着真实的 `文件:行号` 引用回答代码问题;代码、wiki、体检报告、人工文档一起参与检索 |
| **代码检索** | `atlas search "关键词"` | 零 LLM 的融合检索(向量+全文+符号),直接输出 文件:行号 命中列表;无任何 API 配置也可用 |
| **结构查询** | `atlas callers / callees / definition / impact` | 确定性调用链查询(谁调用了 X / 改 X 影响哪里),不经过 LLM,结果 100% 可信 |
| **Agent 问答** | `atlas agent "问题"` | LLM 自主调用检索/调用链/读文件工具,多轮收集证据后作答(适合多跳问题) |
| **wiki 文档** | `atlas wiki <repo>` | 生成 zread 风格的代码导读(章节目录/面包屑/上下页导航,行内引用可点击跳转源码) |
| **OB 索引体检** | `atlas collect / audit` | 解析 OceanBase DDL + 提取代码中的 SQL 访问路径 + 慢查询证据,输出索引合理性报告(无主键/冗余索引/高频条件列无索引/扫描返回比异常等 10 条规则) |
| **运维** | `atlas status / cost / doctor / repair-vectors` | 库状态、费用流水、端点体检、向量表修复 |

工作原理:tree-sitter 提取符号与调用边(每条边带 exact/heuristic 置信标签)→ 按符号切块(真实行号)→ LanceDB 向量 + SQLite FTS5 全文双路召回 → RRF 融合 → 图扩展(调用/导入关系)→ 上下文组装作答。全部数据本地存储(SQLite + LanceDB 文件目录)。

## 安装

前提:[uv](https://docs.astral.sh/uv/getting-started/installation/)(一行安装的 Python 工具链)。

### 方式一:源码(开发者)

```bash
git clone <repo> && cd codeatlas
uv sync                     # 自动锁 Python 3.12 + 全部依赖
uv tool install --editable .   # 可选:获得全局 atlas 命令(否则用 uv run atlas …)
```

### 方式二:离线安装包(内网,零网络,免源码)

```bash
# 构建侧:产物为单 zip(含主包 + 全部依赖 wheel + INSTALL.md)
./scripts/make-bundle.sh
# 多平台:PLATFORMS="macosx_11_0_arm64 manylinux_2_28_x86_64" ./scripts/make-bundle.sh

# 使用侧:解压 zip 后
uv tool install --offline --find-links ./wheels codeatlas
```

## 快速开始

两种起步姿势任选:

- **零门槛试用**(不配任何 API,本地检索/调用链即可用):跳过下面第 1 步,直接
  `atlas index --no-embed` → `atlas search "关键词"` / `atlas callers <符号>`;
  之后想用问答,再补 `.env` 配置并 `atlas index --full` 补嵌向量(数据全复用);
- **完整能力**(问答/wiki):按下面 1-4 步。

```bash
# 1) 配置:首次运行任意命令会在 ~/.codeatlas 生成模板
atlas status
cp ~/.codeatlas/.env.example ~/.codeatlas/.env
# 编辑 .env:填 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 与 EMBED_* 三项(OpenAI 兼容端点)

# 2) 仓库清单
mv ~/.codeatlas/repos.example.yaml ~/.codeatlas/repos.yaml
# 编辑:登记要索引的本地仓库(name/path/languages/exclude)

# 3) 体检连通(打印模型名/维度校验/费用)
atlas doctor

# 4) 索引 → 使用
atlas index                      # 首次索引(增量;可中断重跑续传)
atlas ask "登录是怎么实现的" --repo my-service
atlas callers "userService" --repo my-service
atlas agent "改 X 会影响哪些上游?"
atlas wiki my-service            # 生成代码导读(data/wiki/<repo>/README.md 进入)
```

## 命令总览

**索引与数据**

| 命令 | 说明 |
|---|---|
| `atlas index [--repo N] [--full] [--no-embed]` | 索引 repos.yaml 仓库(遍历→符号→切块→FTS→向量);`--full` 全量重解析;`--no-embed` 纯本地索引(零 API 费用) |
| `atlas rebuild-calls [--repo N]` | 重算全库调用边(不动向量,零嵌入费) |
| `atlas repair-vectors` | 修复向量表(消重复/孤儿行) |
| `atlas index-docs` | `~/.codeatlas/data/docs/` 人工文档入库(ask 可引用) |

**问答与结构查询**

| 命令 | 说明 |
|---|---|
| `atlas ask "问题" [--repo N]` | 单轮融合检索问答(向量+FTS+符号+图扩展) |
| `atlas search "关键词" [--repo N] [-k 10] [--no-vector]` | 零 LLM 检索:直接输出 文件:行号 命中列表 |
| `atlas agent "问题" [--repo N] [--max-turns 6]` | 工具循环问答(检索/调用链/读文件,多跳问题) |
| `atlas definition / callers / callees <符号>` | 符号定义 / 调用方 / 被调方 |
| `atlas impact <符号> [--depth N]` | 影响面(沿调用链向上传播) |
| `atlas calls-sample [-n 30]` | 随机抽样调用边供人工核对 |

**OceanBase 体检**

| 命令 | 说明 |
|---|---|
| `atlas collect` | 生成表规模填报表(fill_sheet.md)与只读采集脚本 |
| `atlas collect --import <fill_sheet.md>` | 导入表规模画像 |
| `atlas collect --import-slow <文件> [--schema 库]` | 导入慢查询(CSV/JSON/纯文本 SQL 三格式) |
| `atlas audit [--repos a,b]` | 体检:DDL + 代码 SQL 访问路径 + 慢查询证据 → 报告(自动入库可被 ask 引用) |

**wiki / 运维**

| 命令 | 说明 |
|---|---|
| `atlas wiki <repo> [--concise] [--max-pages N]` | 生成章节式代码导读;人工改过的页永不被覆盖 |
| `atlas status / cost [--by-model] / doctor` | 库状态 / 费用 / 端点体检 |

## 配置与数据

**目录解析**(按优先级):环境变量 `CODEATLAS_HOME` > 当前目录(存在 `.env` 或 `repos.yaml` 时,即源码检出场景)> `~/.codeatlas`。所有配置、仓库清单与数据(kb.sqlite / lancedb / wiki / docs / profiles / reports / ddl)都跟随该目录;每人独立实例。`DATA_DIR` 可单独指定数据位置。

**.env 关键项**(完整模板见 `.env.example` / 包内模板):

```bash
LLM_BASE_URL / LLM_API_KEY / LLM_MODEL     # OpenAI 兼容对话端点
EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL / EMBED_DIM   # 向量端点(维度必须匹配,doctor 会校验)
MAX_CONCURRENCY=4                            # API 并发
COST_LIMIT_PER_RUN=50                        # 单次命令费用上限(元),超出自动中断
```

**费用**:所有 API 调用记流水(`atlas cost` 可查);嵌入按内容哈希缓存,重复索引不重复计费;换 embedding 模型需全量重嵌(`rm -rf <data>/lancedb && atlas index --full`)。

## 日常更新

```bash
atlas index               # 代码更新后:增量(只重解析变更文件+依赖闭包)
atlas wiki <repo>         # 代码更新后:过期页自动标记,重跑生成(人工改过的写 .suggested.md)
atlas index-docs          # 你自己的文档改了
atlas collect --import-slow <新慢查询文件> && atlas audit   # 体检材料更新
```

## 开发

```bash
uv sync && uv run pytest        # 178 个测试(网络层全部 mock)
```

- 设计与施工依据见 [PLAN.md](PLAN.md)(唯一施工文档);
- wiki prompt 版本化于 `src/codeatlas/gencode/prompts/`(改动须升版本);
- 里程碑:M0 脚手架 → M1 索引管线 → M2 融合检索 → M3 调用边(验收 100%)→ M4 OB 体检 → M5 wiki 生成 → M6 agent 问答。

## 约束与边界

- 单机单实例(SQLite 单写);百万行级仓库首索引约数十分钟,增量秒级;
- 调用边解析保守优先(宁缺毋错):Java/TS 仅输出 exact 边,python/go 保留唯一同名兜底(heuristic 标签);
- OB 体检的统计类规则依赖画像回填(fill_sheet)与慢查询导入,缺省自动标 unavailable 不猜测。
