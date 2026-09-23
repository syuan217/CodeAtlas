#!/usr/bin/env bash
# 构建离线安装包:主 wheel + 全部依赖 wheel(目标机无需网络/源码)。
# 用法:
#   ./scripts/make-bundle.sh                 # 按当前平台(Intel mac)
#   PLATFORMS="macosx_11_0_arm64" ./scripts/make-bundle.sh
#   # 多平台示例:PLATFORMS="macosx_11_0_arm64 manylinux_2_28_x86_64"
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=dist-bundle
rm -rf "$OUT" && mkdir -p "$OUT/wheels"

echo "== 1/4 构建主包 wheel"
uv build --wheel -o "$OUT"

echo "== 2/4 导出依赖清单"
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o "$OUT/requirements.txt"

echo "== 3/4 下载依赖 wheel(平台:当前${PLATFORMS:+ + $PLATFORMS})"
# uv 暂无 download 子命令,用 pip3 按目标平台解析(only-binary 全走 wheel)
# 注意 --no-emit-project:requirements 含 -e . 会让 pip 把项目根打 sdist,
# 与输出目录 dist-bundle 互相嵌套时产生递归膨胀(事故记录 2026-09-23)
CUR_PLATFORM=$(python3 -c 'import sysconfig;print(sysconfig.get_platform())' | tr '.-' '__')
for plat in "$CUR_PLATFORM" ${PLATFORMS:-}; do
  python3 -m pip download -r "$OUT/requirements.txt" -d "$OUT/wheels" \
    --only-binary :all: --python-version 3.12 --platform "$plat" \
    || echo "warn: 平台 $plat 部分包不可用"
done

echo "== 4/4 写安装说明"
cat > "$OUT/INSTALL.md" << 'MD'
# codeatlas 离线安装

前提:[uv](https://docs.astral.sh/uv/)(单文件,可离线分发)。

```bash
# 安装(自动建隔离环境;wheels 目录即依赖源,无需网络)
uv tool install --find-links ./wheels codeatlas --offline

# 升级:换新包目录后
uv tool upgrade --find-links ./wheels codeatlas
```

装好后任意目录执行 `atlas status` → 首次会在 `~/.codeatlas` 生成
`.env.example` 与 `repos.example.yaml`;`cp .env.example .env` 填入
LLM/EMBED 配置,`mv repos.example.yaml repos.yaml` 填仓库清单,
`atlas doctor` 验证连通后即可使用。

多机数据隔离:每个用户的 key/repos/数据都在各自 `~/.codeatlas`;
团队共享大索引可设 `DATA_DIR`(如 NAS/共享盘,注意 SQLite 并发限制为单写)。
MD

STAGE_NAME="codeatlas-bundle-$(date +%Y%m%d)-$(uname -m)"
ZIP="$STAGE_NAME.zip"
rm -rf "/tmp/$STAGE_NAME" && cp -R "$OUT" "/tmp/$STAGE_NAME"
(cd /tmp && zip -qr "$OLDPWD/$OUT/$ZIP" "$STAGE_NAME")
rm -rf "/tmp/$STAGE_NAME"
echo "完成:$OUT/$ZIP(解压后按 INSTALL.md 安装)"
