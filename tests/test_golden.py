"""golden set 回归(PLAN §9.5):问题 → 期望命中文件,hit@k。

fixtures 三仓库 + 伪语义 embedding(SemanticFakeServer,关键词→维度),
向量/FTS/符号/图扩展四路端到端可回归,零网络零费用。
真实仓库的检索质量验收走 atlas ask(PLAN 验收命令)。
"""

import pytest
import yaml

from codeatlas.config import RepoCfg
from codeatlas.retrieve.search import retrieve
from conftest import FIXTURES, Env, SemanticFakeServer, copy_fixture

K = 3
CASES_FILE = FIXTURES / "golden" / "cases.yaml"


@pytest.fixture(scope="module")
def golden_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("golden")
    env = Env(tmp, monkeypatch=None, server=SemanticFakeServer())
    copy_fixture("java_mini", tmp)
    copy_fixture("ts_mini", tmp)
    copy_fixture("py_mini", tmp)
    env.run(RepoCfg(name="jmini", path=tmp / "java_mini"))
    env.run(RepoCfg(name="tmini", path=tmp / "ts_mini"))
    env.run(RepoCfg(name="pmini", path=tmp / "py_mini"))
    yield env
    env.conn.close()
    env.restore()


def load_cases():
    data = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))
    return data["cases"]


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["question"])
async def test_golden_hit_at_k(golden_env, case):
    env = golden_env
    cands = await retrieve(
        case["question"], env.settings, env.conn, env.provider(), env.lance()
    )
    assert cands, f"问题无候选:{case['question']}"
    top_files = set()
    for c in cands[:K]:
        row = env.one(
            "SELECT f.path AS p FROM chunks c LEFT JOIN files f ON c.file_id=f.id "
            "WHERE c.id=?",
            c.chunk_id,
        )
        if row and row["p"]:
            top_files.add(row["p"])
    missing = [f for f in case["expect_files"] if f not in top_files]
    assert not missing, (
        f"[{case['question']}] 期望 {case['expect_files']} 命中 top{K},"
        f"实际 top{K} 文件:{sorted(top_files)}"
    )
