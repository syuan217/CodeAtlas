"""indexer.py 集成测试:M1 验收标准的核心场景。

mock embedding(MockTransport)替代真实 API;库与 LanceDB 全部指向 tmp_path。
"""

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from codeatlas import config
from codeatlas.config import RepoCfg, Settings
from codeatlas.db.fts import fts_search
from codeatlas.db.lance import LanceStore
from codeatlas.db.models import connect, init_db
from codeatlas.ingest.indexer import index_repo
from codeatlas.providers.embedding import EmbeddingProvider

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedServer:
    def __init__(self, dim=4):
        self.dim = dim
        self.request_count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        body = json.loads(request.content)
        data = [
            {"index": i, "embedding": [0.01 * ((i % 97) + 1)] * self.dim}
            for i in range(len(body["input"]))
        ]
        return httpx.Response(
            200, json={"data": data, "usage": {"total_tokens": len(body["input"])}}
        )


class Env:
    """每个测试一套隔离环境:库 + LanceDB + provider(挂同一 conn)。"""

    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DB_PATH", tmp_path / "kb.sqlite")
        monkeypatch.setattr(config, "LANCEDB_DIR", tmp_path / "lancedb")
        self.settings = Settings(
            _env_file=None,
            embed_base_url="http://embed.test/v1",
            embed_api_key="sk-test",
            embed_model="test-embed",
            embed_dim=4,
            embed_batch_size=64,
            chunk_max_tokens=512,
            max_concurrency=4,
            cost_limit_per_run=50.0,
        )
        self.conn = connect()
        init_db(self.conn)
        self.server = FakeEmbedServer()

    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider(
            self.settings, self.conn,
            transport=httpx.MockTransport(self.server), backoff_base=0,
        )

    def run(self, repo: RepoCfg, **kw):
        return index_repo(
            repo, settings=self.settings, embed_provider=self.provider(),
            conn=self.conn, **kw
        )

    def lance(self) -> LanceStore:
        return LanceStore(self.settings)

    def q(self, sql: str, *params):
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, *params):
        return self.conn.execute(sql, params).fetchone()

    def count(self, table: str) -> int:
        return self.one(f"SELECT COUNT(*) AS c FROM {table}")["c"]


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    return Env(tmp_path, monkeypatch)


def copy_fixture(name: str, tmp: Path) -> Path:
    dst = tmp / name
    shutil.copytree(FIXTURES / name, dst)
    return dst


def chunk_ids_by_file(env: Env) -> dict[str, set[int]]:
    rows = env.q(
        "SELECT f.path AS p, c.id AS cid FROM chunks c JOIN files f ON c.file_id=f.id"
    )
    out: dict[str, set[int]] = {}
    for r in rows:
        out.setdefault(r["p"], set()).add(r["cid"])
    return out


def chunk_hashes_by_file(env: Env) -> dict[str, set[str]]:
    rows = env.q(
        "SELECT f.path AS p, c.content_hash AS h FROM chunks c JOIN files f ON c.file_id=f.id"
    )
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r["p"], set()).add(r["h"])
    return out


# ---------------------------------------------------------------------------
# 首次索引:全部组件落库
# ---------------------------------------------------------------------------

def test_first_index_scan_mode_full_pipeline(env, tmp_path):
    repo_root = copy_fixture("java_mini", tmp_path)
    repo = RepoCfg(name="jmini", path=repo_root, languages=["java", "markdown"])
    stats = env.run(repo)

    # scan 模式(非 git 目录);target/、*.log、.gitignore 被排除
    assert stats.mode == "scan"
    assert stats.added == 4  # README.md + 3 java
    assert stats.skipped_binary == 0

    paths = {r["path"] for r in env.q("SELECT path FROM files")}
    assert paths == {
        "README.md",
        "com/example/model/User.java",
        "com/example/service/UserService.java",
        "com/example/util/Strings.java",
    }
    assert all(r["parse_status"] == "ok" for r in env.q("SELECT parse_status FROM files"))

    # 符号:module×4 + class×3 + method×6(UserService 2 / User 3 / Strings 1)
    assert env.count("symbols") == stats.symbols == 13
    assert stats.contains_edges == 9  # 3 class 挂 module + 6 方法挂 class
    assert env.count("edges") == stats.contains_edges + stats.import_edges

    # IMPORTS 边:UserService → User、Strings
    imp = env.q(
        "SELECT a.qualified_name AS s, b.qualified_name AS d FROM edges e "
        "JOIN symbols a ON e.src_id=a.id JOIN symbols b ON e.dst_id=b.id "
        "WHERE e.kind='IMPORTS'"
    )
    assert {(r["s"], r["d"]) for r in imp} == {
        ("com/example/service/UserService.java", "com/example/model/User.java"),
        ("com/example/service/UserService.java", "com/example/util/Strings.java"),
    }

    # chunks:全部有向量引用;code/doc 两种 kind
    n_chunks = env.count("chunks")
    assert n_chunks == env.count("vector_refs") == stats.chunks > 0
    kinds = {r["kind"] for r in env.q("SELECT DISTINCT kind FROM chunks")}
    assert kinds == {"code", "doc"}

    # FTS 可检索
    hits = fts_search(env.conn, "truncate")
    assert any("Strings" in (h["title"] or "") for h in hits)

    # usage_log:index 阶段 embedding 费用落库
    stages = [r["stage"] for r in env.q("SELECT stage FROM usage_log")]
    assert "index" in stages

    # LanceDB 行数 == chunks 数
    assert env.lance().count() == n_chunks


def test_second_run_noop(env, tmp_path):
    repo_root = copy_fixture("java_mini", tmp_path)
    repo = RepoCfg(name="jmini", path=repo_root)
    env.run(repo)
    requests_after_first = env.server.request_count

    stats2 = env.run(repo)
    # scan 模式二次运行:mtime 快速跳过 → 变更集为空,零工作
    assert stats2.mode == "scan"
    assert stats2.added == 0 and stats2.modified == 0 and stats2.deleted == 0
    assert stats2.chunks == 0
    assert env.server.request_count == requests_after_first  # 无新 embedding 请求
    assert env.count("chunks") > 0  # 数据仍在


def test_modify_one_file_only_that_file_reprocessed(env, tmp_path):
    repo_root = copy_fixture("java_mini", tmp_path)
    repo = RepoCfg(name="jmini", path=repo_root)
    env.run(repo)
    before = chunk_ids_by_file(env)

    # 只改一个文件(内容+mtime 都变)
    (repo_root / "com/example/util/Strings.java").write_text(
        "package com.example.util;\n\npublic class Strings {\n"
        "    public static String pad(String s, int n) {\n        return s;\n    }\n}\n"
    )

    stats2 = env.run(repo)
    assert stats2.modified == 1
    assert stats2.deleted == 0
    # scan 模式:mtime 未变的文件在变更集阶段就被快跳(不进 _process_file)

    after = chunk_ids_by_file(env)
    # 未改文件的 chunk id 完全不动
    for path in before:
        if path != "com/example/util/Strings.java":
            assert before[path] == after.get(path), f"{path} 被意外重解析"
    # 改动文件内容已替换(id 可能被 SQLite 复用,以内容为准)
    contents = " ".join(
        r["content"] for r in env.q(
            "SELECT content FROM chunks WHERE file_id IN "
            "(SELECT id FROM files WHERE path='com/example/util/Strings.java')"
        )
    )
    assert "pad" in contents and "truncate" not in contents
    # 图仍完整:IMPORTS 边指向新子树
    imp = env.one(
        "SELECT COUNT(*) AS c FROM edges WHERE kind='IMPORTS' "
        "AND dst_id IN (SELECT id FROM symbols WHERE file_id IN "
        "  (SELECT id FROM files WHERE path='com/example/util/Strings.java'))"
    )
    assert imp["c"] == 1


def test_delete_file_cleans_subtree(env, tmp_path):
    repo_root = copy_fixture("java_mini", tmp_path)
    repo = RepoCfg(name="jmini", path=repo_root)
    env.run(repo)
    chunks_before = env.count("chunks")

    (repo_root / "com/example/util/Strings.java").unlink()
    stats2 = env.run(repo)
    assert stats2.deleted == 1

    assert env.count("files") == 3
    assert env.count("symbols") == 10  # 13 - (module+class+method)
    assert env.count("chunks") < chunks_before
    assert env.count("vector_refs") == env.count("chunks")
    assert env.lance().count() == env.count("chunks")


def test_binary_large_unknown_skipped(env, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("text file")
    (root / "b.dat").write_bytes(bytes([0x00, 0x01, 0x00]) * 600)
    (root / "big.py").write_text("x = 1\n" * 200000)  # ~1.2MB
    repo = RepoCfg(name="mix", path=root)
    stats = env.run(repo)
    assert stats.skipped_unknown_ext == 1  # a.txt
    assert stats.skipped_binary == 1  # b.dat
    assert stats.skipped_large == 1  # big.py
    assert stats.chunks == 0
    statuses = {r["path"]: r["parse_status"] for r in env.q("SELECT path, parse_status FROM files")}
    assert all(v == "skipped" for v in statuses.values())
    # skipped 文件也记录 hash,增量比对稳定
    assert all(r["hash"] for r in env.q("SELECT hash FROM files"))


# ---------------------------------------------------------------------------
# git 模式:基线推进
# ---------------------------------------------------------------------------

def git_(cwd, *args) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(cwd), *args],
        check=True, capture_output=True,
    )


def test_git_mode_baseline_advances(env, tmp_path):
    root = tmp_path / "grepo"
    shutil.copytree(FIXTURES / "py_mini", root)
    git_(root, "init")
    git_(root, "add", "-A")
    git_(root, "commit", "-m", "init")

    repo = RepoCfg(name="py", path=root)
    stats1 = env.run(repo)
    assert stats1.mode == "git"
    assert stats1.base_commit is None
    assert stats1.added == 3  # pkg/__init__.py + pkg/mod.py + main.py

    base1 = env.one("SELECT indexed_commit FROM repos WHERE name='py'")["indexed_commit"]
    assert base1 == stats1.head_commit

    # 无变更:git 模式 noop,基线不动
    stats2 = env.run(repo)
    assert stats2.mode == "git"
    assert stats2.added + stats2.modified + stats2.deleted == 0

    # 改一个文件 + commit → 只处理该文件,基线推进
    (root / "pkg/mod.py").write_text("MAX = 200\n\n\ndef only(v):\n    return v\n")
    git_(root, "add", "-A")
    git_(root, "commit", "-m", "mod")
    stats3 = env.run(repo)
    assert stats3.mode == "git"
    assert stats3.modified == 1 and stats3.base_commit == base1
    base3 = env.one("SELECT indexed_commit FROM repos WHERE name='py'")["indexed_commit"]
    assert base3 == stats3.head_commit and base3 != base1


def test_git_dirty_falls_back_to_scan_and_still_works(env, tmp_path):
    root = tmp_path / "grepo"
    shutil.copytree(FIXTURES / "py_mini", root)
    git_(root, "init")
    git_(root, "add", "-A")
    git_(root, "commit", "-m", "init")
    repo = RepoCfg(name="py", path=root)
    env.run(repo)

    # 未提交改动 → scan 兜底;mtime 未变的文件在变更集阶段快跳
    (root / "main.py").write_text("from pkg.mod import Counter\n\ndef go():\n    return 1\n")
    stats2 = env.run(repo)
    assert stats2.mode == "scan"
    assert stats2.modified == 1
    # 基线不推进(scan 模式)
    assert env.one("SELECT indexed_commit FROM repos WHERE name='py'")["indexed_commit"] is not None


# ---------------------------------------------------------------------------
# 中断自愈与 --full
# ---------------------------------------------------------------------------

def test_incomplete_vectors_are_healed_on_rerun(env, tmp_path):
    repo_root = copy_fixture("java_mini", tmp_path)
    repo = RepoCfg(name="jmini", path=repo_root)
    env.run(repo)

    # 模拟 embed 阶段中断:删掉一半 vector_refs + Lance 行
    ids = [r["id"] for r in env.q("SELECT id FROM chunks")]
    half = ids[: len(ids) // 2]
    env.lance().delete_by_chunk_ids(half)
    env.conn.execute(
        f"DELETE FROM vector_refs WHERE chunk_id IN ({','.join(map(str, half))})"
    )
    env.conn.commit()

    stats2 = env.run(repo)
    assert stats2.modified >= 1  # 受影响文件重处理
    n_missing = env.one(
        "SELECT COUNT(*) AS c FROM chunks c WHERE NOT EXISTS "
        "(SELECT 1 FROM vector_refs v WHERE v.chunk_id=c.id)"
    )["c"]
    assert n_missing == 0
    assert env.count("vector_refs") == env.count("chunks")
    assert env.lance().count() == env.count("chunks")


def test_exclude_filters_git_changesets_too(env, tmp_path):
    """git 模式下 tracked 的构建产物(exclude 匹配)不进索引。"""
    root = tmp_path / "grepo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "A.java").write_text("package a;\npublic class A {}\n")
    (root / "sub" / "target").mkdir(parents=True)  # 模拟历史误提交的产物
    (root / "sub" / "target" / "Gen.class.lst").write_text("noise")
    git_(root, "init")
    git_(root, "add", "-A")  # target 产物也 tracked
    git_(root, "commit", "-m", "init")

    repo = RepoCfg(name="g", path=root, exclude=["**/target/"])
    stats = env.run(repo)
    assert stats.mode == "git"
    assert stats.added == 1  # 只有 src/A.java
    paths = {r["path"] for r in env.q("SELECT path FROM files")}
    assert paths == {"src/A.java"}


def test_full_forces_reparse(env, tmp_path):
    repo_root = copy_fixture("java_mini", tmp_path)
    repo = RepoCfg(name="jmini", path=repo_root)
    env.run(repo)
    before_ids = chunk_ids_by_file(env)

    stats2 = env.run(repo, full=True)
    assert stats2.skipped_unchanged == 0
    assert stats2.modified == 4  # --full:全部已知文件强制进变更集
    after_ids = chunk_ids_by_file(env)
    assert all(before_ids[p] != after_ids[p] for p in before_ids)  # 全部重建
    assert env.count("vector_refs") == env.count("chunks")
