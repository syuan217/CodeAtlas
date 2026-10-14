"""indexer.py 集成测试:M1 验收标准的核心场景。

mock embedding(MockTransport)替代真实 API;库与 LanceDB 全部指向 tmp_path。
"""

import shutil
from pathlib import Path

import pytest

from codeatlas.config import RepoCfg
from codeatlas.db.fts import fts_search
from conftest import FIXTURES, Env, copy_fixture, git_


def chunk_ids_by_file(env: Env) -> dict[str, set[int]]:
    rows = env.q(
        "SELECT f.path AS p, c.id AS cid FROM chunks c JOIN files f ON c.file_id=f.id"
    )
    out: dict[str, set[int]] = {}
    for r in rows:
        out.setdefault(r["p"], set()).add(r["cid"])
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
    assert env.count("edges") == (
        stats.contains_edges + stats.import_edges
        + stats.call_edges_exact + stats.call_edges_heuristic
    )
    # CALLS:UserService 的调用(Strings.truncate/User.getName 等)有 exact 边
    assert stats.call_edges_exact >= 3
    calls = env.q(
        "SELECT a.qualified_name AS c, b.qualified_name AS d, e.resolution "
        "FROM edges e JOIN symbols a ON e.src_id=a.id JOIN symbols b ON e.dst_id=b.id "
        "WHERE e.kind='CALLS'"
    )
    assert any(
        r["d"] == "com.example.util.Strings#truncate" and r["resolution"] == "exact"
        for r in calls
    ), calls

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
    , encoding="utf-8")

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
    (root / "a.txt").write_text("text file", encoding="utf-8")
    (root / "b.dat").write_bytes(bytes([0x00, 0x01, 0x00]) * 600)
    (root / "big.py").write_text("x = 1\n" * 200000, encoding="utf-8")  # ~1.2MB
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
    (root / "pkg/mod.py").write_text("MAX = 200\n\n\ndef only(v):\n    return v\n", encoding="utf-8")
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
    (root / "main.py").write_text("from pkg.mod import Counter\n\ndef go():\n    return 1\n", encoding="utf-8")
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
    (root / "src" / "A.java").write_text("package a;\npublic class A {}\n", encoding="utf-8")
    (root / "sub" / "target").mkdir(parents=True)  # 模拟历史误提交的产物
    (root / "sub" / "target" / "Gen.class.lst").write_text("noise", encoding="utf-8")
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


def test_add_vectors_idempotent_no_duplicates(env):
    """同 chunk_id 重复写入不产生重复行(SQL id 复用场景的防御)。"""
    store = env.lance()
    rows = [(7, 1, "code", bytes([0]) * 16)]
    store.add_vectors(rows)
    store.add_vectors(rows)  # 同 id 再写:先清后写,不叠加
    store.add_vectors([(8, 1, "code", bytes([1]) * 16)])
    t = store.table
    arrow = t.to_arrow()
    ids = arrow.column("chunk_id").to_pylist()
    assert sorted(ids) == [7, 8]
    assert t.count_rows() == 2


def test_rebuild_from_sql_dedupes(env):
    """repair:按 vector_refs+embed_cache 重写后无重复无孤儿。"""
    from codeatlas.config import content_hash

    store = env.lance()
    conn = env.conn
    # 手工造脏数据:lance 三份同 id 行
    blob = bytes([2]) * 16
    store.add_vectors([(9, 1, "code", blob)])
    t = store.table
    t.add([{"id": "dup-1", "vector": [0.0] * 4, "chunk_id": 9, "repo_id": 1, "kind": "code"}])
    t.add([{"id": "dup-2", "vector": [0.0] * 4, "chunk_id": 9, "repo_id": 1, "kind": "code"}])
    assert t.count_rows() == 3
    # SQL 侧:repo/file/chunk/vector_refs/embed_cache 各一条
    conn.execute("INSERT INTO repos(name,path,languages) VALUES('r','/r','[]')")
    repo_id = conn.execute("SELECT id FROM repos").fetchone()["id"]
    conn.execute("INSERT INTO files(repo_id,path) VALUES(?, 'a.py')", (repo_id,))
    file_id = conn.execute("SELECT id FROM files").fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks(repo_id,file_id,kind,content,content_hash) "
        "VALUES(?,?,'code','x',?)", (repo_id, file_id, content_hash("x")))
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()["id"]
    # 重建用 chunk_id=9 对齐上面的脏数据
    conn.execute("UPDATE chunks SET id=9 WHERE id=?", (chunk_id,))
    chunk_id = 9
    conn.execute(
        "INSERT INTO embed_cache(content_hash,model,dim,vector) VALUES(?,?,?,?)",
        (content_hash("x"), "test-embed", 4, blob))
    conn.execute(
        "INSERT INTO vector_refs(chunk_id,lance_id,model) VALUES(?,?,?)",
        (chunk_id, "old", "test-embed"))
    conn.commit()

    rebuilt, missing = store.rebuild_from_sql(conn)
    conn.commit()
    assert (rebuilt, missing) == (1, 0)
    assert store.count() == 1
    row = store.table.to_arrow()
    assert row.column("chunk_id").to_pylist() == [9]
    vr = conn.execute("SELECT lance_id FROM vector_refs WHERE chunk_id=9").fetchone()
    assert vr["lance_id"] != "old"  # lance_id 已同步更新


def test_duplicate_chunks_within_file_deduped(env, tmp_path):
    """同文件内完全重复的内容块(md 重复段落)不得撞 idx_chunks_dedup。"""
    root = tmp_path / "repo"
    root.mkdir()
    dup = ("# T" + chr(10) * 2 + "## A" + chr(10) * 2 + "same line" + chr(10) * 2
            + "## A" + chr(10) * 2 + "same line" + chr(10) * 2 + "## Z")  # 中间两块完全相同
    (root / "dup.md").write_text(dup, encoding="utf-8")
    repo = RepoCfg(name="dup", path=root)
    stats = env.run(repo)
    assert stats.chunks >= 1  # 不炸,重复块被去重
    n = env.one(
        "SELECT COUNT(*) AS c FROM chunks WHERE content LIKE '%same line%'"
    )["c"]
    assert n == 1


def test_dependent_files_many_paths(env):
    """首个 git 索引会把 ls-files 全量路径传入:不得超 SQLite 变量上限。"""
    from codeatlas.ingest.indexer import _dependent_files

    env.conn.execute("INSERT INTO repos(name,path,languages) VALUES('big','/x','[]')")
    rid = env.one("SELECT id FROM repos WHERE name='big'")["id"]
    changes = {f"src/f{i}.py": "A" for i in range(40_000)}
    assert _dependent_files(env.conn, rid, changes) == set()


def _calls(env, repo="r"):
    return {
        (r["s"], r["d"]) for r in env.q(
            "SELECT a.qualified_name AS s, b.qualified_name AS d FROM edges e "
            "JOIN symbols a ON a.id=e.src_id JOIN symbols b ON b.id=e.dst_id "
            "JOIN repos rp ON rp.id=e.repo_id WHERE e.kind='CALLS' AND rp.name=?", repo)
    }


def _py_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "x.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "y.py").write_text("from x import helper\n\n\ndef run():\n    return helper()\n", encoding="utf-8")
    return root


def test_gbk_file_indexed_with_readable_chunks(env, tmp_path):
    root = tmp_path / "g"
    (root / "com/x").mkdir(parents=True)
    (root / "com/x/G.java").write_bytes(
        ('package com.x;\npublic class G {\n    /** 订单校验 */\n'
         '    public boolean ok(String x) { return "中文".equals(x); }\n}\n').encode("gbk"))
    env.run(RepoCfg(name="g", path=root))
    assert env.one("SELECT parse_status FROM files WHERE path='com/x/G.java'")[0] == "ok"
    assert fts_search(env.conn, "订单校验") or any(
        "订单校验" in r["content"] for r in env.q("SELECT content FROM chunks"))


def test_one_bad_file_does_not_abort_run(env, tmp_path, monkeypatch):
    import codeatlas.ingest.indexer as idx

    root = _py_repo(tmp_path / "r")
    (root / "boom.py").write_text("def boom():\n    pass\n", encoding="utf-8")
    real = idx.extract_symbols

    def flaky(rel, lang, raw):
        if rel == "boom.py":
            raise RuntimeError("parser exploded")
        return real(rel, lang, raw)

    monkeypatch.setattr(idx, "extract_symbols", flaky)
    stats = env.run(RepoCfg(name="r", path=root))
    status = {r["path"]: r["parse_status"] for r in env.q("SELECT path, parse_status FROM files")}
    assert status["boom.py"] == "failed" and status["x.py"] == "ok"
    assert stats.parse_failed >= 1
    assert ("y.run", "x.helper") in _calls(env)
    assert env.count("symbols") == env.one(
        "SELECT COUNT(*) FROM symbols s JOIN files f ON f.id=s.file_id")[0]  # 无半截残留


def test_same_fqcn_in_two_maven_modules(env, tmp_path):
    """多模块 Maven 工程同 FQCN:不得撞 UNIQUE(repo_id, qualified_name) 中断索引。"""
    root = tmp_path / "mvn"
    for m in ("m1", "m2"):
        d = root / m / "src/main/java/com/x"
        d.mkdir(parents=True)
        (d / "App.java").write_text("package com.x;\npublic class App { void run() {} }\n", encoding="utf-8")
    env.run(RepoCfg(name="mvn", path=root))
    from codeatlas.graph.queries import find_symbols

    assert len(find_symbols(env.conn, "com.x.App")) == 2


def test_interrupted_first_index_recovers_call_edges(env, tmp_path, monkeypatch):
    import codeatlas.ingest.indexer as idx

    root = _py_repo(tmp_path / "r")
    real = idx.resolve_and_write_calls
    monkeypatch.setattr(idx, "resolve_and_write_calls",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        env.run(RepoCfg(name="r", path=root))
    monkeypatch.setattr(idx, "resolve_and_write_calls", real)
    env.run(RepoCfg(name="r", path=root))
    assert ("y.run", "x.helper") in _calls(env)


def test_interrupted_incremental_keeps_dependent_edges(env, tmp_path, monkeypatch):
    """改 x.py 后在图解析前中断:重跑必须恢复依赖者 y.py → x.helper 的调用边。"""
    import time

    import codeatlas.ingest.indexer as idx

    root = _py_repo(tmp_path / "r")
    env.run(RepoCfg(name="r", path=root))
    assert ("y.run", "x.helper") in _calls(env)
    time.sleep(0.01)
    (root / "x.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
    real = idx.resolve_and_write_calls
    monkeypatch.setattr(idx, "resolve_and_write_calls",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        env.run(RepoCfg(name="r", path=root))
    monkeypatch.setattr(idx, "resolve_and_write_calls", real)
    env.run(RepoCfg(name="r", path=root))
    assert ("y.run", "x.helper") in _calls(env)
    assert env.one("SELECT COUNT(*) FROM files WHERE parse_status='pending'")[0] == 0


def test_dangling_vector_refs_are_healed(env, tmp_path):
    """向量被删但引用还在(删 Lance 后 SQL 未提交即中断):对账撤引用,下轮自愈重嵌。"""
    root = _py_repo(tmp_path / "r")
    env.run(RepoCfg(name="r", path=root))
    lance = env.lance()
    victim = env.one("SELECT chunk_id FROM vector_refs LIMIT 1")["chunk_id"]
    lance.delete_by_chunk_ids([victim])
    env.run(RepoCfg(name="r", path=root))     # 对账发现缺失 → 撤引用
    env.run(RepoCfg(name="r", path=root))     # 自愈重嵌
    ids = set(env.lance().table.to_arrow().column("chunk_id").to_pylist())  # 新句柄读最新版本
    refs = {r["chunk_id"] for r in env.q("SELECT chunk_id FROM vector_refs")}
    assert refs == ids and refs
