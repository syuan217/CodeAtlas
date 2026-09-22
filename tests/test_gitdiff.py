"""gitdiff.py:git 主路径(diff --name-status)与 scan 兜底。"""

import subprocess

from codeatlas.config import RepoCfg
from codeatlas.ingest.gitdiff import compute_change_set


def git(cwd, *args) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
    )


def make_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    (root / "a.txt").write_text("alpha")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("beta")
    git(root, "add", "-A")
    git(root, "commit", "-m", "init")
    return root


def cs_of(root, indexed=None, known=None):
    repo = RepoCfg(name="t", path=root)
    return compute_change_set(repo, indexed, known or {})


def test_git_first_index_lists_tracked(tmp_path):
    root = make_repo(tmp_path)
    cs = cs_of(root)
    assert cs.mode == "git"
    assert cs.base_commit is None
    assert cs.changes == {"a.txt": "A", "sub/b.txt": "A"}
    assert len(cs.head_commit) == 40


def test_git_no_changes_second_run(tmp_path):
    root = make_repo(tmp_path)
    head = cs_of(root).head_commit
    cs = cs_of(root, indexed=head)
    assert cs.mode == "git"
    assert cs.is_noop


def test_git_detects_modify_add_delete(tmp_path):
    root = make_repo(tmp_path)
    head = cs_of(root).head_commit

    (root / "a.txt").write_text("alpha2")
    (root / "c.txt").write_text("gamma")
    (root / "sub" / "b.txt").unlink()
    git(root, "add", "-A")
    git(root, "commit", "-m", "changes")

    cs = cs_of(root, indexed=head)
    assert cs.mode == "git"
    assert cs.changes == {"a.txt": "M", "c.txt": "A", "sub/b.txt": "D"}


def test_git_rename_becomes_delete_plus_add(tmp_path):
    root = make_repo(tmp_path)
    head = cs_of(root).head_commit
    (root / "a.txt").rename(root / "renamed.txt")
    git(root, "add", "-A")
    git(root, "commit", "-m", "rename")
    cs = cs_of(root, indexed=head)
    assert cs.changes == {"a.txt": "D", "renamed.txt": "A"}


def test_dirty_worktree_falls_back_to_scan(tmp_path):
    root = make_repo(tmp_path)
    head = cs_of(root).head_commit
    (root / "a.txt").write_text("uncommitted change")  # 未 commit → diff 看不到
    cs = cs_of(root, indexed=head)
    assert cs.mode == "scan"
    # known 为空 → 全部文件视为新增(scan 模式下由 indexer 的 hash 幂等跳过兜底)
    assert cs.changes == {"a.txt": "A", "sub/b.txt": "A"}


def test_missing_base_commit_falls_back_to_scan(tmp_path):
    root = make_repo(tmp_path)
    cs = cs_of(root, indexed="0" * 40)
    assert cs.mode == "scan"


def test_non_git_dir_uses_scan(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    (root / "x.txt").write_text("x")
    cs = cs_of(root)
    assert cs.mode == "scan"
    assert cs.changes == {"x.txt": "A"}


def test_scan_compares_hash_and_mtime(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    f = root / "x.txt"
    f.write_text("v1")
    from codeatlas.config import hash_bytes

    known = {"x.txt": (hash_bytes(b"v1"), f.stat().st_mtime)}
    cs = cs_of(root, known=known)
    assert cs.is_noop  # hash+mtime 均未变

    f.write_text("v2")
    cs = cs_of(root, known=known)
    assert cs.changes == {"x.txt": "M"}


def test_scan_detects_deletes(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    known = {"gone.txt": ("deadbeef", 1.0)}
    cs = cs_of(root, known=known)
    assert cs.changes == {"gone.txt": "D"}
