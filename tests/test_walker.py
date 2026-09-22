"""walker.py:gitignore/exclude 过滤、嵌套 .gitignore、binary 哨探。"""

from pathlib import Path

import pytest

from codeatlas.config import RepoCfg
from codeatlas.ingest.walker import FileEntry, sniff_binary, walk

FIXTURES = Path(__file__).parent / "fixtures"


def repo(root: Path, **kw) -> RepoCfg:
    return RepoCfg(name="t", path=root, **kw)


def rels(entries: list[FileEntry]) -> list[str]:
    return [e.rel for e in entries]


def test_walk_respects_gitignore_and_exclude():
    r = repo(FIXTURES / "java_mini")
    got = rels(walk(r))
    assert "README.md" in got
    assert "com/example/service/UserService.java" in got
    assert "com/example/model/User.java" in got
    # .gitignore:target/ 与 *.log
    assert "target/generated/Ignored.java" not in got
    assert "debug.log" not in got
    # .gitignore 自身被 walker 跳过(与 .git 同列,对知识库无价值)
    assert ".gitignore" not in got


def test_walk_extra_exclude():
    r = repo(
        FIXTURES / "java_mini",
        exclude=["**/util/**", "README.md"],
    )
    got = rels(walk(r))
    assert "README.md" not in got
    assert "com/example/util/Strings.java" not in got
    assert "com/example/model/User.java" in got


def test_walk_nested_gitignore(tmp_path):
    (tmp_path / "src" / "gen").mkdir(parents=True)
    (tmp_path / "src" / "keep.ts").write_text("export const a = 1;\n")
    (tmp_path / "src" / "gen" / "skip.ts").write_text("export const b = 2;\n")
    (tmp_path / "src" / ".gitignore").write_text("gen/\n")
    (tmp_path / "root.txt").write_text("hi\n")
    got = rels(walk(repo(tmp_path)))
    assert got == ["root.txt", "src/keep.ts"]


def test_walk_sorted_deterministic(tmp_path):
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).write_text("x")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "z.txt").write_text("x")
    assert rels(walk(repo(tmp_path))) == ["a.txt", "b.txt", "c.txt", "dir/z.txt"]


def test_walk_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        walk(repo(tmp_path / "nope"))


def test_walk_records_size_and_mtime(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello")
    e = walk(repo(tmp_path))[0]
    assert e.size == 5
    assert e.mtime == pytest.approx(f.stat().st_mtime)
    assert e.abs_path == f


def test_sniff_binary():
    binary = bytes([0x00, 0x01, 0x02, 0x00]) * 64
    text = b"plain ascii text with words\n".ljust(1024, b"x")
    utf8 = "中文注释也是文本\n".encode("utf-8") * 64
    assert sniff_binary(binary) is True
    assert sniff_binary(text) is False
    assert sniff_binary(utf8) is False  # 多字节 UTF-8 不算控制字符
    assert sniff_binary(b"") is False


def test_fixture_binary_sample_is_binary():
    head = (FIXTURES / "binary_sample.dat").open("rb").read(1024)
    assert sniff_binary(head) is True
