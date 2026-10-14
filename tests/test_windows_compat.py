"""Windows 兼容回归:在任何平台上用 Windows 语义(PureWindowsPath / cp1252 流 /
不支持 add_signal_handler 的事件循环)复现,保证 macOS/Linux 上也能拦住回退。"""

import asyncio
import io
import signal
import subprocess
from pathlib import PureWindowsPath

import pytest

from codeatlas.config import RepoCfg


def test_ts_module_qname_uses_forward_slashes(monkeypatch):
    """TS/Go 模块限定名曾用 str(Path(rel)),Windows 上得到 src\\order,
    与按 '/' 解析的导入/调用边对不上。"""
    import codeatlas.ingest.symbols as sym

    monkeypatch.setattr(sym, "Path", PureWindowsPath)  # 模拟 Windows 的 Path 语义
    fs = sym.extract_symbols("src/order.ts", "typescript", b"export function total() {}\n")
    assert [s.qualified_name for s in fs.symbols] == ["src/order.total"]
    gofs = sym.extract_symbols("pkg/svc/run.go", "go", b"package svc\nfunc Run() {}\n")
    assert all("\\" not in s.qualified_name for s in gofs.symbols)


def test_wiki_relative_links_on_windows_paths():
    from codeatlas.gencode.wiki import _rel_link

    root = PureWindowsPath(r"C:\Users\me\.codeatlas\data\wiki\repo")
    page = root / "01-ch" / "p.md"
    readme = root / "README.md"
    assert _rel_link(page, readme) == "01-ch/p.md"
    assert _rel_link(readme, page) == "../README.md"
    assert _rel_link(root / "02-x" / "q.md", page) == "../02-x/q.md"


def test_source_links_are_valid_file_uris():
    from codeatlas.gencode.wiki import _file_uri

    assert _file_uri(PureWindowsPath(r"C:\code\svc\A.java")) == "file:///C:/code/svc/A.java"


def test_git_output_decoded_as_utf8(monkeypatch):
    """git 输出 UTF-8;Windows 默认按系统代码页(cp936 等)解码,中文路径会乱码。"""
    import codeatlas.ingest.gitdiff as gd

    seen = {}
    real = subprocess.run

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(gd.subprocess, "run", spy)
    gd._git(PureWindowsPath("."), "--version")
    assert seen.get("encoding") == "utf-8"


def test_missing_git_falls_back_to_scan(tmp_path, monkeypatch):
    """仓库有 .git 但 git 不在 PATH(Windows 常见):退回 scan,而不是 FileNotFoundError。"""
    import codeatlas.ingest.gitdiff as gd

    (tmp_path / ".git").mkdir()
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")

    def no_git(*a, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(gd.subprocess, "run", no_git)
    cs = gd.compute_change_set(RepoCfg(name="t", path=tmp_path), None, {})
    assert cs.mode == "scan" and cs.changes == {"a.py": "A"}


def test_cli_forces_utf8_stdio(monkeypatch):
    """重定向/管道时 Windows 的 stdout 是系统代码页:⚠️ ❯ 等字符会 UnicodeEncodeError。"""
    import codeatlas.cli as cli_mod

    buf = io.BytesIO()
    fake = io.TextIOWrapper(buf, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", fake)
    cli_mod._ensure_utf8_stdio()
    import sys

    print("⚠️ 已过期 ❯")
    sys.stdout.flush()
    assert "已过期".encode() in buf.getvalue()


def test_agent_ctrl_c_without_add_signal_handler(monkeypatch):
    """Windows 的事件循环不支持 add_signal_handler:Ctrl-C 仍只中断本问。"""
    import codeatlas.cli as cli_mod
    from test_cli import _answer, _fake_session

    loop_cls = type(asyncio.new_event_loop())

    def unsupported(self, *a, **kw):
        raise NotImplementedError

    monkeypatch.setattr(loop_cls, "add_signal_handler", unsupported)
    monkeypatch.setattr(loop_cls, "remove_signal_handler", unsupported)
    calls = {"n": 0}

    async def fake_run(question, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(5)
        return _answer(f"答:{question}")

    out = _fake_session(monkeypatch, ["第一问", "第二问"], fake_run)
    asyncio.run(cli_mod._agent_session(None, None, 6, 8))
    joined = "\n".join(out)
    assert "已中断本问" in joined and "答:第二问" in joined


@pytest.mark.parametrize("enc", ["gbk", "utf-8"])
def test_mybatis_mapper_in_gbk(tmp_path, enc):
    from codeatlas.schema.sql_extract import extract_from_xml

    xml = tmp_path / "M.xml"
    xml.write_bytes(('<mapper namespace="x"><!-- 订单查询 -->\n'
                     '<select id="q">SELECT id FROM t_order WHERE name = \'中文\'</select>\n'
                     '</mapper>').encode(enc))
    qs = extract_from_xml(xml, "M.xml")
    assert qs and "中文" in qs[0].sql
