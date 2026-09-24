"""python -m codeatlas 入口(供 update 包装命令以子进程复用 CLI)。"""
from codeatlas.cli import app

app()
