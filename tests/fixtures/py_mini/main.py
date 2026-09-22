"""入口:演示 import 解析。"""

from pkg.mod import Counter, clamp
import pkg.mod as pm


def run(n: int) -> int:
    c = Counter()
    for _ in range(n):
        c.bump()
    return clamp(c.value, high=pm.MAX)
