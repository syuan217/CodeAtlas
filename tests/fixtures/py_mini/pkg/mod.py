"""pkg.mod —— 工具模块。"""

MAX = 100


def clamp(value: int, low: int = 0, high: int = MAX) -> int:
    """把 value 限制在 [low, high] 区间。"""
    return max(low, min(high, value))


class Counter:
    def __init__(self, start: int = 0):
        self.value = start

    def bump(self, step: int = 1) -> int:
        self.value = clamp(self.value + step)
        return self.value
