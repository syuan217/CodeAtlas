"""图 schema 常量:边类型与置信标签(PLAN §6/§7)。"""

from __future__ import annotations

# 边类型
EDGE_CONTAINS = "CONTAINS"
EDGE_IMPORTS = "IMPORTS"
EDGE_CALLS = "CALLS"

# CALLS 边 resolution 标签
RES_EXACT = "exact"        # 作用域/import/类型精确解析
RES_HEURISTIC = "heuristic"  # 库内唯一同名兜底

# 节点(符号)kind,与 db/models.py 注释一致
SYM_MODULE = "module"
SYM_CLASS = "class"
SYM_INTERFACE = "interface"
SYM_FUNCTION = "function"
SYM_METHOD = "method"

METHOD_SEPARATOR = "#"  # qualified_name 中方法分隔符(容器用 '.')


def split_method_qname(qname: str) -> tuple[str, str | None]:
    """com.x.OrderService#create → ("com.x.OrderService", "create");类 → (qname, None)。"""
    if METHOD_SEPARATOR in qname:
        owner, _, name = qname.rpartition(METHOD_SEPARATOR)
        return owner, name
    return qname, None
