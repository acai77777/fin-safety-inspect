"""Tasks: Inspect AI tasks for financial agent safety."""

from fin_safety_inspect.tasks.financebench_numeric import financebench_numeric
from fin_safety_inspect.tasks.fintrust_safety import fintrust_safety

__all__ = ["fintrust_safety", "financebench_numeric"]
