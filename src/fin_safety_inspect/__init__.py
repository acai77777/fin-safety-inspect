"""fin-safety-inspect: financial agent safety task pack for Inspect AI.

Tasks/solvers/scorers are re-exported at the package top level so that
`inspect eval fin_safety_inspect/<name>` resolves them.
"""

# Importing these triggers @task / @solver / @scorer registration.
from fin_safety_inspect.scorers import keyword_refusal, tool_call_state  # noqa: F401
from fin_safety_inspect.solvers import langgraph_solver  # noqa: F401
from fin_safety_inspect.tasks import financebench_numeric, fintrust_safety  # noqa: F401

__version__ = "0.1.0.dev0"

__all__ = [
    "fintrust_safety",
    "financebench_numeric",
    "langgraph_solver",
    "keyword_refusal",
    "tool_call_state",
]
