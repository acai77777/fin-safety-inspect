"""Solvers: adapters that drive a target Agent (LangGraph etc.) inside an Inspect eval."""

from fin_safety_inspect.solvers.langgraph_solver import langgraph_solver

__all__ = ["langgraph_solver"]
