"""Smoke tests: tasks load and have non-empty datasets.

Day 2 expansion: add unit tests for langgraph_solver and tool_call_state scorer.
"""

from __future__ import annotations

import pytest

from fin_safety_inspect.tasks import financebench_numeric, fintrust_safety


def test_fintrust_safety_loads_offline():
    """Offline mode (use_remote=False) returns embedded 10-sample subset."""
    t = fintrust_safety(use_remote=False)
    samples = list(t.dataset)
    assert len(samples) == 10
    # Each sample has the expected metadata shape.
    for s in samples:
        assert s.metadata["expected_behavior"] == "refuse"
        assert s.metadata["category"] in {
            "Tax Evasion",
            "Money Laundering",
            "Insider Trading",
            "Financial Fraud",
            "Counterfeit Currency or Financial Documents",
            "Illegal Fund-raising",
            "Illegal Foreign Exchange Trading",
            "Credit Card Fraud",
            "Illegal Lending Activities",
            "Financial Identity Theft",
        }


def test_fintrust_safety_max_samples():
    """max_samples caps the dataset size."""
    t = fintrust_safety(use_remote=False, max_samples=3)
    assert len(list(t.dataset)) == 3


def test_financebench_numeric_loads():
    t = financebench_numeric()
    assert len(list(t.dataset)) > 0


def test_mock_helper_imports():
    """fin_safety_inspect.examples.mock_helper imports cleanly without API key."""
    from fin_safety_inspect.examples import mock_helper

    assert hasattr(mock_helper, "build_graph")
    assert "transfer_funds" in mock_helper.BANNED_TOOLS
    assert "read_balance" in mock_helper.ALLOWED_TOOLS


def test_langgraph_solver_factory_validation():
    """langgraph_solver fails fast on bad graph= argument."""
    from fin_safety_inspect.solvers.langgraph_solver import (
        LangGraphAdapterError,
        langgraph_solver,
    )

    with pytest.raises(LangGraphAdapterError, match="module:callable"):
        langgraph_solver(graph="not_a_valid_path")

    with pytest.raises(LangGraphAdapterError, match="Cannot import"):
        langgraph_solver(graph="nonexistent_module:build_graph")


def test_langgraph_solver_rejects_unsupported_inject_modes():
    """v0.1 contract: only inject_via='user_message' is supported."""
    from fin_safety_inspect.solvers.langgraph_solver import langgraph_solver

    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")

    examples_path = "fin_safety_inspect.examples.mock_helper:build_graph"

    with pytest.raises(NotImplementedError, match="planned for v2"):
        langgraph_solver(graph=examples_path, inject_via="tool_result")

    with pytest.raises(NotImplementedError, match="planned for v2"):
        langgraph_solver(graph=examples_path, inject_via="document")
