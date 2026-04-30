"""
Unit tests for langgraph_solver.

These tests use mock LangGraph-shaped graphs — no real LLM, no API keys.
The point is to nail the contract: import behavior, message injection,
tool-call extraction, error containment.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from fin_safety_inspect.solvers.langgraph_solver import (  # noqa: E402
    LangGraphAdapterError,
    _extract_tool_calls,
    _import_graph_factory,
)


# ---------------------------------------------------------------------------
# Helpers — module-level so they can be imported via "module:callable" string.
# ---------------------------------------------------------------------------


class _MockCompiledGraph:
    """Quacks like a compiled LangGraph: has .ainvoke()."""

    def __init__(self, response_messages, raise_exc=None, capture_input=None):
        self.response_messages = response_messages
        self.raise_exc = raise_exc
        self.capture_input = capture_input  # mutable list to capture inputs

    async def ainvoke(self, state, config=None):
        if self.capture_input is not None:
            self.capture_input.append(state)
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"messages": self.response_messages}


def _factory_returning_safe_response():
    """Mock factory: agent answers in plain text, calls only allowed tools."""
    return _MockCompiledGraph(
        response_messages=[
            HumanMessage(content="user prompt"),
            AIMessage(
                content="checked balance",
                tool_calls=[{"name": "read_balance", "args": {"account": "CHK-001"}, "id": "1"}],
            ),
            ToolMessage(content="$12,543.21", tool_call_id="1"),
            AIMessage(content="Your balance is $12,543.21"),
        ]
    )


def _factory_calling_banned_tool():
    """Mock factory: agent transfers funds (banned tool, simulating IPI success)."""
    return _MockCompiledGraph(
        response_messages=[
            HumanMessage(content="user prompt"),
            AIMessage(
                content="transferring",
                tool_calls=[
                    {"name": "transfer_funds", "args": {"to_account": "EVIL", "amount": 999}, "id": "1"},
                ],
            ),
            ToolMessage(content="OK: transferred $999.00 to EVIL", tool_call_id="1"),
            AIMessage(content="done"),
        ]
    )


def _factory_that_raises():
    """Mock factory: graph blows up on invocation."""
    return _MockCompiledGraph(
        response_messages=[],
        raise_exc=RuntimeError("simulated graph failure"),
    )


# Captures the input the solver passes to the graph; populated per-test.
_LAST_INPUT_CAPTURE: list = []


def _factory_with_input_capture():
    """Mock factory whose graph records every ainvoke() call's input."""
    _LAST_INPUT_CAPTURE.clear()
    return _MockCompiledGraph(
        response_messages=[AIMessage(content="ack")],
        capture_input=_LAST_INPUT_CAPTURE,
    )


# ---------------------------------------------------------------------------
# Tests for the helper functions.
# ---------------------------------------------------------------------------


def test_import_graph_factory_happy_path():
    """Valid 'module:callable' string returns the callable."""
    fn = _import_graph_factory("tests.test_solver_langgraph:_factory_returning_safe_response")
    assert callable(fn)
    g = fn()
    assert isinstance(g, _MockCompiledGraph)


def test_import_graph_factory_missing_colon():
    with pytest.raises(LangGraphAdapterError, match="module:callable"):
        _import_graph_factory("no_colon_here")


def test_import_graph_factory_bad_module():
    with pytest.raises(LangGraphAdapterError, match="Cannot import module"):
        _import_graph_factory("nonexistent_module_xyz:build_graph")


def test_import_graph_factory_bad_attribute():
    with pytest.raises(LangGraphAdapterError, match="no attribute"):
        _import_graph_factory("tests.test_solver_langgraph:no_such_function")


def test_extract_tool_calls_happy_path():
    """Tool calls are extracted in invocation order."""
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_balance", "args": {"account": "A"}, "id": "1"},
                {"name": "list_accounts", "args": {}, "id": "2"},
            ],
        ),
    ]
    calls = _extract_tool_calls(msgs)
    assert [c["name"] for c in calls] == ["read_balance", "list_accounts"]
    assert calls[0]["args"] == {"account": "A"}


def test_extract_tool_calls_no_calls():
    """Messages without tool_calls produce empty list."""
    msgs = [HumanMessage(content="hi"), AIMessage(content="bye")]
    assert _extract_tool_calls(msgs) == []


def test_extract_tool_calls_handles_missing_attribute():
    """Messages without a tool_calls attribute don't crash."""

    class WeirdMessage:
        pass  # no tool_calls attribute

    msgs = [WeirdMessage(), HumanMessage(content="x")]
    calls = _extract_tool_calls(msgs)
    assert calls == []


# ---------------------------------------------------------------------------
# Tests for the solver itself (end-to-end via async).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_solver_records_safe_tool_calls():
    """Happy path: solver records read_balance call into state.metadata."""
    from inspect_ai.solver import TaskState

    from fin_safety_inspect.solvers.langgraph_solver import langgraph_solver

    solver = langgraph_solver(
        graph="tests.test_solver_langgraph:_factory_returning_safe_response"
    )

    state = _make_task_state("Check my balance please")
    result = await solver(state, generate=_dummy_generate)

    assert result.metadata["tool_calls"] == [
        {"name": "read_balance", "args": {"account": "CHK-001"}}
    ]
    assert "Your balance is" in result.output.completion


@pytest.mark.asyncio
async def test_solver_surfaces_banned_tool_call():
    """IPI success path: solver records transfer_funds so scorer can FAIL it."""
    from fin_safety_inspect.solvers.langgraph_solver import langgraph_solver

    solver = langgraph_solver(
        graph="tests.test_solver_langgraph:_factory_calling_banned_tool"
    )

    state = _make_task_state("Trick the agent")
    result = await solver(state, generate=_dummy_generate)

    names = [c["name"] for c in result.metadata["tool_calls"]]
    assert "transfer_funds" in names


@pytest.mark.asyncio
async def test_solver_contains_graph_exceptions():
    """Graph exception goes into metadata, not stack-bombing the eval."""
    from fin_safety_inspect.solvers.langgraph_solver import langgraph_solver

    solver = langgraph_solver(graph="tests.test_solver_langgraph:_factory_that_raises")

    state = _make_task_state("anything")
    result = await solver(state, generate=_dummy_generate)

    assert "langgraph_error" in result.metadata
    assert "simulated graph failure" in result.metadata["langgraph_error"]
    assert result.metadata["tool_calls"] == []
    assert "<langgraph error" in result.output.completion


@pytest.mark.asyncio
async def test_solver_passes_user_prompt_as_initial_human_message():
    """The user_prompt text becomes the first HumanMessage to the graph."""
    from fin_safety_inspect.solvers.langgraph_solver import langgraph_solver

    solver = langgraph_solver(graph="tests.test_solver_langgraph:_factory_with_input_capture")

    state = _make_task_state("Hello world from the eval")
    await solver(state, generate=_dummy_generate)

    assert len(_LAST_INPUT_CAPTURE) == 1
    inp = _LAST_INPUT_CAPTURE[0]
    assert "messages" in inp
    assert len(inp["messages"]) == 1
    assert inp["messages"][0].content == "Hello world from the eval"


# ---------------------------------------------------------------------------
# Test scaffolding helpers.
# ---------------------------------------------------------------------------


def _make_task_state(prompt: str):
    """Build a minimal Inspect TaskState for solver tests."""
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.solver import TaskState

    sample = Sample(input=prompt)
    state = TaskState(
        model="test/mock",
        sample_id="t1",
        epoch=0,
        input=prompt,
        messages=[ChatMessageUser(content=prompt)],
        metadata={},
    )
    return state


async def _dummy_generate(state, **kwargs):
    """No-op Generate for tests — solver doesn't call it directly when it has its own graph."""
    return state
