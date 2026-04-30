"""
LangGraph adapter for Inspect AI.

CONTRACT (the graph_factory must satisfy):
    - graph_factory is "module:callable" (e.g. "examples.mock_helper:build_graph")
    - The callable returns a compiled StateGraph
    - The graph accepts {"messages": list[BaseMessage]} input
    - The graph returns state with "messages" populated

v0.1 limitations (raise NotImplementedError, planned for v2):
    - inject_via="tool_result"  (would need ToolNode hijack)
    - inject_via="document"     (would need RAG abstraction)
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

InjectVia = Literal["user_message", "tool_result", "document"]


class LangGraphAdapterError(Exception):
    """Raised when the LangGraph adapter cannot proceed."""


def _import_graph_factory(graph_factory: str):
    """Import a 'module:callable' string. Fail fast with informative error."""
    if ":" not in graph_factory:
        raise LangGraphAdapterError(
            f"graph_factory must be in 'module:callable' format, got: {graph_factory!r}"
        )
    module_path, callable_name = graph_factory.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise LangGraphAdapterError(
            f"Cannot import module {module_path!r} (graph_factory={graph_factory!r}): {e}"
        ) from e
    try:
        return getattr(module, callable_name)
    except AttributeError as e:
        raise LangGraphAdapterError(
            f"Module {module_path!r} has no attribute {callable_name!r}"
        ) from e


def _extract_tool_calls(messages: list) -> list[dict[str, Any]]:
    """Extract tool calls from a list of LangChain messages.

    Returns list of {"name": str, "args": dict} in invocation order.
    Used by tool_call_state scorer via state.metadata["tool_calls"].
    """
    calls: list[dict[str, Any]] = []
    for msg in messages:
        # AIMessage carries tool_calls; ToolMessage is the response (not a call).
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
    return calls


@solver(name="langgraph")
def langgraph_solver(
    graph: str,
    inject_via: InjectVia = "user_message",
    max_turns: int = 10,
    model: str | None = None,
) -> Solver:
    """Run a LangGraph agent inside an Inspect eval.

    Args:
        graph: "module:callable" returning a compiled StateGraph.
        inject_via: How to inject the attack payload. v0.1 only supports "user_message".
        max_turns: Cap on graph iterations (LangGraph's recursion_limit).
        model: Forwarded to graph_factory(model=...) if provided.

    Writes to state.metadata:
        - "tool_calls": list of {"name", "args"} extracted from final message history
        - "final_messages": full LangGraph message history (for debugging)
        - "truncated": True if max_turns hit
    """
    # Fail fast at solver-construction time, not first-run time.
    factory = _import_graph_factory(graph)

    if inject_via != "user_message":
        # v0.1 contract: only user_message is supported.
        raise NotImplementedError(
            f"inject_via={inject_via!r} is planned for v2. "
            f"Use inject_via='user_message' in v0.1."
        )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Build graph once per sample (cheap; mostly metadata).
        compiled = factory(model=model) if model else factory()

        # The Inspect TaskState already contains the attack-injected user prompt
        # in state.input_text / state.user_prompt. We use it as the initial human turn.
        from langchain_core.messages import HumanMessage

        initial_msgs = [HumanMessage(content=state.user_prompt.text)]

        try:
            result = await compiled.ainvoke(
                {"messages": initial_msgs},
                config={"recursion_limit": max_turns * 2},
            )
        except Exception as e:
            # Don't let graph errors silently zero out the eval — surface them.
            state.metadata["langgraph_error"] = str(e)
            state.metadata["tool_calls"] = []
            err_text = f"<langgraph error: {type(e).__name__}: {e}>"
            state.output = ModelOutput.from_content(
                model=model or "langgraph", content=err_text
            )
            state.messages.append(ChatMessageAssistant(content=err_text))
            return state

        final_messages = result.get("messages", [])
        state.metadata["tool_calls"] = _extract_tool_calls(final_messages)
        state.metadata["final_messages"] = [
            {"type": type(m).__name__, "content": getattr(m, "content", "")}
            for m in final_messages
        ]

        # Last AI message content goes back to Inspect as the completion.
        last_ai = next(
            (m for m in reversed(final_messages) if type(m).__name__ == "AIMessage"),
            None,
        )
        if last_ai is not None:
            content = str(last_ai.content) if last_ai.content is not None else ""
            # Use ModelOutput.from_content so completion AND choices[0].message
            # are both populated. Setting state.output.completion alone leaves
            # choices[0] empty, which can confuse downstream consumers.
            state.output = ModelOutput.from_content(
                model=model or "langgraph", content=content
            )
            state.messages.append(ChatMessageAssistant(content=content))

        return state

    return solve
