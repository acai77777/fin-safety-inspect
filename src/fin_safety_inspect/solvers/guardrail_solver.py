"""
Guardrail solver — wraps any Inspect AI solver to intercept Mode #3 harmful outputs.

After the inner solver produces output, the guardrail runs an LLM classifier to check
for harmful operational guidance. If found, the output is replaced with a safe refusal.
Metadata records the original output and guardrail decision for downstream evaluation.

Usage:
    # Wrap the default generate() solver:
    solver=guardrail_solver(inner=generate())

    # Wrap a LangGraph agent:
    solver=guardrail_solver(
        inner=langgraph_solver(graph="...:build_graph", model="openai:deepseek-chat"),
    )

    # Detect-only mode (log but don't replace):
    solver=guardrail_solver(inner=generate(), mode="detect_only")
"""

from __future__ import annotations

from typing import Literal

from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from fin_safety_inspect.scorers.guardrail_classifier import (
    SAFE_REFUSAL,
    _run_guardrail_check,
)


@solver(name="guardrail")
def guardrail_solver(
    inner: Solver,
    guardrail_model: str = "deepseek-v4-flash",
    mode: Literal["replace", "detect_only"] = "replace",
) -> Solver:
    """Wrap an inner solver with output-side guardrail against Mode #3.

    Args:
        inner: The solver to wrap (e.g. generate() or langgraph_solver(...)).
        guardrail_model: Model name (no provider prefix) for the guardrail classifier.
            Reads JUDGE_API_KEY / JUDGE_BASE_URL from env (falls back to OPENAI_*).
        mode: "replace" rewrites harmful outputs with a safe refusal; "detect_only"
            stores guardrail metadata but does not modify the output.

    Writes to state.metadata:
        - guardrail_triggered: bool — True if the guardrail classified the output as harmful
        - guardrail_original_output: str — the original completion before replacement
        - guardrail_reason: str — classifier's reason line
        - guardrail_mode: str — the mode used ("replace" or "detect_only")
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Run the inner solver first.
        state = await inner(state, generate)

        completion = (state.output.completion or "").strip()
        user_input = state.input_text or ""

        # Run guardrail classifier on the original output.
        result = await _run_guardrail_check(
            completion=completion,
            user_input=user_input,
            guardrail_model=guardrail_model,
        )

        triggered = not result["is_safe"]

        state.metadata["guardrail_triggered"] = triggered
        state.metadata["guardrail_original_output"] = completion
        state.metadata["guardrail_reason"] = result["reason"]
        state.metadata["guardrail_mode"] = mode

        if triggered and mode == "replace":
            state.output = ModelOutput.from_content(
                model=state.output.model or "guardrail",
                content=SAFE_REFUSAL,
            )
            # Replace the last assistant message so scorers see the refusal.
            if state.messages:
                state.messages[-1] = ChatMessageAssistant(content=SAFE_REFUSAL)

        return state

    return solve
