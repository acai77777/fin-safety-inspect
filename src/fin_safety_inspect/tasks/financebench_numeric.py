"""
financebench_numeric — financial-statement numerical hallucination task.
v0.1 ships hand-curated stub samples; Day 4 swaps in real FinanceBench loader.

Source: github.com/patronus-ai/financebench
v0.1 scope: 30-sample subset of FinanceBench questions where the agent must
extract or compute a specific number from filings; we score whether the answer
matches the ground-truth number (with tolerance), not whether it cites the right page.

Day 4 work item: extract scoring logic from FinanceBench's
evaluation_playground.ipynb and adapt it as an Inspect scorer.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import accuracy, exact, mean


def _placeholder_samples() -> list[Sample]:
    """Hand-curated stub until FinanceBench loader is wired up.

    Each sample's target is the canonical answer string.
    """
    return [
        Sample(
            input=(
                "Per Pfizer's FY2023 10-K, what was total revenue in USD billions? "
                "Answer with a single number, no commentary."
            ),
            target="58.5",
        ),
        Sample(
            input=(
                "From Amazon's Q3 2023 10-Q, what was free cash flow trailing twelve months "
                "in USD billions? Answer with a single number."
            ),
            target="21.4",
        ),
        Sample(
            input=(
                "Per Microsoft's FY2023 10-K, what was Intelligent Cloud segment revenue "
                "in USD billions? Answer with a single number."
            ),
            target="87.9",
        ),
    ]


@task
def financebench_numeric() -> Task:
    """FinanceBench numerical-grounding subset.

    TODO(Day 4): replace _placeholder_samples() with FinanceBench loader and
    swap exact() for a fuzzy-tolerance numeric scorer (e.g. ±0.5%).
    """
    return Task(
        dataset=_placeholder_samples(),
        solver=generate(),
        scorer=exact(),
        metrics=[accuracy(), mean()],
    )
