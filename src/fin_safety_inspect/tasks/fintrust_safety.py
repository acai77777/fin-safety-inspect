"""
fintrust_safety — financial-crime refusal task built on FinTrust safety/plain_question.

Source: github.com/HughieHu/FinTrust (EMNLP 2025), 100 prompts spanning 10 financial-crime
categories (Tax Evasion, Money Laundering, Insider Trading, Financial Fraud,
Counterfeit, Illegal Fund-raising, Illegal FX Trading, Credit Card Fraud,
Illegal Lending, Financial Identity Theft).

The full dataset is fetched on demand from upstream (cached locally). When offline,
falls back to a 10-sample hand-picked subset (one prompt per category).
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, mean
from inspect_ai.solver import Solver, generate

from fin_safety_inspect.datasets.fintrust import load_fintrust_safety
from fin_safety_inspect.scorers import keyword_refusal, llm_judge_refusal, mode3_guardrail
from fin_safety_inspect.solvers.guardrail_solver import guardrail_solver


def _samples(use_remote: bool, max_samples: int | None) -> list[Sample]:
    pairs = load_fintrust_safety(use_remote=use_remote)
    if max_samples is not None:
        pairs = pairs[:max_samples]
    return [
        Sample(
            input=prompt,
            target="refused",
            metadata={"category": category, "expected_behavior": "refuse"},
        )
        for category, prompt in pairs
    ]


@task
def fintrust_safety(
    use_remote: bool = True,
    max_samples: int | None = None,
    judge_model: str = "deepseek-v4-flash",
    guardrail: bool = False,
    guardrail_model: str = "deepseek-v4-flash",
    guardrail_mode: str = "replace",
) -> Task:
    """FinTrust safety / plain_question refusal task.

    Runs keyword_refusal + llm_judge_refusal + mode3_guardrail in parallel so
    each scorer inspects the same completion.

    Args:
        use_remote: Pull the full 100-sample dataset from upstream (default).
            Set False to use the 10-sample offline subset.
        max_samples: Cap the number of samples (useful for smoke tests / cost control).
        judge_model: model name (no provider prefix) for the LLM-judge scorer.
            Defaults to "deepseek-v4-flash". Reads JUDGE_API_KEY / JUDGE_BASE_URL.
        guardrail: Wrap the solver with output-side guardrail (Mode #3 interception).
        guardrail_model: model name for the guardrail classifier.
        guardrail_mode: "replace" rewrites harmful outputs; "detect_only" logs only.
    """
    solver: Solver = generate()
    if guardrail:
        solver = guardrail_solver(
            inner=generate(),
            guardrail_model=guardrail_model,
            mode=guardrail_mode,  # type: ignore[arg-type]
        )

    return Task(
        dataset=_samples(use_remote=use_remote, max_samples=max_samples),
        solver=solver,
        scorer=[
            keyword_refusal(expect_refusal=True),
            llm_judge_refusal(judge_model=judge_model, expect_refusal=True),
            mode3_guardrail(guardrail_model=guardrail_model),
        ],
        metrics=[accuracy(), mean()],
    )
