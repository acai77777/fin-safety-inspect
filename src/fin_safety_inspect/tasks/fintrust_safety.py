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

from fin_safety_inspect.datasets.fintrust import (
    load_fintrust_safety,
    load_fintrust_safety_hau,
    load_fintrust_safety_zh,
)
from fin_safety_inspect.scorers import keyword_refusal, llm_judge_refusal, mode3_guardrail
from fin_safety_inspect.solvers.guardrail_solver import guardrail_solver


def _load_pairs(language: str, use_remote: bool) -> list[tuple[str, str]]:
    """Dispatch to the correct loader based on language."""
    if language == "en":
        return load_fintrust_safety(use_remote=use_remote)
    if language == "zh":
        return load_fintrust_safety_zh()
    if language == "hau":
        return load_fintrust_safety_hau(use_remote=use_remote)
    raise ValueError(f"unknown language {language!r}; expected one of 'en', 'zh', 'hau'")


def _samples(language: str, use_remote: bool, max_samples: int | None) -> list[Sample]:
    pairs = _load_pairs(language=language, use_remote=use_remote)
    if max_samples is not None:
        pairs = pairs[:max_samples]
    return [
        Sample(
            input=prompt,
            target="refused",
            metadata={
                "category": category,
                "expected_behavior": "refuse",
                "language": language,
            },
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
    language: str = "en",
) -> Task:
    """FinTrust safety / plain_question refusal task.

    Runs keyword_refusal + llm_judge_refusal + mode3_guardrail in parallel so
    each scorer inspects the same completion.

    Args:
        use_remote: Pull the full 100-sample dataset from upstream (default).
            Set False to use the 10-sample offline subset (en only).
        max_samples: Cap the number of samples (useful for smoke tests / cost control).
        judge_model: model name (no provider prefix) for the LLM-judge scorer.
            Defaults to "deepseek-v4-flash". Reads JUDGE_API_KEY / JUDGE_BASE_URL.
        guardrail: Wrap the solver with output-side guardrail (Mode #3 interception).
        guardrail_model: model name for the guardrail classifier.
        guardrail_mode: "replace" rewrites harmful outputs; "detect_only" logs only.
        language: which FinTrust prompt set to use. "en" is the upstream
            plain_question 100; "zh" is our v0.4 machine translation of "en"
            (run translate_fintrust_zh.py first); "hau" is upstream
            language_barrier (a SEPARATE 100-prompt set, not aligned 1-to-1
            with "en"). See docs/three-language-baseline-v0.4.md.
    """
    solver: Solver = generate()
    if guardrail:
        solver = guardrail_solver(
            inner=generate(),
            guardrail_model=guardrail_model,
            mode=guardrail_mode,  # type: ignore[arg-type]
        )

    return Task(
        dataset=_samples(language=language, use_remote=use_remote, max_samples=max_samples),
        solver=solver,
        scorer=[
            keyword_refusal(expect_refusal=True),
            llm_judge_refusal(judge_model=judge_model, expect_refusal=True),
            mode3_guardrail(guardrail_model=guardrail_model),
        ],
        metrics=[accuracy(), mean()],
    )
