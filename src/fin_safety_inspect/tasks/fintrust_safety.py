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
from inspect_ai.solver import generate

from fin_safety_inspect.datasets.fintrust import load_fintrust_safety
from fin_safety_inspect.scorers import keyword_refusal


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
) -> Task:
    """FinTrust safety / plain_question refusal task.

    Args:
        use_remote: Pull the full 100-sample dataset from upstream (default).
            Set False to use the 10-sample offline subset.
        max_samples: Cap the number of samples (useful for smoke tests / cost control).
    """
    return Task(
        dataset=_samples(use_remote=use_remote, max_samples=max_samples),
        solver=generate(),
        scorer=keyword_refusal(expect_refusal=True),
        metrics=[accuracy(), mean()],
    )
