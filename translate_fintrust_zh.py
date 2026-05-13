"""
One-shot translation: FinTrust safety/plain_question 100 prompts → Chinese.

Reads the pinned upstream JSON, translates each prompt with DeepSeek-v4-flash
(via JUDGE_BASE_URL / JUDGE_API_KEY), writes the result into
`src/fin_safety_inspect/datasets/_cache/fintrust_safety_zh_v0.4.json`.

Translation policy:
- temperature = 0 (deterministic)
- preserve English finance jargon in parentheses (hawala, ghost employees, etc.)
- preserve "I" → "我" / "my" → "我的" (first-person narrative, matches retail user voice)
- preserve question marks at end

Usage (after setting JUDGE_API_KEY + JUDGE_BASE_URL):
    python translate_fintrust_zh.py            # writes the cache JSON
    python translate_fintrust_zh.py --dry-run  # prints first 3 translations, no write

This is v0.4 work, not for installation. Re-run only when FINTRUST_COMMIT changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Reuse the upstream loader for the EN source.
from fin_safety_inspect.datasets.fintrust import (
    FINTRUST_COMMIT,
    load_fintrust_safety,
)

REPO = Path(__file__).resolve().parent
CACHE_DIR = REPO / "src" / "fin_safety_inspect" / "datasets" / "_cache"
ZH_CACHE_PATH = CACHE_DIR / f"fintrust_safety_zh_v0.4_{FINTRUST_COMMIT[:8]}.json"

TRANSLATION_PROMPT = """\
Translate the following English text into Simplified Chinese. Rules:
1. Preserve the first-person voice ("I", "my", "we") as "我", "我的", "我们".
2. Keep specialized English finance terms in parentheses after the Chinese term on first use (e.g., "哈瓦拉（hawala）", "幽灵员工（ghost employees）"). Common terms (e.g., "tax", "loan", "money") do not need parentheses.
3. Preserve the casual, retail-user tone — do not over-formalize.
4. Keep question marks and other punctuation.
5. Output ONLY the Chinese translation, no preface, no explanation.

ENGLISH:
{text}

CHINESE:"""


async def translate_one(llm, text: str) -> str:
    """Translate a single prompt via the configured judge backbone."""
    resp = await llm.ainvoke(TRANSLATION_PROMPT.format(text=text))
    content = str(resp.content) if hasattr(resp, "content") else str(resp)
    return content.strip()


async def main_async(dry_run: bool, model: str) -> None:
    from langchain_openai import ChatOpenAI

    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        print("ERROR: set JUDGE_BASE_URL + JUDGE_API_KEY (or OPENAI_*).", file=sys.stderr)
        sys.exit(2)
    print(f"Translation backbone: {model} @ {base_url}")
    print(f"Source commit:        {FINTRUST_COMMIT[:8]}")
    print(f"Cache output:         {ZH_CACHE_PATH.relative_to(REPO)}")
    print()

    pairs = load_fintrust_safety(use_remote=True)
    print(f"Source EN prompts: {len(pairs)} items")
    print()

    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)

    # Group by category for the output JSON, matching the upstream schema.
    zh_by_category: dict[str, list[str]] = {}
    for i, (cat, en) in enumerate(pairs):
        zh = await translate_one(llm, en)
        zh_by_category.setdefault(cat, []).append(zh)
        if dry_run and i < 3:
            print(f"[{i}] [{cat}]")
            print(f"  EN: {en}")
            print(f"  ZH: {zh}")
            print()
        else:
            print(f"[{i+1:3d}/{len(pairs)}] [{cat[:25]:25s}] {zh[:60]}…")

        if dry_run and i >= 2:
            print(f"\n(dry-run: stopped after 3 items, not writing cache)")
            return

    if dry_run:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ZH_CACHE_PATH.write_text(
        json.dumps(zh_by_category, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total = sum(len(v) for v in zh_by_category.values())
    print(f"\nWrote {total} translations across {len(zh_by_category)} categories")
    print(f"  to {ZH_CACHE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Translate only the first 3 prompts and print, don't write cache.")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Translation backbone (default deepseek-v4-flash).")
    args = parser.parse_args()
    asyncio.run(main_async(dry_run=args.dry_run, model=args.model))


if __name__ == "__main__":
    main()
