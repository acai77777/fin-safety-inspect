"""
v0.4 three-language baseline analyzer.

Reads the most recent eval logs for each (model, language) combination in
{DeepSeek-chat, Qwen-plus} x {EN, ZH, HAU}, prints:

1. 6-row headline accuracy table (model x language)
2. Per-category ff matrix (one block per language)
3. Mode #2 fp (soft-refusal) gap analysis across languages
4. Sample outputs for a few categories to expose the language patterns

EN logs are reused from v0.2.1; ZH and HAU are the v0.4 runs.

Usage:
    python analyze_three_languages.py
"""

from __future__ import annotations

import sys
import io
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from inspect_ai.log import list_eval_logs, read_eval_log

REPO = Path(__file__).resolve().parent

# (model substring, language) -> we look up the latest valid log
TARGETS = [
    ("deepseek-chat", "en"),
    ("deepseek-chat", "zh"),
    ("deepseek-chat", "hau"),
    ("qwen-plus", "en"),
    ("qwen-plus", "zh"),
    ("qwen-plus", "hau"),
]


def _is_match(log, model_substr: str, language: str) -> bool:
    """Does this log match the (model, language) we want?"""
    model = log.eval.model if log.eval else ""
    if model_substr not in model:
        return False
    # If this is an EN log from v0.2.1, the task didn't have a 'language' arg.
    # Treat missing language as "en" so we can still reuse those logs.
    task_args = log.eval.task_args if log.eval and hasattr(log.eval, "task_args") else {}
    log_lang = task_args.get("language", "en")
    if log_lang != language:
        return False
    # Skip 401-error / smoke logs
    if not log.samples or len(log.samples) < 90:
        return False
    return True


def find_log(model_substr: str, language: str) -> str | None:
    """Find the most recent log matching this (model, language)."""
    candidates = []
    for info in list_eval_logs(str(REPO / "logs")):
        log = read_eval_log(info)
        if _is_match(log, model_substr, language):
            candidates.append((info.mtime, info))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1].name).replace("file://", "")


def analyze_log(path: str) -> dict:
    """Per-category 4-cell + accuracy. Returns dict with totals + per_cat."""
    log = read_eval_log(path)
    per_cat = defaultdict(lambda: {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "n": 0})
    for s in log.samples:
        cat = s.metadata.get("category", "?") if s.metadata else "?"
        scores = s.scores if s.scores else {}
        kw = jd = None
        for n, sc in scores.items():
            v = sc.value if hasattr(sc, "value") else sc
            if "keyword_refusal" in n:
                kw = v == "C"
            elif "llm_judge_refusal" in n:
                jd = v == "C"
        if kw is None or jd is None:
            continue
        cell = ("pp" if (kw and jd) else
                "pf" if (kw and not jd) else
                "fp" if (not kw and jd) else "ff")
        per_cat[cat][cell] += 1
        per_cat[cat]["n"] += 1

    totals = {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "n": 0}
    for c in per_cat.values():
        for k in totals:
            totals[k] += c[k]
    kw_acc = (totals["pp"] + totals["pf"]) / totals["n"] if totals["n"] else 0
    jd_acc = (totals["pp"] + totals["fp"]) / totals["n"] if totals["n"] else 0
    return {
        "model": log.eval.model if log.eval else "?",
        "per_cat": dict(per_cat),
        "totals": totals,
        "kw_acc": kw_acc,
        "jd_acc": jd_acc,
        "log_path": path,
        "samples": log.samples,
    }


def main() -> None:
    # Step 1: locate the logs
    print("=" * 80)
    print("v0.4 three-language baseline")
    print("=" * 80)
    print()
    print("Logs located:")
    stats: dict[tuple[str, str], dict] = {}
    missing = []
    for model, lang in TARGETS:
        path = find_log(model, lang)
        if not path:
            missing.append((model, lang))
            print(f"  {model:18s} x {lang:4s}  MISSING")
            continue
        stats[(model, lang)] = analyze_log(path)
        print(f"  {model:18s} x {lang:4s}  {Path(path).name}")
    print()

    if missing:
        print(f"WARNING: {len(missing)} target(s) missing logs. Continuing with available data.")
        print()

    # Step 2: headline table
    print("=" * 80)
    print("HEADLINE: keyword + judge accuracy (per 100 samples)")
    print("=" * 80)
    print(f"{'Model':18s} | {'Lang':4s} | {'n':>4s} | {'kw':>6s} | {'judge':>6s} | {'fp':>4s} | {'ff':>4s} | {'Δ(j-kw)':>8s}")
    print("-" * 80)
    for model, lang in TARGETS:
        s = stats.get((model, lang))
        if not s:
            print(f"{model:18s} | {lang:4s} | {'-':>4s} | {'-':>6s} | {'-':>6s} | {'-':>4s} | {'-':>4s} | {'-':>8s}")
            continue
        t = s["totals"]
        delta = s["jd_acc"] - s["kw_acc"]
        print(f"{model:18s} | {lang:4s} | {t['n']:4d} | {s['kw_acc']:6.3f} | {s['jd_acc']:6.3f} | {t['fp']:4d} | {t['ff']:4d} | {delta:+8.3f}")
    print()

    # Step 3: per-category ff comparison across languages
    for model in ["deepseek-chat", "qwen-plus"]:
        print("=" * 80)
        print(f"PER-CATEGORY ff (Mode #3 candidates) for {model}")
        print("=" * 80)
        cats = set()
        for lang in ["en", "zh", "hau"]:
            s = stats.get((model, lang))
            if s:
                cats.update(s["per_cat"].keys())
        cats = sorted(cats)
        print(f"{'Category':38s} | {'EN ff':>5s} {'EN fp':>5s} | {'ZH ff':>5s} {'ZH fp':>5s} | {'HAU ff':>6s} {'HAU fp':>6s}")
        print("-" * 80)
        for cat in cats:
            cells = []
            for lang in ["en", "zh", "hau"]:
                s = stats.get((model, lang))
                if not s:
                    cells.append("   - ")
                    cells.append("   - ")
                else:
                    pc = s["per_cat"].get(cat, {"ff": 0, "fp": 0})
                    cells.append(f"{pc['ff']:5d}")
                    cells.append(f"{pc['fp']:5d}")
            print(f"{cat[:38]:38s} | {cells[0]} {cells[1]} | {cells[2]} {cells[3]} | {cells[4]} {cells[5]}")
        # Total row
        totals_str = []
        for lang in ["en", "zh", "hau"]:
            s = stats.get((model, lang))
            if not s:
                totals_str.append("   - ")
                totals_str.append("   - ")
            else:
                totals_str.append(f"{s['totals']['ff']:5d}")
                totals_str.append(f"{s['totals']['fp']:5d}")
        print("-" * 80)
        print(f"{'TOTAL':38s} | {totals_str[0]} {totals_str[1]} | {totals_str[2]} {totals_str[3]} | {totals_str[4]} {totals_str[5]}")
        print()

    # Step 4: HAU-specific diagnostic
    # When kw is low but judge is high for HAU, the keyword scorer is the issue.
    print("=" * 80)
    print("HAU DIAGNOSTIC: did models respond in Hausa or in English?")
    print("=" * 80)
    for model in ["deepseek-chat", "qwen-plus"]:
        s = stats.get((model, "hau"))
        if not s:
            continue
        print(f"\n{model} x HAU — sample first chars of 3 random outputs:")
        for i, sample in enumerate(s["samples"][:3]):
            comp = sample.output.completion if sample.output else ""
            print(f"  [{i}] {repr(comp)[:200]}")

    # Step 5: cross-language fp pattern
    print()
    print("=" * 80)
    print("Mode #2 (soft-refusal) gap across languages")
    print("=" * 80)
    for model in ["deepseek-chat", "qwen-plus"]:
        print(f"\n{model}:")
        for lang in ["en", "zh", "hau"]:
            s = stats.get((model, lang))
            if not s:
                continue
            t = s["totals"]
            gap = s["jd_acc"] - s["kw_acc"]
            print(f"  {lang:4s}: kw={s['kw_acc']:.3f}  judge={s['jd_acc']:.3f}  fp={t['fp']:3d}  gap={gap:+.3f}")


if __name__ == "__main__":
    main()
