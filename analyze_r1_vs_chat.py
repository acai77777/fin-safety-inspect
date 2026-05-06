"""
Per-category R1 vs DeepSeek-chat comparison.

Analyzes the most recent eval log for each model, breaking down
keyword and judge scores by FinTrust category (10 categories).
Outputs a side-by-side 4-cell matrix per category + ff sample excerpts.
"""

import json
import os
import sys
from collections import defaultdict
from inspect_ai.log import read_eval_log, list_eval_logs

LOGS_DIR = "logs"


def _latest_log_for_model(model_substring: str, min_samples: int = 90):
    """Return the most recent eval log matching the model name."""
    candidates = []
    for info in list_eval_logs(LOGS_DIR):
        log = read_eval_log(info)
        model = log.eval.model if log.eval else ""
        if model_substring in model and len(log.samples or []) >= min_samples:
            candidates.append((info.mtime, info, log))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


def _scorer_name(score):
    """Extract readable scorer name from score metadata."""
    return score.scorer if hasattr(score, "scorer") else "?"


def _get_scores(sample):
    """Return (keyword_passed, judge_passed) or (None, None) if unscored."""
    scores = sample.scores if sample.scores else []
    kw = None
    jd = None
    for name, sc in scores.items() if isinstance(scores, dict) else []:
        if not isinstance(name, str):
            continue
        val = sc.value if hasattr(sc, "value") else sc
        if "keyword_refusal" in name:
            kw = val == "C"  # CORRECT
        elif "llm_judge_refusal" in name:
            jd = val == "C"
    return kw, jd


def _classify(kw: bool | None, jd: bool | None) -> str:
    """Classify into 4-cell: pp, pf, fp, ff."""
    if kw is None or jd is None:
        return "??"
    if kw and jd:
        return "pp"
    if kw and not jd:
        return "pf"
    if not kw and jd:
        return "fp"
    return "ff"


def analyze(model_a: str, model_b: str):
    """Side-by-side per-category comparison of two models."""
    log_a = _latest_log_for_model(model_a)
    log_b = _latest_log_for_model(model_b)

    if not log_a:
        print(f"ERROR: no log found for '{model_a}'")
        sys.exit(1)
    if not log_b:
        print(f"ERROR: no log found for '{model_b}'")
        sys.exit(1)

    print(f"Model A: {log_a.eval.model}  ({len(log_a.samples)} samples)")
    print(f"Model B: {log_b.eval.model}  ({len(log_b.samples)} samples)")
    print()

    # Collect per-sample results
    def collect(log):
        per_cat = defaultdict(lambda: {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "total": 0})
        ff_samples = []
        fp_samples = []
        for s in log.samples:
            cat = s.metadata.get("category", "?") if s.metadata else "?"
            kw, jd = _get_scores(s)
            cell = _classify(kw, jd)
            per_cat[cat][cell] += 1
            per_cat[cat]["total"] += 1
            if cell == "ff":
                ff_samples.append((cat, s.input, s.messages))
            if cell == "fp":
                fp_samples.append((cat, s.input, s.messages))
        return per_cat, ff_samples, fp_samples

    cat_a, ff_a, fp_a = collect(log_a)
    cat_b, ff_b, fp_b = collect(log_b)

    # All categories sorted
    all_cats = sorted(set(list(cat_a.keys()) + list(cat_b.keys())))

    # --- Per-category table ---
    print("=" * 100)
    print(f"{'Category':30s} | {'kw_A':>5s} {'jd_A':>5s} {'ff_A':>5s} {'fp_A':>5s} | {'kw_B':>5s} {'jd_B':>5s} {'ff_B':>5s} {'fp_B':>5s} | {'ff_diff':>6s}")
    print("-" * 100)

    totals_a = {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "total": 0}
    totals_b = {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "total": 0}

    for cat in all_cats:
        ca = cat_a.get(cat, {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "total": 0})
        cb = cat_b.get(cat, {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "total": 0})
        n_a = ca["total"]
        n_b = cb["total"]

        kw_a = (ca["pp"] + ca["pf"]) / n_a if n_a else 0
        jd_a = (ca["pp"] + ca["fp"]) / n_a if n_a else 0
        kw_b = (cb["pp"] + cb["pf"]) / n_b if n_b else 0
        jd_b = (cb["pp"] + cb["fp"]) / n_b if n_b else 0
        ff_diff = cb["ff"] - ca["ff"]

        marker = " <<<" if abs(ff_diff) >= 2 else ""
        print(
            f"{cat[:30]:30s} | {kw_a:5.2f} {jd_a:5.2f} {ca['ff']:5d} {ca['fp']:5d} | "
            f"{kw_b:5.2f} {jd_b:5.2f} {cb['ff']:5d} {cb['fp']:5d} | {ff_diff:+6d}{marker}"
        )

        for k in totals_a:
            totals_a[k] += ca[k]
        for k in totals_b:
            totals_b[k] += cb[k]

    # Totals row
    print("-" * 100)
    kw_a = (totals_a["pp"] + totals_a["pf"]) / totals_a["total"]
    jd_a = (totals_a["pp"] + totals_a["fp"]) / totals_a["total"]
    kw_b = (totals_b["pp"] + totals_b["pf"]) / totals_b["total"]
    jd_b = (totals_b["pp"] + totals_b["fp"]) / totals_b["total"]
    ff_diff = totals_b["ff"] - totals_a["ff"]
    print(
        f"{'TOTAL':30s} | {kw_a:5.2f} {jd_a:5.2f} {totals_a['ff']:5d} {totals_a['fp']:5d} | "
        f"{kw_b:5.2f} {jd_b:5.2f} {totals_b['ff']:5d} {totals_b['fp']:5d} | {ff_diff:+6d}"
    )

    # --- Mode #2 gap (fp — soft refusals) ---
    print()
    print("=" * 60)
    print("Mode #2 (soft refusal) gap: fp = keyword FAIL but judge PASS")
    print(f"  {model_a}: fp={totals_a['fp']}  (judge - kw = {jd_a - kw_a:+.3f})")
    print(f"  {model_b}: fp={totals_b['fp']}  (judge - kw = {jd_b - kw_b:+.3f})")

    # --- ff sample excerpts ---
    print()
    print("=" * 60)
    print(f"{model_a} ff samples ({len(ff_a)}):")
    for cat, inp, msgs in ff_a:
        resp = _last_ai_content(msgs)
        print(f"  [{cat}] {inp[:120]}...")
        print(f"      -> {resp[:200]}...")
        print()

    print()
    print("=" * 60)
    print(f"{model_b} ff samples ({len(ff_b)}):")
    for cat, inp, msgs in ff_b:
        resp = _last_ai_content(msgs)
        print(f"  [{cat}] {inp[:120]}...")
        print(f"      -> {resp[:200]}...")
        print()

    # --- Unique ff: topics where one model fails but the other doesn't ---
    print("=" * 60)
    print("Topics where R1 ff > chat ff (R1 specifically worse):")
    for cat in all_cats:
        ca = cat_a.get(cat, {"ff": 0})
        cb = cat_b.get(cat, {"ff": 0})
        if cb["ff"] > ca["ff"]:
            print(f"  {cat}: R1={cb['ff']} vs chat={ca['ff']}  (diff: +{cb['ff'] - ca['ff']})")

    print()
    print("Topics where chat ff > R1 ff (chat specifically worse):")
    for cat in all_cats:
        ca = cat_a.get(cat, {"ff": 0})
        cb = cat_b.get(cat, {"ff": 0})
        if ca["ff"] > cb["ff"]:
            print(f"  {cat}: chat={ca['ff']} vs R1={cb['ff']}  (diff: +{ca['ff'] - cb['ff']})")

    # --- fp analysis: soft refusals per category ---
    print()
    print("=" * 60)
    print("Soft refusals (fp) per category:")
    for cat in all_cats:
        ca = cat_a.get(cat, {"fp": 0})
        cb = cat_b.get(cat, {"fp": 0})
        if ca["fp"] > 0 or cb["fp"] > 0:
            print(f"  {cat[:30]:30s}  {model_a}: {ca['fp']:2d}  {model_b}: {cb['fp']:2d}")


def _last_ai_content(messages) -> str:
    """Extract last AI message content from a list of messages."""
    if not messages:
        return ""
    for m in reversed(messages):
        if isinstance(m, dict):
            if m.get("role") == "assistant":
                return str(m.get("content", ""))[:300]
        elif hasattr(m, "role") and m.role == "assistant":
            return str(getattr(m, "content", ""))[:300]
    return ""


if __name__ == "__main__":
    # R1 vs DeepSeek-chat
    analyze("deepseek/deepseek-r1", "deepseek-chat")
