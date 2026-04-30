"""
Compare cross-model baselines from logs/.

Reads .eval logs, groups by the model field (most recent per model wins),
and prints:
  1. Per-model 2-scorer accuracy table (keyword vs llm-judge).
  2. Per-model 4-cell (pp/pf/fp/ff) agreement matrix.
  3. Per-model "ff" rate by category (where Mode #1 and Mode #3 hide).

Run after you've completed all 4 cross-model baseline runs (see RUN.md).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

LOGS = Path("logs")
log_files = sorted(LOGS.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
if not log_files:
    raise SystemExit("No .eval logs found in logs/")


def _model_label(log) -> str:
    """Strip 'openai/' / 'anthropic/' prefix so labels are readable."""
    raw = log.eval.model or "(unknown)"
    if "/" in raw:
        return raw.split("/", 1)[-1]
    return raw


# Group: most recent log per model.
by_model: dict[str, tuple[Path, object]] = {}
for f in log_files:
    log = read_eval_log(str(f))
    label = _model_label(log)
    if label not in by_model:
        by_model[label] = (f, log)

if not by_model:
    raise SystemExit("Could not extract any model from logs/.")

print(f"Found {len(by_model)} model(s) in logs/:")
for label, (path, _) in by_model.items():
    print(f"  {label:40s}  {path.name}")
print()


def _bucket(kw_value: str, jg_value: str) -> str:
    kw_pass = kw_value == "C"
    jg_pass = jg_value == "C"
    if kw_pass and jg_pass:
        return "both_pass"
    if not kw_pass and not jg_pass:
        return "both_fail"
    if not kw_pass and jg_pass:
        return "kw_fail_judge_pass"
    return "kw_pass_judge_fail"


# Compute per-model stats.
model_stats: dict[str, dict] = {}
for label, (_, log) in by_model.items():
    samples = log.samples or []
    bucket_counts: Counter[str] = Counter()
    cat_ff: Counter[str] = Counter()  # Mode #1/#3 candidate counts per category
    n = 0
    for s in samples:
        if not s.scores:
            continue
        kw = s.scores.get("keyword_refusal")
        jg = s.scores.get("llm_judge_refusal")
        if kw is None or jg is None:
            continue
        b = _bucket(kw.value, jg.value)
        bucket_counts[b] += 1
        n += 1
        if b == "both_fail":
            cat = (s.metadata or {}).get("category", "(unknown)")
            cat_ff[cat] += 1

    if n == 0:
        continue
    kw_acc = (bucket_counts["both_pass"] + bucket_counts["kw_pass_judge_fail"]) / n
    jg_acc = (bucket_counts["both_pass"] + bucket_counts["kw_fail_judge_pass"]) / n
    model_stats[label] = {
        "n": n,
        "buckets": bucket_counts,
        "kw_acc": kw_acc,
        "jg_acc": jg_acc,
        "cat_ff": cat_ff,
    }

# Sort models alphabetically for stable output.
models = sorted(model_stats.keys())
labels_w = max(len(m) for m in models)


# -- Table 1: accuracy per model -------------------------------------------

print("=== Per-model accuracy ===")
print(f"  {'model':<{labels_w}}  n   kw_acc  jg_acc  delta")
print(f"  {'-' * labels_w}  --- ------  ------  ------")
for m in models:
    s = model_stats[m]
    delta = s["jg_acc"] - s["kw_acc"]
    print(
        f"  {m:<{labels_w}}  {s['n']:>3}  "
        f"{s['kw_acc']:.3f}   {s['jg_acc']:.3f}   "
        f"{'+' if delta >= 0 else ''}{delta:.3f}"
    )
print()


# -- Table 2: 4-cell distribution per model --------------------------------

print("=== Per-model 4-cell distribution ===")
print(f"  {'model':<{labels_w}}   pp   pf   fp   ff   (legend below)")
print(f"  {'-' * labels_w}  ---  ---  ---  ---")
for m in models:
    b = model_stats[m]["buckets"]
    print(
        f"  {m:<{labels_w}}  "
        f"{b['both_pass']:>3}  "
        f"{b['kw_pass_judge_fail']:>3}  "
        f"{b['kw_fail_judge_pass']:>3}  "
        f"{b['both_fail']:>3}"
    )
print()
print("  legend:")
print("    pp = both_pass            (kw=PASS  judge=PASS)")
print("    pf = kw_pass_judge_fail   (kw=PASS  judge=FAIL — judge stricter)")
print("    fp = kw_fail_judge_pass   (kw=FAIL  judge=PASS — Mode #2 rescued)")
print("    ff = both_fail            (kw=FAIL  judge=FAIL — Mode #1 or #3)")
print()


# -- Table 3: per-category ff rate per model -------------------------------

print("=== Per-model per-category 'ff' counts (Mode #1/#3 candidates) ===")
all_cats = sorted({c for s in model_stats.values() for c in s["cat_ff"]})
if not all_cats:
    print("  (no ff samples in any model — both scorers passed everything)")
else:
    cats_w = max(len(c) for c in all_cats)
    header = f"  {'category':<{cats_w}}  " + "  ".join(f"{m[:10]:>10}" for m in models)
    print(header)
    print(f"  {'-' * cats_w}  " + "  ".join("-" * 10 for _ in models))
    for cat in all_cats:
        row = f"  {cat:<{cats_w}}  " + "  ".join(
            f"{model_stats[m]['cat_ff'][cat]:>10}" for m in models
        )
        print(row)

print()
print("Use this table to decide v0.2.1 trace-scorer priority:")
print("  - If 'ff' is concentrated on one model, the failure mode is model-specific")
print("    (likely Mode #1 = backbone's prompt-following style).")
print("  - If 'ff' is uniform across models, the failure mode is task/agent-specific")
print("    (likely Mode #3 = the prompts themselves are hard, model can't refuse cleanly).")
