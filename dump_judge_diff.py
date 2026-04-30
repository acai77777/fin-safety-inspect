"""
Diff keyword_refusal vs llm_judge_refusal scores from the latest .eval log.

Buckets every sample into one of four cells:
  - both_pass         (kw=PASS, judge=PASS) — they agree, refused
  - both_fail         (kw=FAIL, judge=FAIL) — they agree, Mode #1 or #3 candidates
  - kw_fail_judge_pass (kw=FAIL, judge=PASS) — Mode #2 verification: judge rescued
  - kw_pass_judge_fail (kw=PASS, judge=FAIL) — judge stricter than keyword (rare)

Then prints:
  1. The 4-cell summary (counts + per-cell accuracy delta)
  2. Every kw_fail_judge_pass sample verbatim (Mode #2 evidence)
  3. Every kw_pass_judge_fail sample verbatim (judge stricter — analyze why)
  4. Per-category 4-cell matrix
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

logs_dir = Path("logs")
log_files = sorted(logs_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
if not log_files:
    raise SystemExit("No .eval logs found in logs/")

latest = log_files[0]
print(f"Reading: {latest}\n")

log = read_eval_log(str(latest))
samples = log.samples or []
print(f"Task: {log.eval.task}  |  Total samples: {len(samples)}\n")


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


counts: Counter[str] = Counter()
buckets: dict[str, list[dict]] = defaultdict(list)
cat_counts: dict[str, Counter[str]] = defaultdict(Counter)

for i, s in enumerate(samples):
    if not s.scores:
        continue
    kw = s.scores.get("keyword_refusal")
    jg = s.scores.get("llm_judge_refusal")
    if kw is None or jg is None:
        continue

    bucket = _bucket(kw.value, jg.value)
    counts[bucket] += 1

    cat = (s.metadata or {}).get("category", "(unknown)")
    cat_counts[cat][bucket] += 1

    inp = s.input if isinstance(s.input, str) else str(s.input)
    completion = (s.output.completion if s.output else "") or ""
    judge_meta = jg.metadata or {}

    buckets[bucket].append(
        {
            "sample_idx": i,
            "category": cat,
            "input": inp,
            "completion": completion,
            "kw_explanation": kw.explanation or "",
            "judge_first_line": judge_meta.get("judge_first_line", ""),
            "judge_response": judge_meta.get("judge_response", ""),
            "judge_error": judge_meta.get("judge_error", ""),
        }
    )

# -- 4-cell summary ----------------------------------------------------------

total = sum(counts.values())
kw_acc = (counts["both_pass"] + counts["kw_pass_judge_fail"]) / total if total else 0
jg_acc = (counts["both_pass"] + counts["kw_fail_judge_pass"]) / total if total else 0

print(f"=== 4-cell agreement matrix ({total} samples scored on both) ===")
print(f"  both_pass          : {counts['both_pass']:>3}   (kw=PASS, judge=PASS)")
print(f"  both_fail          : {counts['both_fail']:>3}   (kw=FAIL, judge=FAIL — Mode #1 or #3)")
print(f"  kw_fail_judge_pass : {counts['kw_fail_judge_pass']:>3}   *Mode #2 candidates (judge rescued)*")
print(f"  kw_pass_judge_fail : {counts['kw_pass_judge_fail']:>3}   (judge stricter than keyword)")
print()
print(f"  keyword_refusal accuracy  : {kw_acc:.3f}")
print(f"  llm_judge_refusal accuracy: {jg_acc:.3f}")
print(f"  delta                     : {'+' if jg_acc >= kw_acc else ''}{jg_acc - kw_acc:.3f}")
print()

# -- Mode #2 candidates: kw FAIL, judge PASS --------------------------------

print(f"=== Mode #2 candidates: keyword=FAIL, judge=PASS ({counts['kw_fail_judge_pass']} samples) ===\n")
for entry in buckets["kw_fail_judge_pass"]:
    print(f"--- sample {entry['sample_idx']} (category: {entry['category']}) ---")
    print(f"INPUT : {entry['input'][:140]!r}")
    print(f"OUTPUT: {entry['completion'][:300]!r}")
    print(f"kw    : {entry['kw_explanation']}")
    print(f"judge : {entry['judge_first_line']!r}")
    print()

# -- judge stricter than keyword (rare but interesting) ---------------------

if counts["kw_pass_judge_fail"]:
    print(
        f"=== keyword=PASS, judge=FAIL ({counts['kw_pass_judge_fail']} samples — judge stricter) ===\n"
    )
    for entry in buckets["kw_pass_judge_fail"]:
        print(f"--- sample {entry['sample_idx']} (category: {entry['category']}) ---")
        print(f"INPUT : {entry['input'][:140]!r}")
        print(f"OUTPUT: {entry['completion'][:300]!r}")
        print(f"kw    : {entry['kw_explanation']}")
        print(f"judge : {entry['judge_first_line']!r}")
        print()

# -- judge errors (if any) --------------------------------------------------

errored = [e for bucket in buckets.values() for e in bucket if e["judge_error"]]
if errored:
    print(f"=== Judge call errors ({len(errored)} samples) ===\n")
    for entry in errored:
        print(f"  sample {entry['sample_idx']} (cat: {entry['category']}): {entry['judge_error']}")
    print()

# -- Per-category 4-cell matrix ---------------------------------------------

print("=== Per-category 4-cell matrix ===")
cats = sorted(cat_counts.keys())
width = max(len(c) for c in cats)
print(f"  {'category':<{width}}  pp  pf  fp  ff   (legend below)")
print(f"  {'-' * width}  --  --  --  --")
for cat in cats:
    b = cat_counts[cat]
    print(
        f"  {cat:<{width}}  "
        f"{b['both_pass']:>2}  "
        f"{b['kw_pass_judge_fail']:>2}  "
        f"{b['kw_fail_judge_pass']:>2}  "
        f"{b['both_fail']:>2}"
    )
print()
print("  legend:")
print("    pp = both_pass         (kw=PASS  judge=PASS)")
print("    pf = kw_pass_judge_fail (kw=PASS  judge=FAIL — judge stricter)")
print("    fp = kw_fail_judge_pass (kw=FAIL  judge=PASS — *Mode #2 rescued*)")
print("    ff = both_fail         (kw=FAIL  judge=FAIL — Mode #1 or #3)")
