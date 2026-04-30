"""Dump failure samples from a baseline log.

Behaviors:
  - With no args: reads the latest .eval log.
  - With one arg: substring-matches log.eval.model (e.g. "z-ai/glm-4.5",
    "deepseek-chat", "qwen") and uses the most recent matching log.
  - Auto-detects whether the log has both keyword + judge scorers:
      * Dual scorer  → dumps "both_fail" samples (kw=FAIL AND judge=FAIL,
        a.k.a. the `ff` cell — Mode #1 / Mode #3 hiding spots).
      * Single scorer (v0.1 logs) → dumps keyword=FAIL samples (the FN list).

Run:
    python dump_failures.py                       # latest log, auto-mode
    python dump_failures.py "z-ai/glm-4.5"        # GLM's most recent log
    python dump_failures.py "deepseek-chat"       # DS-chat's most recent log
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

LOGS = Path("logs")
target_model = sys.argv[1] if len(sys.argv) > 1 else None

log_files = sorted(LOGS.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
if not log_files:
    raise SystemExit("No .eval logs found in logs/")

# Find first (most recent) log whose eval.model matches the filter.
chosen: tuple[Path, object] | None = None
for f in log_files:
    log = read_eval_log(str(f))
    model = log.eval.model or ""
    if target_model is None or target_model in model:
        chosen = (f, log)
        break

if chosen is None:
    print(f"No .eval log matched model filter {target_model!r}.")
    print("Available models in logs/:")
    seen: set[str] = set()
    for f in log_files:
        log = read_eval_log(str(f))
        m = log.eval.model or "(unknown)"
        if m not in seen:
            seen.add(m)
            print(f"  - {m}  ({f.name})")
    raise SystemExit(1)

f, log = chosen
samples = log.samples or []
print(f"Reading: {f.name}")
print(f"Model:   {log.eval.model}")
print(f"Samples: {len(samples)}")

# Detect whether this log has the v0.2 dual scorer setup.
has_judge = bool(
    samples
    and samples[0].scores
    and "llm_judge_refusal" in samples[0].scores
)
mode_label = "BOTH-FAIL (ff)" if has_judge else "KEYWORD FAIL (FN)"
print(f"Mode:    {mode_label}\n")

failures: list[tuple[int, str, str, str, str, str]] = []
total_by_cat: Counter[str] = Counter()
fail_by_cat: Counter[str] = Counter()

for i, s in enumerate(samples):
    category = (s.metadata or {}).get("category", "(unknown)")
    total_by_cat[category] += 1

    if not s.scores:
        continue

    kw = s.scores.get("keyword_refusal")
    if kw is None:
        continue

    if has_judge:
        jg = s.scores.get("llm_judge_refusal")
        if not (kw.value == "I" and jg is not None and jg.value == "I"):
            continue
    else:
        jg = None
        if kw.value != "I":
            continue

    fail_by_cat[category] += 1
    inp = s.input if isinstance(s.input, str) else str(s.input)
    completion = (s.output.completion if s.output else "") or ""
    judge_first_line = ""
    if jg is not None and jg.metadata:
        judge_first_line = jg.metadata.get("judge_first_line", "") or ""
    failures.append(
        (
            i,
            category,
            inp,
            completion,
            kw.explanation or "",
            judge_first_line,
        )
    )

print(f"=== {mode_label}: {len(failures)} / {len(samples)} ===\n")

for idx, (i, cat, inp, out, kw_expl, jg_line) in enumerate(failures, 1):
    print(f"--- FAIL #{idx} (sample {i}, category: {cat}) ---")
    print(f"INPUT : {inp[:140]!r}")
    print(f"OUTPUT: {out[:300]!r}")
    if has_judge:
        print(f"judge : {jg_line!r}")
    print()

print(f"=== {mode_label} BY CATEGORY ===")
rows = []
for cat in sorted(total_by_cat.keys()):
    fn = fail_by_cat[cat]
    tot = total_by_cat[cat]
    rate = fn / tot if tot else 0.0
    rows.append((cat, fn, tot, rate))

rows.sort(key=lambda r: r[3], reverse=True)
width = max(len(r[0]) for r in rows) if rows else 30
for cat, fn, tot, rate in rows:
    bar = "█" * int(rate * 20)
    print(f"  {cat:<{width}}  {fn:>2}/{tot:<2}  ({rate:5.1%})  {bar}")

if total_by_cat:
    overall_rate = len(failures) / sum(total_by_cat.values())
    print(f"\nOverall {mode_label.lower()} rate: {len(failures)}/{sum(total_by_cat.values())} = {overall_rate:.1%}")
