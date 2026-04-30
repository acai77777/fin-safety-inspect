"""Dump only the false-negative samples (expected refusal but agent did not refuse).

Reads the latest .eval log; prints each FAIL with category, input, output start,
and explanation. Ends with a per-category breakdown of FN counts.
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

failures: list[tuple[int, str, str, str, str]] = []
total_by_cat: Counter[str] = Counter()
fail_by_cat: Counter[str] = Counter()

for i, s in enumerate(samples):
    category = (s.metadata or {}).get("category", "(unknown)")
    total_by_cat[category] += 1

    if not s.scores:
        continue
    score = s.scores.get("keyword_refusal")
    if score is None or score.value != "I":
        continue

    fail_by_cat[category] += 1
    input_text = s.input if isinstance(s.input, str) else str(s.input)
    completion = (s.output.completion if s.output else "") or ""
    failures.append((i, category, input_text, completion, score.explanation or ""))

print(f"=== FALSE NEGATIVES: {len(failures)} / {len(samples)} ===\n")

for idx, (i, cat, inp, out, expl) in enumerate(failures, 1):
    print(f"--- FAIL #{idx} (sample {i}, category: {cat}) ---")
    print(f"INPUT : {inp[:140]!r}")
    print(f"OUTPUT: {out[:300]!r}")
    print()

print("=== FALSE NEGATIVE RATE BY CATEGORY ===")
rows = []
for cat in sorted(total_by_cat.keys()):
    fn = fail_by_cat[cat]
    tot = total_by_cat[cat]
    rate = fn / tot if tot else 0.0
    rows.append((cat, fn, tot, rate))

rows.sort(key=lambda r: r[3], reverse=True)
width = max(len(r[0]) for r in rows)
for cat, fn, tot, rate in rows:
    bar = "█" * int(rate * 20)
    print(f"  {cat:<{width}}  {fn:>2}/{tot:<2}  ({rate:5.1%})  {bar}")

print()
print(f"Overall FN rate: {len(failures)}/{len(samples)} = {len(failures) / len(samples):.1%}")
