"""Dump the latest eval log: input, completion (first 300 chars), score, explanation."""

from __future__ import annotations

from pathlib import Path

from inspect_ai.log import read_eval_log

logs_dir = Path("logs")
log_files = sorted(logs_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
if not log_files:
    raise SystemExit("No .eval logs found in logs/")

latest = log_files[0]
print(f"Reading: {latest}\n")

log = read_eval_log(str(latest))
print(f"Task: {log.eval.task}  |  Samples: {len(log.samples or [])}\n")

for i, s in enumerate(log.samples or []):
    print(f"=== sample {i} ===")
    input_text = s.input if isinstance(s.input, str) else str(s.input)[:100]
    print(f"INPUT          : {input_text[:120]!r}")

    completion = s.output.completion if s.output else ""
    print(f"OUTPUT.completion: {(completion or '')[:300]!r}")

    if s.scores:
        for name, score in s.scores.items():
            print(f"SCORE[{name}].value      : {score.value!r}")
            print(f"SCORE[{name}].explanation: {score.explanation}")
            print(f"SCORE[{name}].metadata   : {score.metadata}")
    else:
        print("SCORE          : (none)")
    print()
