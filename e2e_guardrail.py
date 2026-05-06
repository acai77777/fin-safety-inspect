"""
End-to-end guardrail validation script.

Runs fintrust_safety with and without guardrail, then compares:
- Mode #3 failure rate reduction
- guardrail precision (did it correctly intercept harmful outputs?)
- guardrail impact on already-safe outputs (false positives)

Usage:
    $env:OPENAI_API_KEY  = "<key>"
    $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
    $env:JUDGE_API_KEY   = "<key>"   # same key for guardrail classifier
    $env:JUDGE_BASE_URL  = "https://api.deepseek.com/v1"
    python e2e_guardrail.py --samples 10 --model deepseek-chat
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent


def run_eval(guardrail: bool, model: str, max_samples: int) -> str:
    """Run a single eval and return the log file path. Output goes to logs/."""
    cmd = [
        "inspect", "eval",
        "src/fin_safety_inspect/tasks/fintrust_safety.py@fintrust_safety",
        "--model", f"openai/{model}",
        "-T", f"max_samples={max_samples}",
        "-T", f"guardrail={str(guardrail).lower()}",
        "-T", "guardrail_mode=replace",
        "--display", "none",
    ]
    print(f"  Running: {' '.join(cmd)}")
    print(f"  guardrail={'ON' if guardrail else 'OFF'}, model={model}, samples={max_samples}")

    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, encoding="utf-8")

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        sys.exit(1)

    # Inspect prints the log path to stdout. Find it.
    for line in result.stdout.split("\n"):
        line = line.strip()
        if "log:" in line.lower() or ".eval" in line:
            print(f"  Output: {line}")
        if "results" in line.lower():
            print(f"  {line}")

    # Find latest log file
    log_files = sorted(REPO.glob("logs/*.eval"), key=os.path.getmtime, reverse=True)
    if not log_files:
        print("  ERROR: No log file found after eval")
        sys.exit(1)

    log_path = str(log_files[0])
    print(f"  Log: {log_path}")
    return log_path


def analyze_log(log_path: str):
    """Extract guardrail stats from an eval log."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(log_path)

    if not log.samples:
        return {}

    total = len(log.samples)
    triggered = 0
    ff_total = 0  # both scorers fail
    kw_pass = 0
    judge_pass = 0
    guardrail_pass = 0

    for s in log.samples:
        scores = s.scores if s.scores else {}
        metadata = s.metadata if s.metadata else {}

        if metadata.get("guardrail_triggered"):
            triggered += 1

        for name, sc in (scores.items() if isinstance(scores, dict) else []):
            val = sc.value if hasattr(sc, "value") else sc
            passed = val == "C"
            if "keyword_refusal" in name and passed:
                kw_pass += 1
            elif "llm_judge_refusal" in name and passed:
                judge_pass += 1
            elif "mode3_guardrail" in name and passed:
                guardrail_pass += 1

        # Count ff: both keyword and judge fail
        kw_ok = False
        jd_ok = False
        for name, sc in (scores.items() if isinstance(scores, dict) else []):
            val = sc.value if hasattr(sc, "value") else sc
            passed = val == "C"
            if "keyword_refusal" in name:
                kw_ok = passed
            elif "llm_judge_refusal" in name:
                jd_ok = passed
        if not kw_ok and not jd_ok:
            ff_total += 1

    return {
        "total": total,
        "guardrail_triggered": triggered,
        "ff": ff_total,
        "kw_accuracy": kw_pass / total if total else 0,
        "judge_accuracy": judge_pass / total if total else 0,
        "guardrail_accuracy": guardrail_pass / total if total else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("E2E GUARDRAIL VALIDATION")
    print(f"  model: {args.model}, samples: {args.samples}")
    print("=" * 60)
    print()

    # Run WITHOUT guardrail first
    print("[1/2] Baseline WITHOUT guardrail")
    log_no_guard = run_eval(guardrail=False, model=args.model, max_samples=args.samples)
    stats_no = analyze_log(log_no_guard)
    print(f"  Results: ff={stats_no.get('ff', '?')}, "
          f"kw_acc={stats_no.get('kw_accuracy', 0):.2f}, "
          f"judge_acc={stats_no.get('judge_accuracy', 0):.2f}")
    print()

    # Run WITH guardrail
    print("[2/2] Baseline WITH guardrail")
    log_with_guard = run_eval(guardrail=True, model=args.model, max_samples=args.samples)
    stats_with = analyze_log(log_with_guard)
    print(f"  Results: triggered={stats_with.get('guardrail_triggered', '?')}, "
          f"ff={stats_with.get('ff', '?')}, "
          f"kw_acc={stats_with.get('kw_accuracy', 0):.2f}, "
          f"judge_acc={stats_with.get('judge_accuracy', 0):.2f}, "
          f"guardrail_acc={stats_with.get('guardrail_accuracy', 0):.2f}")
    print()

    # --- Comparison ---
    print("=" * 60)
    print("COMPARISON")
    print("-" * 60)
    ff_no = stats_no.get("ff", 0)
    ff_with = stats_with.get("ff", 0)
    triggered = stats_with.get("guardrail_triggered", 0)

    print(f"  ff WITHOUT guardrail:  {ff_no}")
    print(f"  ff WITH guardrail:     {ff_with}")
    print(f"  guardrail triggered:   {triggered} / {stats_with.get('total', 0)}")
    print(f"  ff reduction:          {ff_no - ff_with}")
    print()

    if ff_no > ff_with:
        print(f"  => Guardrail successfully intercepted {ff_no - ff_with} Mode #3 failure(s).")
    elif ff_no == 0 and ff_with == 0:
        print("  => No Mode #3 failures in baseline — guardrail had nothing to intercept.")
        print("     Try a larger sample size or a different model.")
    else:
        print("  => Guardrail did NOT reduce ff count. Possible causes:")
        print("     1. Guardrail classifier missed harmful outputs (false negative).")
        print("     2. Guardrail triggered but replacement didn't take effect.")
        print("     3. New failures introduced by guardrail (false positive → replacement wasn't counted).")

    print()
    print("=" * 60)
    print("Log files:")
    print(f"  WITHOUT: {log_no_guard}")
    print(f"  WITH:    {log_with_guard}")


if __name__ == "__main__":
    main()
