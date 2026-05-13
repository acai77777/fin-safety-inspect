"""
v0.4 three-language baseline driver.

Runs fintrust_safety on (model, language) combinations to support the
EN / ZH / HAU comparison report. EN baselines are reused from v0.2.1 logs;
this script runs the missing 4: {DeepSeek-chat, Qwen-plus} x {ZH, HAU}.

Each run:
- Picks up env vars for the agent backbone (OPENAI_*) and judge (JUDGE_*)
- Sets task arg language=<zh|hau> via -T
- Writes a normal inspect_ai .eval log

Usage (smoke 5 samples first to verify the language wiring):
    python experiment_three_languages.py --smoke

Then the real run (4 baselines, ~15 min total, ~¥1.6):
    python experiment_three_languages.py --all

Env (DeepSeek run):
    OPENAI_API_KEY  = <deepseek-key>
    OPENAI_BASE_URL = https://api.deepseek.com/v1
    JUDGE_API_KEY   = <deepseek-key>     # same key OK
    JUDGE_BASE_URL  = https://api.deepseek.com/v1

Env (Qwen run, swap before launching --model qwen-plus):
    OPENAI_API_KEY  = <dashscope-key>
    OPENAI_BASE_URL = https://dashscope.aliyuncs.com/compatible-mode/v1
    # JUDGE_* stays on DeepSeek
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def preflight() -> None:
    """Verify env vars exist before launching."""
    missing = []
    for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "JUDGE_API_KEY", "JUDGE_BASE_URL"):
        if not os.environ.get(k):
            missing.append(k)
    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        sys.exit(2)
    print("Env OK:")
    print(f"  OPENAI_BASE_URL = {os.environ['OPENAI_BASE_URL']}")
    print(f"  JUDGE_BASE_URL  = {os.environ['JUDGE_BASE_URL']}")
    print()


def run_one(model: str, language: str, max_samples: int) -> str:
    """Run one (model, language) baseline. Returns log path."""
    cmd = [
        "inspect", "eval",
        "src/fin_safety_inspect/tasks/fintrust_safety.py@fintrust_safety",
        "--model", f"openai/{model}",
        "-T", f"max_samples={max_samples}",
        "-T", f"language={language}",
        "--display", "none",
    ]
    print(f"  Running: model={model}, language={language}, n={max_samples}")
    print(f"          {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"  ERROR (rc={result.returncode}):")
        print(f"  STDERR: {result.stderr[:1500]}")
        sys.exit(1)
    # Print last few lines of stdout (Inspect prints the accuracy table + log path here)
    for line in result.stdout.strip().split("\n")[-15:]:
        print(f"    {line}")
    # Find the latest log file
    log_files = sorted((REPO / "logs").glob("*.eval"), key=os.path.getmtime, reverse=True)
    return str(log_files[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 5-sample smoke on DS-chat × ZH only.")
    parser.add_argument("--all", action="store_true",
                        help="Run all 4 baselines: {DS-chat, Qwen-plus} × {ZH, HAU}. 100 samples each.")
    parser.add_argument("--single", metavar="MODEL:LANG",
                        help='Run one baseline: "deepseek-chat:zh" or "qwen-plus:hau".')
    parser.add_argument("--samples", type=int, default=100,
                        help="Override sample count (default 100 for --all, 5 for --smoke).")
    args = parser.parse_args()

    if not (args.smoke or args.all or args.single):
        parser.print_help()
        sys.exit(0)

    preflight()

    if args.smoke:
        print("[SMOKE] DeepSeek-chat × ZH × 5 samples (verify language wiring)")
        log = run_one(model="deepseek-chat", language="zh", max_samples=5)
        print(f"  log: {log}")
        print()
        print("If accuracy numbers look sensible (non-zero), proceed with --all.")
        return

    if args.single:
        if ":" not in args.single:
            print(f"ERROR: --single needs 'MODEL:LANG' format, got {args.single!r}", file=sys.stderr)
            sys.exit(2)
        model, lang = args.single.split(":", 1)
        log = run_one(model=model, language=lang, max_samples=args.samples)
        print(f"\nLog: {log}")
        return

    # --all
    plan = [
        ("deepseek-chat", "zh"),
        ("deepseek-chat", "hau"),
        ("qwen-plus", "zh"),
        ("qwen-plus", "hau"),
    ]
    print(f"[ALL] {len(plan)} baselines, n={args.samples} each")
    print()

    # Note: env may need to change between models. Print a reminder, don't auto-switch.
    current_base = os.environ.get("OPENAI_BASE_URL", "")
    inferred_provider = (
        "deepseek" if "deepseek" in current_base.lower()
        else "qwen" if "dashscope" in current_base.lower() or "qwen" in current_base.lower()
        else "?"
    )
    print(f"Current OPENAI_BASE_URL implies provider: {inferred_provider}")
    print("WARNING: re-set OPENAI_API_KEY + OPENAI_BASE_URL between provider switches.")
    print()

    results: list[tuple[str, str, str]] = []
    for i, (model, lang) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {model} × {lang}")
        log = run_one(model=model, language=lang, max_samples=args.samples)
        results.append((model, lang, log))
        print()

    print("=" * 60)
    print("All 4 baselines complete. Log files:")
    for model, lang, log in results:
        print(f"  {model:20s} × {lang:4s} -> {Path(log).name}")
    print()
    print("Next: run `python analyze_three_languages.py` to build the comparison report.")


if __name__ == "__main__":
    main()
