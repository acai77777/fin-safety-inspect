"""
R1 mini-study, ablation B — does a safety-first system prompt fix R1's
"helpful explainer" failure mode?

Hypothesis (from v0.2.1 cross-model finding + v0.3 per-category analysis):
R1's reasoning RLHF biases it toward "decompose and explain" over "refuse early."
Adding an explicit safety-first system prompt may compress that bias by giving
R1 a stronger refusal prior before reasoning starts.

Experimental variables:
  - Control:    R1 + no system prompt (replays existing v0.2.1 R1 log)
  - Treatment:  R1 + R1_SAFETY_SYSTEM_PROMPT (this script runs it fresh)
  Everything else is held constant: same dataset commit, same judge backbone,
  same task, same scorer set.

Metrics:
  - ff (both scorers FAIL): primary outcome — R1's "helpful explainer" failures
  - fp (kw FAIL, judge PASS): R1's signature soft-refusal rate
  - judge accuracy: overall safety
  - per-category breakdown: is the gain concentrated on Money Laundering /
    Illegal Lending (the categories where R1 was specifically worse than chat)?

Usage:
    $env:OPENAI_API_KEY  = "<openrouter-key>"
    $env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    $env:JUDGE_API_KEY   = "<deepseek-key>"
    $env:JUDGE_BASE_URL  = "https://api.deepseek.com/v1"
    python experiment_r1_safety_prompt.py --samples 100
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONTROL_MODEL_SUBSTR = "deepseek/deepseek-r1"  # for finding the existing v0.2.1 R1 log


def preflight_env_check(model: str) -> None:
    """Send a 1-token probe to OPENAI_BASE_URL to verify env vars route correctly.

    Catches the failure mode where OPENAI_BASE_URL is left over from a previous
    experiment (e.g. DeepSeek) but the model name expects OpenRouter — this
    silently 401s every sample for an entire 100-prompt run.

    Aborts the script if the probe doesn't return 200.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    judge_base = os.environ.get("JUDGE_BASE_URL", "")
    judge_key = os.environ.get("JUDGE_API_KEY", "")

    print("Preflight environment check:")
    print(f"  OPENAI_BASE_URL = {base_url or '(unset)'}")
    print(f"  OPENAI_API_KEY  = {'set (' + str(len(api_key)) + ' chars)' if api_key else '(unset)'}")
    print(f"  JUDGE_BASE_URL  = {judge_base or '(unset, falls back to OPENAI_BASE_URL)'}")
    print(f"  JUDGE_API_KEY   = {'set' if judge_key else '(unset)'}")

    if not api_key:
        print("\nERROR: OPENAI_API_KEY is not set. Aborting.")
        sys.exit(2)
    if not base_url:
        print("\nERROR: OPENAI_BASE_URL is not set. Aborting.")
        sys.exit(2)

    # Heuristic: model "deepseek/deepseek-r1" needs OpenRouter; "deepseek-chat" needs DeepSeek direct.
    if "/" in model and "openrouter" not in base_url.lower():
        print(f"\nWARNING: model '{model}' looks like an OpenRouter model id (contains '/'),")
        print(f"  but OPENAI_BASE_URL is '{base_url}'. Expected 'https://openrouter.ai/api/v1'.")
        print("  If this is wrong, abort with Ctrl-C and reset env. Continuing in 5s…")
        import time
        time.sleep(5)

    # Live probe: 1-token completion via langchain ChatOpenAI on the same env.
    print("  Probing endpoint…", end=" ", flush=True)
    try:
        from langchain_openai import ChatOpenAI
        # Strip the "openai:" prefix that init_chat_model uses; ChatOpenAI takes the bare model id.
        bare_model = model.split(":", 1)[1] if model.startswith("openai:") else model
        llm = ChatOpenAI(model=bare_model, base_url=base_url, api_key=api_key, max_tokens=1)
        resp = llm.invoke("hi")
        _ = str(resp.content)[:10]
        print("OK")
    except Exception as e:
        print("FAILED")
        print(f"\nERROR: probe request to {base_url} returned: {type(e).__name__}: {str(e)[:300]}")
        print("Fix env vars and retry. Aborting before burning a 100-sample run.")
        sys.exit(2)
    print()


def find_latest_log_for_model(model_substring: str, min_samples: int = 90) -> str | None:
    """Find the most recent .eval log whose model field contains the substring."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    candidates = []
    for info in list_eval_logs(str(REPO / "logs")):
        log = read_eval_log(info)
        model = log.eval.model if log.eval else ""
        if model_substring in model and len(log.samples or []) >= min_samples:
            # Skip logs that already had a system prompt applied (post-experiment runs)
            plan_steps = log.plan.steps if log.plan else []
            has_safety_prompt = any(
                "safety_prompt" in str(step.params).lower() if hasattr(step, "params") else False
                for step in plan_steps
            )
            if not has_safety_prompt:
                candidates.append((info.mtime, info))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return str(candidates[0][1].name).replace("file://", "")


def run_treatment_eval(model: str, max_samples: int) -> str:
    """Run R1 + safety prompt and return the log file path."""
    cmd = [
        "inspect", "eval",
        "src/fin_safety_inspect/tasks/fintrust_safety.py@fintrust_safety",
        "--model", f"openai/{model}",
        "-T", f"max_samples={max_samples}",
        "--solver", "src/fin_safety_inspect/solvers/langgraph_solver.py@langgraph",
        "-S", "graph=fin_safety_inspect.examples.mock_helper:build_graph_with_safety_prompt",
        "-S", f"model=openai:{model}",
        "--display", "none",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:1000]}")
        sys.exit(1)
    log_files = sorted((REPO / "logs").glob("*.eval"), key=os.path.getmtime, reverse=True)
    return str(log_files[0])


def analyze_log(path: str) -> dict:
    """Per-category 4-cell + accuracy from a single log."""
    from inspect_ai.log import read_eval_log

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
    }


def print_comparison(control: dict, treatment: dict):
    """Side-by-side per-category comparison."""
    print("=" * 100)
    print(f"Control:    {control['model']}  (no system prompt)")
    print(f"Treatment:  {treatment['model']}  (R1_SAFETY_SYSTEM_PROMPT)")
    print()

    print(f"{'Category':35s} | {'C kw':>5s} {'C jd':>5s} {'C ff':>5s} {'C fp':>5s} | {'T kw':>5s} {'T jd':>5s} {'T ff':>5s} {'T fp':>5s} | {'ff Δ':>5s} {'fp Δ':>5s}")
    print("-" * 100)

    all_cats = sorted(set(list(control["per_cat"].keys()) + list(treatment["per_cat"].keys())))
    for cat in all_cats:
        cc = control["per_cat"].get(cat, {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "n": 0})
        tc = treatment["per_cat"].get(cat, {"pp": 0, "pf": 0, "fp": 0, "ff": 0, "n": 0})
        n_c = cc["n"] or 1
        n_t = tc["n"] or 1
        kw_c = (cc["pp"] + cc["pf"]) / n_c
        jd_c = (cc["pp"] + cc["fp"]) / n_c
        kw_t = (tc["pp"] + tc["pf"]) / n_t
        jd_t = (tc["pp"] + tc["fp"]) / n_t
        ff_diff = tc["ff"] - cc["ff"]
        fp_diff = tc["fp"] - cc["fp"]
        marker = " <" if ff_diff < 0 else ""
        print(
            f"{cat[:35]:35s} | {kw_c:5.2f} {jd_c:5.2f} {cc['ff']:5d} {cc['fp']:5d} | "
            f"{kw_t:5.2f} {jd_t:5.2f} {tc['ff']:5d} {tc['fp']:5d} | {ff_diff:+5d} {fp_diff:+5d}{marker}"
        )

    print("-" * 100)
    ct = control["totals"]
    tt = treatment["totals"]
    print(
        f"{'TOTAL':35s} | {control['kw_acc']:5.2f} {control['jd_acc']:5.2f} {ct['ff']:5d} {ct['fp']:5d} | "
        f"{treatment['kw_acc']:5.2f} {treatment['jd_acc']:5.2f} {tt['ff']:5d} {tt['fp']:5d} | "
        f"{tt['ff'] - ct['ff']:+5d} {tt['fp'] - ct['fp']:+5d}"
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print(f"  ff:  control={ct['ff']}  →  treatment={tt['ff']}  (Δ {tt['ff'] - ct['ff']:+d})")
    print(f"  fp:  control={ct['fp']}  →  treatment={tt['fp']}  (Δ {tt['fp'] - ct['fp']:+d})")
    print(f"  kw:  {control['kw_acc']:.3f}  →  {treatment['kw_acc']:.3f}  (Δ {treatment['kw_acc'] - control['kw_acc']:+.3f})")
    print(f"  jd:  {control['jd_acc']:.3f}  →  {treatment['jd_acc']:.3f}  (Δ {treatment['jd_acc'] - control['jd_acc']:+.3f})")
    print()
    if tt["ff"] < ct["ff"]:
        print(f"  => Safety prompt reduced ff by {ct['ff'] - tt['ff']}.")
    elif tt["ff"] == ct["ff"]:
        print("  => Safety prompt had no effect on ff count.")
    else:
        print(f"  => Safety prompt INCREASED ff by {tt['ff'] - ct['ff']}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek/deepseek-r1",
                        help="Treatment model (must be R1 unless rerunning the control)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--control-log", default=None,
                        help="Path to control log; if omitted, uses latest non-safety-prompt R1 log")
    args = parser.parse_args()

    print("=" * 60)
    print("R1 SAFETY-PROMPT ABLATION (mini-study, ablation B)")
    print(f"  treatment model: {args.model}")
    print(f"  samples:         {args.samples}")
    print("=" * 60)
    print()

    # Locate control log (existing v0.2.1 R1 baseline)
    control_log = args.control_log or find_latest_log_for_model(CONTROL_MODEL_SUBSTR, min_samples=args.samples - 5)
    if not control_log:
        print(f"ERROR: no existing R1 log found with ≥{args.samples - 5} samples.")
        print("Run a baseline R1 run first (run_smoke.ps1 -Model deepseek/deepseek-r1) to create the control.")
        sys.exit(1)
    print(f"[1/2] Using existing control log:")
    print(f"      {control_log}")
    control_stats = analyze_log(control_log)
    print(f"      ff={control_stats['totals']['ff']}, fp={control_stats['totals']['fp']}, "
          f"kw={control_stats['kw_acc']:.3f}, jd={control_stats['jd_acc']:.3f}")
    print()

    # Run treatment (R1 + safety prompt) — preflight env check first to avoid burning a run on 401.
    preflight_env_check(args.model)

    print(f"[2/2] Running treatment (R1 + R1_SAFETY_SYSTEM_PROMPT)…")
    treatment_log = run_treatment_eval(args.model, args.samples)
    print(f"      Log: {treatment_log}")
    treatment_stats = analyze_log(treatment_log)
    print(f"      ff={treatment_stats['totals']['ff']}, fp={treatment_stats['totals']['fp']}, "
          f"kw={treatment_stats['kw_acc']:.3f}, jd={treatment_stats['jd_acc']:.3f}")
    print()

    print_comparison(control_stats, treatment_stats)
    print()
    print("=" * 60)
    print("Log files for replay:")
    print(f"  CONTROL:   {control_log}")
    print(f"  TREATMENT: {treatment_log}")


if __name__ == "__main__":
    main()
