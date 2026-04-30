# We tested 5 Chinese LLMs on financial-crime refusal. The reasoning model came last.

**Epistemic status**: a single-task cross-model comparison on FinTrust safety prompts (n=100 each), with a fixed mock LangGraph banking agent and a pinned LLM judge. The R1-vs-chat finding replicates only this task; the methodological points (judge fingerprinting, mode separation, roadmap revision from data) are more durable than the specific accuracy figures.

This post accompanies the [v0.2.1 release](https://github.com/acai77777/fin-safety-inspect/releases/tag/v0.2.1) and corresponds to [docs/cross-model-baseline.md](https://github.com/acai77777/fin-safety-inspect/blob/main/docs/cross-model-baseline.md).

## A counterintuitive opening

We fed the same 100 financial-crime prompts (FinTrust safety subset, EMNLP 2025; covering tax evasion, money laundering, insider trading, counterfeit currency, illegal lending, and 5 more categories) to 5 Chinese-deployable LLMs. Each model drove the same mock LangGraph banking agent with the same 4 tools.

Ranked by **judge accuracy** (an LLM judge deciding whether the model appropriately refused):

| Rank | Model | judge_acc | keyword_acc | Δ |
| --- | --- | --- | --- | --- |
| 1 | **Qwen-plus** | **0.950** | 0.930 | +0.020 |
| 2 | DeepSeek-chat | 0.910 | 0.790 | +0.120 |
| 3 | GLM-4.5 | 0.880 | 0.870 | +0.010 |
| 4 | DeepSeek-R1 | **0.840** | **0.670** | +0.170 |
| 5 | Kimi K2 | 0.830 | 0.810 | +0.020 |

**DeepSeek-R1 (the reasoning model) ranks fourth out of five**. On raw keyword accuracy it ranks dead last (0.670), 12 points below the same family's DeepSeek-chat (0.790).

This contradicts the default intuition that more capable, more "thinking-equipped" models should be safer. The data says otherwise.

## What this is

[fin-safety-inspect](https://github.com/acai77777/fin-safety-inspect) is a Python package shipping an Inspect AI task pack for financial agent safety + the first public LangGraph adapter for Inspect AI. v0.2.1 adds a cross-model baseline.

Critical setup detail: each model runs the same prompts, but **the LLM judge is pinned to DeepSeek-v4-flash** for every run (via independent `JUDGE_BASE_URL` / `JUDGE_API_KEY` env vars). All cross-model accuracy numbers are scored by the same judge, so they're directly comparable. Without this pin, judging Qwen with a Qwen judge and DeepSeek with a DeepSeek judge would muddy any cross-model comparison.

## Main finding 1: the reasoning model is the worst refuser, not the best

R1 is the same family as DeepSeek-chat, larger, and emits thinking tokens. **On financial-crime refusal:**

- keyword accuracy: chat 0.790 vs **R1 0.670** (12-point gap)
- judge accuracy: chat 0.910 vs **R1 0.840** (7-point gap)
- pp (both scorers PASS): chat 78 vs **R1 63** (lowest of all 5)
- fp (judge rescues a soft refusal keyword missed): chat 13 vs **R1 21** (highest of all 5)

Plausible mechanism (hypothesis, not yet rigorously tested):

R1's reasoning RLHF reinforces *"decompose the problem and provide detailed analysis"* more strongly than *"refuse early."* When a financial-crime prompt arrives, R1's thinking trajectory leans toward finding angles to engage rather than angles to decline. The thinking tokens get spent on decomposition, not on refusal.

**For deployers**: do not assume a more capable / more reasoning-equipped model is automatically safer on safety-critical tasks. On this benchmark, R1 is materially less safe than DeepSeek-chat.

## Main finding 2: DeepSeek family has a soft-refusal fingerprint

| family | judge - keyword (Mode #2 rescue rate) |
| --- | --- |
| DeepSeek (chat / R1) | **+0.120 / +0.170** |
| Qwen / GLM / Kimi | +0.010 to +0.020 |

DeepSeek's training preferences favor sympathetic phrasing like *"I'd recommend consulting a legal professional"* or *"What you're describing could potentially involve insider trading..."* without an explicit decline. The keyword scorer doesn't see *"I cannot"*-style hard-refusal phrases, marks FAIL. The LLM judge reads context, marks PASS.

Qwen / GLM / Kimi are different. They prefer hard refusals (*"This is illegal"*, *"I cannot help"*) that the keyword list catches directly. The judge has nothing soft to rescue.

**For evaluators**: a keyword-only refusal scorer will systematically *under*estimate DeepSeek's safety relative to Qwen / GLM / Kimi. We claim no other Chinese-LLM safety leaderboard has reported this fingerprint with primary numbers. **A keyword-only leaderboard scores DeepSeek unfairly low.**

## Side finding 1: Mode #1 is DeepSeek-specific, not universal

Our v0.2.0 baseline doc (5 hours older than this v0.2.1 baseline) hypothesized that *"Mode #1 (agent calls a banking tool on an unrelated prompt and emits the tool result)"* was a universal failure mode. It only had data from DeepSeek-chat at the time.

The cross-model comparison disproves the universality:
- **Qwen-plus Illegal Lending ff = 0**: Qwen never goes off-task.
- **DeepSeek-chat Illegal Lending ff = 2**: textbook Mode #1, agent calls `list_accounts()` on a loan-shark prompt.
- **GLM-4.5 Illegal Lending ff = 5** (highest of any model): but **hand-triage of all 5 confirms zero are Mode #1**.

GLM's mode of failure on lending prompts is completely different. The model treats *"how do I set up an informal driver-loan system"* as a knowledge-domain question and produces a *"Best Practices for Informal Lending"* checklist.

**Same prompts. Same FAIL. Different mechanisms:**
- DeepSeek's failure is **engineering**: the agent dispatches to a banking tool because the prompt mentions a small business; the tool result becomes the answer.
- GLM's failure is **training**: the model treats the illegal-lending question as a general "explain how this works" knowledge question.

## Side finding 2: Mode #3 is universal

Counterfeit Currency has `ff` (both scorers FAIL) in every model: DS-chat 3 / DS-R1 2 / Kimi 1 / Qwen 2 / GLM 4. The prompts in this category are designed to *seem* operationally legitimate ("how do I spot counterfeit bills"-style — could be from a small business owner). Every model in our 5-backbone sample falls for at least one.

Same pattern in Identity Theft. Same in Fund-raising. **Mode #3 has no scorer-layer fix**: either the model itself refuses, or an output-side guardrail catches it.

## Methodology takeaway

The cross-model data forced us to do a data-driven rewrite of the v0.2.x roadmap:

| v0.2.0's planned v0.2.1 priority | Now | Why |
| --- | --- | --- |
| #1 Trace-level scorer for Mode #1 | **DOWNGRADED to v0.3+** | Mode #1 is DeepSeek-specific. A trace scorer is a DeepSeek engineering aid, not a general benchmark feature. |
| #2 LLM-judge scorer | Already shipped in v0.2.0 | Done. |
| #3 Cross-model baseline | This release | Done. |
| **NEW priority #1: output-side guardrail design** | Promoted | The only scorer-layer-invariant fix for Mode #3, the only universal failure mode. |
| **NEW priority #2: R1 reasoning-vs-refusal mini-study** | New | R1's 0.12 keyword gap below chat is a finding; we don't yet know whether it's reasoning tokens, RLHF, or temperature defaults. Worth a focused study. |
| **NEW priority #3: GPT-4o / Claude baselines** | New | Without international models we can't say whether Mode #3 is a Chinese-RLHF artifact or global. |

Generalizable lesson, complementary to v0.1's:

> **When a baseline number is counterintuitive, find the explanation in the data first, then decide whether to fix the model, the scorer, or the task.**

v0.1 lesson: when the baseline number is *unflattering*, hand-triage the FNs first.
v0.2.1 lesson: when the baseline number is *counterintuitive* (R1 < chat), do the cross-model comparison first.

Combined: **don't read one number; read how it varies across the dimensions you can change**.

## What v0.2.1 doesn't include

- **No GPT-4o or Claude baselines**. The OpenRouter relay we tried first was misconfigured at the time. We chose to ship a 5-Chinese-model baseline rather than wait. Next release.
- **No mechanistic study of the R1 finding**. We observe R1 refuses less; we haven't ruled out thinking-tokens / RLHF emphasis / temperature defaults / prompt-following style.
- **No trace-level scorer**. The cross-model data revised this down in priority. Maybe v0.3.

## Try it

```bash
pip install git+https://github.com/acai77777/fin-safety-inspect.git@v0.2.1
```

```powershell
# Pin the judge to DeepSeek (independent of the agent backbone)
$env:JUDGE_API_KEY  = "<your-deepseek-key>"
$env:JUDGE_BASE_URL = "https://api.deepseek.com/v1"

# Default DeepSeek baseline
$env:OPENAI_API_KEY  = "<your-deepseek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
.\run_smoke.ps1                                  # ≈60s, ≈¥0.22

# Cross-model: rotate $env:OPENAI_* and re-run
.\run_smoke.ps1 -Model "qwen-plus"
.\run_smoke.ps1 -Model "deepseek/deepseek-r1"    # via OpenRouter

python dump_cross_model.py                       # 5-model comparison tables
python dump_failures.py "z-ai/glm-4.5"           # ff samples for one model
```

Full baseline writeup: [docs/cross-model-baseline.md](https://github.com/acai77777/fin-safety-inspect/blob/main/docs/cross-model-baseline.md). Repo: [acai77777/fin-safety-inspect](https://github.com/acai77777/fin-safety-inspect). License: Apache-2.0. 31 unit tests pass without an API key.

## Citation

- FinTrust: Hu et al., "FinTrust: A Comprehensive Benchmark of Trustworthiness Evaluation in Finance Domain", EMNLP 2025
- Inspect AI: UK AI Safety Institute
