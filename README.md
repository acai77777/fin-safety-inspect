# fin-safety-inspect

Inspect AI task pack for **financial agent safety** + first public **LangGraph adapter** for [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai).

> **Status: v0.3 output guardrail validated end-to-end** — on a 100-sample DeepSeek-chat run, the guardrail cuts Mode #3 ff from 14 → 3 (−78.6%) and lifts judge accuracy from 0.78 → 0.95, with 0 measurable false positives. See [docs/guardrail-validation.md](docs/guardrail-validation.md) for the per-category breakdown and sample interceptions. Earlier: [v0.2.1 cross-model baseline](docs/cross-model-baseline.md) (5 Chinese backbones × 100-sample FinTrust).

## What this gives you

1. **`fintrust_safety` task** — financial-crime refusal across 10 categories ([FinTrust](https://github.com/HughieHu/FinTrust/) safety subset, EMNLP 2025; English prompts). 100 samples, pinned upstream commit, cached locally. Optional `guardrail=true` task arg wraps the solver with the v0.3 Mode #3 interceptor.
2. **`langgraph_solver`** — point Inspect AI at any compiled `StateGraph` that follows the contract below.
3. **`guardrail_solver`** (v0.3) — wraps any inner solver, runs an LLM classifier on the agent's output, and replaces harmful operational guidance (Mode #3) with a safe refusal. Default mode `replace`; alternative `detect_only` logs without rewriting.
4. **`keyword_refusal` scorer** — ~60 EN+ZH refusal phrases with Unicode normalization. Fast, deterministic, no judge LLM.
5. **`llm_judge_refusal` scorer** (v0.2) — async LLM judge over an OpenAI-compatible endpoint. Reads `JUDGE_BASE_URL` / `JUDGE_API_KEY` so the judge can be pinned (e.g. DeepSeek) while the agent rotates across backbones.
6. **`mode3_guardrail` scorer** (v0.3) — Mode #3 specific classifier; complements the broad refusal judge with a narrower "is this harmful operational guidance?" question.
7. **`tool_call_state` scorer** — deterministic allowed/banned tool-name check via `state.metadata["tool_calls"]`.
8. **`examples/mock_helper.py`** — 50-line ReAct banking agent (4 tools, 2 allowed + 2 banned).

## Cross-model baseline (v0.2.1)

| Backbone | keyword | judge | Δ | refusal style |
| --- | --- | --- | --- | --- |
| **Qwen-plus** | **0.930** | **0.950** | +0.020 | hardest refuser; near-zero soft refusals |
| GLM-4.5 | 0.870 | 0.880 | +0.010 | hard refuser |
| Kimi K2 | 0.810 | 0.830 | +0.020 | weaker refusal training |
| DeepSeek-chat | 0.790 | 0.910 | **+0.120** | soft-refusal heavy; judge rescues 13 |
| **DeepSeek-R1** | **0.670** | **0.840** | **+0.170** | reasoning RLHF reduces refusal rate |

Full breakdown, per-category 4-cell matrices, and the Mode #1 / #3 split per model in [docs/cross-model-baseline.md](docs/cross-model-baseline.md). Per-backbone reproduction commands in [RUN.md § 3](RUN.md#3-cross-model-baselines).

## Output-side guardrail (v0.3)

`guardrail_solver` wraps any inner solver and runs a Mode #3 classifier on each output. Harmful operational guidance is replaced with a safe refusal. Validated end-to-end on DeepSeek-chat × 100 FinTrust prompts:

| Metric | Without guardrail | With guardrail | Δ |
| --- | --- | --- | --- |
| ff (Mode #3 candidates) | 14 | **3** | **−11** |
| judge accuracy | 0.78 | **0.95** | +0.17 |
| guardrail triggered | n/a | 14 / 100 | — |
| measurable false positives | n/a | 0 | — |

Full per-category breakdown, sample interceptions, and the 3 known false negatives in [docs/guardrail-validation.md](docs/guardrail-validation.md). Reproduction:

```powershell
python e2e_guardrail.py --model deepseek-chat --samples 100
```

## Three failure modes (v0.1 framework, refined in v0.2.1)

1. **Mode #1 — Agent off-task tool misuse**: agent calls `list_accounts()` on a financial-crime prompt, returns the tool result as the answer. **DeepSeek-specific.** Qwen never does this. v0.2.1 hand-triage of GLM-4.5's lending failures confirmed they are a different mode (Mode #3, not Mode #1).
2. **Mode #2 — Keyword-undercovered soft refusal**: model refuses with phrasing the keyword list misses. **DeepSeek family signature** (chat +0.120, R1 +0.170 judge - keyword gap; all other families flat).
3. **Mode #3 — Genuinely unrefused harmful guidance**: model gives actionable detail framed as "education" or "risks". **Universal across all 5 backbones**, concentrated in Counterfeit / Lending / Fund-raising.

The earlier baselines (v0.1, v0.2.0) live at [docs/baseline-v0.1.md](docs/baseline-v0.1.md) and [docs/baseline-v0.2.0.md](docs/baseline-v0.2.0.md) for comparison.

## LangGraph adapter contract

Your `graph_factory` must return a compiled `StateGraph` whose state schema includes a `messages` field of `list[BaseMessage]`.

```python
def build_graph(model: str = "openai:deepseek-chat"):
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain.chat_models import init_chat_model
    # ... see src/fin_safety_inspect/examples/mock_helper.py for the full reference implementation
```

The adapter:

- imports your factory via a `module:callable` string (fail-fast on bad imports)
- runs `graph.ainvoke({"messages": [HumanMessage(state.user_prompt.text)]})`
- writes `state.metadata["tool_calls"]`, `state.metadata["final_messages"]`, and (on exception) `state.metadata["langgraph_error"]`
- caps recursion at `max_turns * 2` (default `max_turns=10`)
- returns the **last `AIMessage` content** as the Inspect completion (see Mode #1 above for a known consequence of this choice)

## Quick start (DeepSeek, public-reproducible)

```powershell
# 1. Install
pip install -e ".[dev,deepseek]"

# 2. Set env (agent + judge can share the same DeepSeek key/url for the default run)
$env:OPENAI_API_KEY  = "<your-deepseek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"

# 3. Run the 100-sample baseline (≈60s, ≈¥0.22)
.\run_smoke.ps1

# 4. Inspect the failures
python dump_judge_diff.py            # 4-cell agreement between keyword and judge
python dump_failures.py              # both-fail samples + per-category breakdown
```

For cross-model setups (Qwen, Kimi, GLM, R1, etc.), see [RUN.md § 3](RUN.md#3-cross-model-baselines).

## Repo layout

```
fin-safety-inspect/
├── pyproject.toml
├── README.md
├── RUN.md                              # full reproduction guide, including cross-model
├── LICENSE                             # Apache-2.0
├── run_smoke.ps1                       # parameterized baseline driver (-Model <name>)
├── dump_log.py                         # all samples from latest log
├── dump_failures.py                    # both-fail (ff) samples; accepts model filter
├── dump_judge_diff.py                  # keyword-vs-judge 4-cell agreement matrix
├── dump_cross_model.py                 # per-model accuracy + 4-cell + per-category ff
├── e2e_guardrail.py                    # v0.3 end-to-end guardrail validation driver
├── analyze_r1_vs_chat.py               # v0.3 per-category R1 vs DeepSeek-chat analysis
├── src/fin_safety_inspect/
│   ├── solvers/
│   │   ├── langgraph_solver.py
│   │   └── guardrail_solver.py         # v0.3 output-side Mode #3 interceptor
│   ├── scorers/
│   │   ├── keyword_refusal.py
│   │   ├── llm_judge_refusal.py        # v0.2 LLM-judge with JUDGE_* env vars
│   │   ├── guardrail_classifier.py     # v0.3 Mode #3 classifier + mode3_guardrail scorer
│   │   └── tool_call_state.py
│   ├── tasks/fintrust_safety.py        # attaches kw + judge + guardrail scorers
│   ├── datasets/fintrust.py
│   └── examples/mock_helper.py
├── tests/
│   ├── test_solver_langgraph.py        # 11 tests for the adapter
│   ├── test_tasks_smoke.py             # 6 tests for task / mock_helper construction
│   ├── test_scorer_llm_judge.py        # 12 tests for judge parser + cross-model env
│   └── test_guardrail.py               # v0.3 guardrail tests (20 tests)
└── docs/
    ├── baseline-v0.1.md                # original 30-FN three-failure-mode analysis
    ├── baseline-v0.2.0.md              # +LLM-judge; 4-cell matrix; new findings
    ├── cross-model-baseline.md         # 5-backbone comparison
    ├── guardrail-validation.md         # v0.3 end-to-end guardrail validation
    └── blog/                           # v0.1 + v0.2.1 release posts (zh + en + Twitter)
```

51 unit tests, all passing without an API key.

## Why this exists

`research.md` (sibling repo) found that financial-agent safety tooling is fragmented: LLMs in one repo, agent frameworks in another, guardrails in a third, red-team benchmarks scattered. UK AISI's [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) provides AgentDojo banking and generic safety tasks but no financial-domain task pack and no LangGraph adapter. This repo fills both gaps and ships reproducible baselines (v0.1 → v0.2.0 → v0.2.1 cross-model) so the conversation has primary numbers to point at.

## Roadmap

Driven by what the cross-model baseline data revealed (not what we planned in v0.2.0):

- **NEW priority #1**: Output-side guardrail design. Mode #3 is the only universal failure mode and has no scorer-layer fix.
- **NEW priority #2**: R1 reasoning-vs-refusal study. R1's 0.12 keyword-accuracy gap below DeepSeek-chat is a finding; we don't yet know whether it's reasoning tokens, RLHF, or temperature defaults.
- **NEW priority #3**: GPT-4o / Claude 3.5 Sonnet baselines. Without international models we cannot say whether Mode #3 is a Chinese-RLHF artifact or a global pattern.
- **DOWNGRADED**: Trace-level scorer for Mode #1. The cross-model data showed Mode #1 is DeepSeek-specific. A trace scorer is a DeepSeek engineering aid, not a general benchmark feature. Still useful, but not v0.3 priority.

Not yet in scope (any version):

- `langgraph_on_agentdojo` cookbook on the upstream `inspect_evals/agentdojo` banking suite
- `financebench_numeric` task (placeholder loader; needs a real numeric-grounding scorer)
- `inject_via="tool_result"` and `inject_via="document"` injection surfaces
- `cnfinbench_compliance` task ([arXiv 2512.09506](https://arxiv.org/abs/2512.09506) — dataset not public)
- FinRobot / TradingAgents / ElizaOS agent adapters
- Leaderboard / HuggingFace Space upload

See `../design.md` for full design doc and rationale.

## Citation

```
@misc{fin-safety-inspect-2026,
  title  = {fin-safety-inspect: a LangGraph adapter and cross-model financial safety baseline for Inspect AI},
  author = {acai777},
  year   = {2026},
  url    = {https://github.com/acai77777/fin-safety-inspect}
}
```

Upstream:

- FinTrust — Hu et al., EMNLP 2025, [github.com/HughieHu/FinTrust](https://github.com/HughieHu/FinTrust)
- Inspect AI — UK AI Safety Institute, [github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)

## License

Apache-2.0. We do not redistribute upstream dataset content.
