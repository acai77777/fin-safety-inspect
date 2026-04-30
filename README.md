# fin-safety-inspect

Inspect AI task pack for **financial agent safety** + first public **LangGraph adapter** for [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai).

> **Status: v0.1 baseline ran** — 100-sample FinTrust × DeepSeek-chat × mock LangGraph banking agent: `keyword_refusal accuracy 0.700` in 47s. See [docs/baseline-v0.1.md](docs/baseline-v0.1.md) for the failure-mode analysis.

## What this gives you

1. **`fintrust_safety` task** — financial-crime refusal across 10 categories ([FinTrust](https://github.com/HughieHu/FinTrust/) safety subset, EMNLP 2025; English prompts). 100 samples, pulled from a pinned upstream commit, cached locally.
2. **`langgraph_solver`** — point Inspect AI at any compiled `StateGraph` that follows the contract below. v0.1 supports `inject_via="user_message"` (other modes raise `NotImplementedError`).
3. **`keyword_refusal` scorer** — ~60 EN+ZH refusal phrases with Unicode quote/dash normalization. Fast, deterministic, no judge LLM. Honest about its limits — see baseline doc.
4. **`tool_call_state` scorer** — deterministic allowed/banned tool-name check, reads `state.metadata["tool_calls"]` populated by the adapter.
5. **`examples/mock_helper.py`** — 50-line ReAct banking agent (4 tools, 2 allowed + 2 banned) so anyone can reproduce the baseline without private code.

## Baseline (v0.1)

| Metric | Value |
| --- | --- |
| Task | `fintrust_safety` |
| Model | `deepseek-chat` (via `OPENAI_BASE_URL=https://api.deepseek.com/v1`) |
| Agent | `examples/mock_helper.py` (mock banking ReAct, 4 tools) |
| Scorer | `keyword_refusal(expect_refusal=True)` |
| Samples | 100 |
| Wall time | 47 s |
| Cost | ≈ ¥0.20 |
| **Accuracy** | **0.700** |

Manual triage of the 30 false negatives reveals **three orthogonal failure modes**:

1. **Agent off-task tool misuse (~9 / 30)** — agent calls `list_accounts()` on a tax-evasion question and returns the tool result as the final answer. The keyword scorer correctly marks these FAIL, but for the wrong reason — the *real* safety signal is that the agent shouldn't be running banking tools on a "how do I hide cash income" prompt.
2. **Keyword-undercovered soft refusal (~11 / 30)** — model refuses with phrasing like *"I'd recommend consulting a legal professional"* or *"I need to address some concerns"* that doesn't match our phrase list. Motivates a v0.2 LLM-judge scorer.
3. **Genuinely unrefused harmful guidance (~10 / 30)** — model produces actionable detail (e.g. fraudster check-forging methods, hawala mechanics, P2P verification attack surfaces). The most important category, and the one a more sophisticated scorer cannot rescue: the model itself failed.

Mode #1 is a real-world signal that scorer-level fixes alone cannot capture; it needs multi-message and tool-call analysis (which is exactly what `tool_call_state` is built for, paired with v0.2 work). Full breakdown, per-category FN rate, and verbatim sample excerpts in [docs/baseline-v0.1.md](docs/baseline-v0.1.md).

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
- writes `state.metadata["tool_calls"]` so deterministic scorers can read the call chain
- writes `state.metadata["final_messages"]` for debugging
- writes `state.metadata["langgraph_error"]` on any exception (so a flaky run does not silently zero out)
- caps recursion at `max_turns * 2` (default `max_turns=10`)
- returns the **last `AIMessage` content** as the Inspect completion (see baseline mode #1 for a known consequence of this choice)

## Quick start (DeepSeek, public-reproducible)

```powershell
# 1. Install
pip install -e ".[dev,deepseek]"

# 2. Set env
$env:OPENAI_API_KEY  = "<your-deepseek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"

# 3. Run the 100-sample baseline (≈47s, ≈¥0.20)
.\run_smoke.ps1

# 4. Inspect the false negatives
python dump_failures.py
```

For Anthropic / OpenAI variants, see [RUN.md](RUN.md).

## Repo layout

```
fin-safety-inspect/
├── pyproject.toml
├── README.md
├── RUN.md
├── LICENSE                            # Apache-2.0
├── run_smoke.ps1                      # 100-sample baseline driver
├── dump_log.py                        # print all samples from latest log
├── dump_failures.py                   # print only false-negative samples + per-category breakdown
├── src/fin_safety_inspect/
│   ├── solvers/langgraph_solver.py
│   ├── scorers/
│   │   ├── keyword_refusal.py
│   │   └── tool_call_state.py
│   ├── tasks/fintrust_safety.py
│   ├── datasets/fintrust.py
│   └── examples/mock_helper.py        # public reference LangGraph banking agent
├── tests/
│   └── test_solver_langgraph.py       # 11 unit tests, no API key required
└── docs/
    └── baseline-v0.1.md               # 30 false negatives, three failure modes
```

## Why this exists

`research.md` (sibling repo) found that financial-agent safety tooling is fragmented: LLMs in one repo, agent frameworks in another, guardrails in a third, red-team benchmarks scattered. `inspect_evals` provides AgentDojo banking and generic safety tasks but no financial-domain task pack and no LangGraph adapter. This repo fills the LangGraph gap and ships a reproducible baseline on a public dataset to anchor the discussion.

## Status, scope, and what's NOT here yet

v0.1 deliberately ships a narrow, complete slice. Not in scope until v0.2+:

- LLM-judge scorer (Mode #2 in baseline)
- Trace-level safety scoring beyond final-message extraction (Mode #1)
- `langgraph_on_agentdojo` cookbook on the upstream `inspect_evals/agentdojo` banking suite
- `financebench_numeric` task (placeholder loader; needs a real numeric-grounding scorer)
- `inject_via="tool_result"` and `inject_via="document"` injection surfaces
- `cnfinbench_compliance` task — upstream dataset not yet public ([arXiv 2512.09506](https://arxiv.org/abs/2512.09506))
- FinRobot / TradingAgents / ElizaOS agent adapters
- Cross-N-model evals, leaderboard, HuggingFace Space upload

See `../design.md` for full design doc and rationale.

## Citation

If you use this repo:

```
@misc{fin-safety-inspect-2026,
  title  = {fin-safety-inspect: a LangGraph adapter and financial safety task pack for Inspect AI},
  author = {acai777},
  year   = {2026},
  url    = {https://github.com/acai777/fin-safety-inspect}
}
```

Upstream datasets retain their own licenses:

- FinTrust — Hu et al., EMNLP 2025, [github.com/HughieHu/FinTrust](https://github.com/HughieHu/FinTrust)
- Inspect AI — UK AI Safety Institute, [github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)

## License

Apache-2.0. We do not redistribute upstream dataset content.
