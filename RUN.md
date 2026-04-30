# Run baseline

Two-step path: smoke (5 samples, prove the pipeline) → baseline (100 samples, get real numbers). DeepSeek-chat is the **reference backbone** — it's what `.\run_smoke.ps1` defaults to and what [docs/baseline-v0.1.md](docs/baseline-v0.1.md) and [docs/baseline-v0.2.0.md](docs/baseline-v0.2.0.md) report against.

`run_smoke.ps1` accepts `-Model <name>` so you can swap backbones without editing the script. See **§ 3. Cross-model baselines** below.

## 0. Pre-flight (one-time)

```powershell
# Editable install with the right backend extras
pip install -e ".[dev,deepseek]"     # DeepSeek (OpenAI-compatible, what we run against)
# or
pip install -e ".[dev,openai]"       # OpenAI direct
# or
pip install -e ".[dev,anthropic]"    # Anthropic
```

## 1. Verify connectivity (recommended before first run)

```powershell
python test_relay.py        # 1-line httpx call to /chat/completions
python test_langchain.py    # 1-line LangChain init_chat_model + invoke
```

Both should print `Hi` or similar. If either fails, fix that before invoking Inspect.

## 2. Default DeepSeek baseline (100 samples, ≈60 s, ≈¥0.22)

```powershell
$env:OPENAI_API_KEY  = "<your-deepseek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
.\run_smoke.ps1
```

This is the reference run reported in `docs/baseline-v0.2.0.md` (keyword 0.790, judge 0.910). First run pulls the FinTrust safety set from the pinned upstream commit; subsequent runs hit the local cache at `~/.cache/fin_safety_inspect/`.

## 3. Cross-model baselines

Each backbone is a separate run. Set the right env vars, then call `.\run_smoke.ps1 -Model <name>`. Each run produces a fresh `.eval` log under `logs/`. After all four runs, `python dump_cross_model.py` builds the comparison table.

### 3.1 DeepSeek-chat (reference)

```powershell
$env:OPENAI_API_KEY  = "<your-deepseek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
.\run_smoke.ps1 -Model "deepseek-chat"
```

### 3.2 Qwen-plus (Alibaba DashScope)

```powershell
$env:OPENAI_API_KEY  = "<your-dashscope-key>"
$env:OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
.\run_smoke.ps1 -Model "qwen-plus"
```

DashScope is OpenAI-compatible. Same `OPENAI_API_KEY` env name, just different value and base URL. If you want `qwen2.5-72b-instruct` or `qwen-max`, replace the `-Model` argument.

### 3.3 GPT-4o-mini (via OpenRouter relay)

```powershell
$env:OPENAI_API_KEY  = "<your-relay-key>"
$env:OPENAI_BASE_URL = "https://www.openclaudecode.cn/v1"
.\run_smoke.ps1 -Model "openai/gpt-4o-mini"
```

OpenRouter model names contain a slash (`openai/gpt-4o-mini`). The script forwards this verbatim. If Inspect rejects the slashed name, check that your OpenRouter relay also exposes a non-slashed alias.

### 3.4 Claude 3.5 Sonnet (via OpenRouter relay)

```powershell
$env:OPENAI_API_KEY  = "<your-relay-key>"
$env:OPENAI_BASE_URL = "https://www.openclaudecode.cn/v1"
.\run_smoke.ps1 -Model "anthropic/claude-3.5-sonnet"
```

Routed via OpenRouter as an OpenAI-compatible call. We do NOT use `anthropic:` provider here because the relay normalizes everything to OpenAI protocol.

### 3.5 Compare all 4 runs

```powershell
python dump_cross_model.py
```

Prints:
- Per-model 2-scorer accuracy (keyword vs judge)
- Per-model 4-cell distribution (pp / pf / fp / ff)
- Per-model per-category `ff` counts (where Mode #1 / #3 hide)

Use the per-category `ff` table to decide v0.2.1 priority: if `ff` clusters on one model, Mode #1 is model-specific (worth a trace scorer); if uniform across models, it's task-specific (Mode #3, no scorer fix).

## 4. View results

```powershell
inspect view
# browse to http://localhost:7575
```

Or stay on the CLI:

```powershell
python dump_log.py          # all samples, input + completion + score
python dump_failures.py     # only false negatives, plus per-category breakdown
python dump_judge_diff.py   # 4-cell agreement matrix between keyword and judge
python dump_cross_model.py  # comparison across all 4 backbones
```

## 5. What success looks like

- 100 samples scored, no `langgraph_error` in `state.metadata`
- Two scorer accuracy columns printed at the end (`keyword_refusal` and `llm_judge_refusal`)
- DeepSeek-chat reference run is `kw 0.79 / judge 0.91` (as of 2026-04-30)
- `state.metadata.tool_calls` populated even for chat-only runs (empty list is fine)

If accuracy is below `0.5` on any backbone, something is wrong — most likely connectivity-driven timeouts or relay misconfig. Check `dump_log.py` output for `<langgraph error: ...>`.
