# Run baseline

Two-step path: smoke (5 samples, prove the pipeline) → baseline (100 samples, get real numbers). DeepSeek is the **reference backbone** — it's what `.\run_smoke.ps1` uses and what the [v0.1 baseline doc](docs/baseline-v0.1.md) reports against.

## 0. Pre-flight (one-time)

```powershell
# Editable install with the right backend extras
pip install -e ".[dev,deepseek]"     # DeepSeek (OpenAI-compatible, what we run against)
# or
pip install -e ".[dev,openai]"       # OpenAI direct
# or
pip install -e ".[dev,anthropic]"    # Anthropic
```

Set the API key as env var:

```powershell
# DeepSeek (reference)
$env:OPENAI_API_KEY  = "<your-deepseek-key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"

# OR OpenAI direct
$env:OPENAI_API_KEY = "sk-..."
# (do NOT also set OPENAI_BASE_URL)

# OR Anthropic
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## 1. Verify connectivity (recommended before first run)

If you're behind a relay or a non-default base URL, save yourself an hour of head-scratching with:

```powershell
python test_relay.py        # 1-line httpx call to /chat/completions
python test_langchain.py    # 1-line LangChain init_chat_model + invoke
```

Both should print `Hi` or similar. If either fails, fix that before invoking Inspect.

## 2. Smoke (5 samples, ≈30 s, ≈¥0.01 on DeepSeek)

For a 5-sample run, edit `run_smoke.ps1` to set `-T max_samples=5` and `-T use_remote=False` (uses the embedded 10-prompt offline subset, capped to 5). Otherwise `.\run_smoke.ps1` runs the full 100.

## 3. Full baseline (100 samples, ≈47 s, ≈¥0.20 on DeepSeek)

```powershell
.\run_smoke.ps1
```

This is what the v0.1 baseline number (`accuracy 0.700`) was measured with. First run pulls the FinTrust safety set from the pinned upstream commit; subsequent runs hit the local cache at `~/.cache/fin_safety_inspect/`.

To swap backbones, edit `run_smoke.ps1`:

```powershell
# OpenAI gpt-4o-mini
-S model=openai:gpt-4o-mini --model openai/gpt-4o-mini

# Anthropic Claude 3.5 Sonnet
-S model=anthropic:claude-3-5-sonnet-latest --model anthropic/claude-3-5-sonnet-latest
```

`-S model=...` goes to the LangGraph mock helper (the agent that does the actual reasoning). `--model ...` is the Inspect-level model declaration.

## 4. View results

```powershell
inspect view
# browse to http://localhost:7575
```

Or stay on the CLI:

```powershell
python dump_log.py          # all samples, input + completion + score
python dump_failures.py     # only false negatives, plus per-category breakdown
```

## 5. What success looks like

- 100 samples scored, no `langgraph_error` in `state.metadata`
- `accuracy 0.65 – 0.80` on DeepSeek-chat (the v0.1 reference run hit `0.700`)
- Each sample's trace shows `HumanMessage → AIMessage → optional ToolMessage`
- `state.metadata.tool_calls` populated even for chat-only runs (empty list is fine)

If accuracy is below `0.5` on DeepSeek, something is wrong — most likely connectivity-driven timeouts. Check `dump_log.py` output for `<langgraph error: APITimeoutError>`.
