# Full baseline: 100 samples (FinTrust safety/plain_question), ~¥0.20, ~3-5 min on deepseek-chat.
# Required env:
#   $env:OPENAI_API_KEY  = "<deepseek key, sk-...>"
#   $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"

inspect eval src/fin_safety_inspect/tasks/fintrust_safety.py@fintrust_safety `
    -T use_remote=True `
    -T max_samples=100 `
    --solver src/fin_safety_inspect/solvers/langgraph_solver.py@langgraph `
    -S graph=fin_safety_inspect.examples.mock_helper:build_graph `
    -S model=openai:deepseek-chat `
    --model openai/deepseek-chat
