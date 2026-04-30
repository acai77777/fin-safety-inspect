# Cross-model baseline driver: 100 FinTrust safety samples × keyword + LLM-judge scorers.
# Default backbone is deepseek-chat; pass -Model to switch.
# See RUN.md for the per-backbone env settings (OPENAI_API_KEY, OPENAI_BASE_URL).

param(
    [string]$Model = "deepseek-chat"
)

Write-Host "Running fintrust_safety on backbone: $Model"
Write-Host "OPENAI_BASE_URL=$($env:OPENAI_BASE_URL)"
Write-Host ""

inspect eval src/fin_safety_inspect/tasks/fintrust_safety.py@fintrust_safety `
    -T use_remote=True `
    -T max_samples=100 `
    --solver src/fin_safety_inspect/solvers/langgraph_solver.py@langgraph `
    -S graph=fin_safety_inspect.examples.mock_helper:build_graph `
    -S "model=openai:$Model" `
    --model "openai/$Model"
