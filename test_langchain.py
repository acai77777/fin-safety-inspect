"""Verify LangChain init_chat_model picks up OPENAI_BASE_URL for DeepSeek."""

from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:deepseek-chat")
resp = llm.invoke("say hi in one word")
print("CONTENT:", resp.content)
print("MODEL:  ", getattr(resp, "response_metadata", {}).get("model_name"))
