# -*- coding: utf-8 -*-
from utils.model_resolver import ensure_local_model

for repo in [
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/deepseek-llm-7b-chat",
]:
    print("== Prefetch:", repo)
    p = ensure_local_model(repo, allow_download=True)
    print(" ->", p)
print("All prefetched.")
