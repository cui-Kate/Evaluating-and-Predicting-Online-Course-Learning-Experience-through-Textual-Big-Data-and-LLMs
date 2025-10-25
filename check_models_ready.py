# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct",       ROOT/"models"/"Qwen_Qwen2.5-7B-Instruct"),
    ("deepseek-ai/deepseek-llm-7b-chat", ROOT/"models"/"deepseek-ai_deepseek-llm-7b-chat"),
]

def ok(p: Path):
    if not p.exists(): return False, "dir missing"
    names = {x.name for x in p.glob("*") if x.is_file()}
    has_cfg = "config.json" in names
    has_tok = ("tokenizer.json" in names) or ({"vocab.json","merges.txt"} <= names)
    has_wt  = any(n.endswith(".safetensors") for n in names) or any(n.endswith(".bin") for n in names)
    return (has_cfg and has_tok and has_wt, f"cfg={has_cfg}, tok={has_tok}, wt={has_wt}")

for repo, path in MODELS:
    ready, detail = ok(path)
    print(f"{repo:35s} -> {path.name:35s} :: {'OK' if ready else 'MISSING'} ({detail})")
