# -*- coding: utf-8 -*-
import subprocess, sys, os, time, shutil, traceback
from pathlib import Path

# ===== 绝对路径（按你本机路径改）=====
ROOT = r"E:\00project_code\main_cui\002\LLM"
DATA_DIR = rf"{ROOT}\data_proc"
BASE_QWEN     = rf"{ROOT}\models\Qwen_Qwen2.5-7B-Instruct"
BASE_DEEPSEEK = rf"{ROOT}\models\deepseek-ai_deepseek-llm-7b-chat"

SFT_TRAIN = rf"{DATA_DIR}\sft_train.jsonl"
SFT_VAL   = rf"{DATA_DIR}\sft_val.jsonl"
DPO_TRAIN = rf"{DATA_DIR}\dpo_train.jsonl"
DPO_VAL   = rf"{DATA_DIR}\dpo_val.jsonl"

# ===== 静默日志 =====
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

LOCAL_ONLY = 1
SKIP_IF_RUNOK = 1
SKIP_TRAIN_IF_MODEL_EXISTS = 1
SKIP_INFER_IF_RESULTS_EXIST = 1

# 可选：只跑指定 tag（逗号分隔），或跳过指定 tag
ONLY_TAGS = set(os.getenv("ONLY_TAGS","").split(",")) if os.getenv("ONLY_TAGS") else None
SKIP_TAGS = set(os.getenv("SKIP_TAGS","").split(",")) if os.getenv("SKIP_TAGS") else set()

COMBOS = [
  {"tag":"sft_qwen7b",     "base": BASE_QWEN,     "mode":"sft", "max_len":"128", "max_steps":"100"},
  {"tag":"dpo_qwen7b",     "base": BASE_QWEN,     "mode":"dpo", "max_len":"256", "max_steps":"80"},
  {"tag":"sft_deepseek7b", "base": BASE_DEEPSEEK, "mode":"sft", "max_len":"128", "max_steps":"100"},
  {"tag":"dpo_deepseek7b", "base": BASE_DEEPSEEK, "mode":"dpo", "max_len":"192", "max_steps":"80"},
]

os.makedirs(rf"{ROOT}\out\res", exist_ok=True)

def run(cmd):
    print(">>", " ".join(cmd)); sys.stdout.flush()
    return subprocess.run(cmd, check=True)

def model_ready(out_dir: str) -> bool:
    # LoRA 适配器或整模型任一存在即可视为“训练已完成”
    markers = [
        "adapter_model.safetensors", "adapter_config.json",
        "pytorch_model.bin", "model.safetensors", "consolidated.00.pth"
    ]
    return any((Path(out_dir)/m).exists() for m in markers)

def results_ready(res_dir: str) -> bool:
    need = ["review_scores.csv","course_scores.csv","eval_metrics.csv"]
    return all((Path(res_dir)/f).exists() for f in need)

def move_outputs_to_res(res_dir: str):
    for f in ["review_scores.csv","course_scores.csv","eval_metrics.csv"]:
        src = Path(ROOT)/"out"/f
        if src.exists():
            shutil.move(str(src), str(Path(res_dir)/f))

for c in COMBOS:
    tag, base, mode = c["tag"], c["base"], c["mode"]
    if ONLY_TAGS and tag not in ONLY_TAGS: 
        print(f"\n=== SKIP -> {tag} (not in ONLY_TAGS) ==="); continue
    if tag in SKIP_TAGS:
        print(f"\n=== SKIP -> {tag} (in SKIP_TAGS) ==="); continue

    out_dir = rf"{ROOT}\out\{tag}"
    res_dir = rf"{ROOT}\out\res\{tag}"
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== RUN -> {tag} ==="); sys.stdout.flush()

    # 1) 整体完成标记：直接跳过
    if SKIP_IF_RUNOK and (Path(res_dir)/"RUN_OK.txt").exists():
        print(f"[SKIP] {tag} :: RUN_OK.txt 存在，跳过本组合")
        continue

    # 2) 训练阶段（可按需跳过）
    skip_train = SKIP_TRAIN_IF_MODEL_EXISTS and model_ready(out_dir)
    if skip_train:
        print(f"[SKIP-TRAIN] {tag} :: 检测到模型权重，跳过训练")
    else:
        # 清空旧目录，避免残留冲突
        shutil.rmtree(out_dir, ignore_errors=True)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        try:
            if mode == "sft":
                run([
                    "python","src/train_sft.py",
                    "--base_model", base,
                    "--train_file", SFT_TRAIN, "--val_file", SFT_VAL,
                    "--out_dir", out_dir, "--local_only", str(LOCAL_ONLY),
                    "--max_len", c["max_len"], "--max_steps", c["max_steps"],
                    "--eval_steps","0","--save_steps","100000","--logging_steps","10"
                ])
            else:
                extra = []
                if "deepseek" in tag:
                    extra = ["--max_length","192","--max_prompt_length","96"]
                else:
                    extra = ["--max_length", c["max_len"], "--max_prompt_length","128"]
                run([
                    "python","src/train_dpo.py",
                    "--base_model", base,
                    "--train_file", DPO_TRAIN, "--val_file", DPO_VAL,
                    "--out_dir", out_dir, "--local_only", str(LOCAL_ONLY),
                    "--max_steps", c["max_steps"], "--eval_steps","0","--save_steps","100000","--logging_steps","10",
                    *extra
                ])
        except Exception as e:
            with open(rf"{ROOT}\out\res\FAILED.log","a",encoding="utf-8") as fp:
                fp.write(f"[FAILED-TRAIN] {tag} :: {e}\n{traceback.format_exc()}\n")
            continue

    # 3) 推断 + 评测（可按需跳过）
    if SKIP_INFER_IF_RESULTS_EXIST and results_ready(res_dir):
        print(f"[SKIP-INFER] {tag} :: 已存在结果文件，跳过推断与评测")
    else:
        try:
            # 推断聚合（传入基座，走4bit加载，显存更稳）
            run([
              "python","src/infer_and_aggregate.py",
              "--model_dir", out_dir,
              "--base_model", base,
              "--use_4bit","1"
            ])
            # 评测
            run(["python","src/evaluate.py"])
            # 归档结果
            move_outputs_to_res(res_dir)
        except Exception as e:
            with open(rf"{ROOT}\out\res\FAILED.log","a",encoding="utf-8") as fp:
                fp.write(f"[FAILED-INFER] {tag} :: {e}\n{traceback.format_exc()}\n")
            continue

    # 4) 标记完成
    with open(rf"{res_dir}\RUN_OK.txt","w",encoding="utf-8") as fp:
        fp.write(time.strftime("%F %T"))

print("\n=== ALL DONE ===")
