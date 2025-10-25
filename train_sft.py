# -*- coding: utf-8 -*-
"""
SFT-Regression：QLoRA（4bit）在8GB显存可运行。支持 --local_only 纯离线。
"""
import argparse, os
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training  # ★ 关键：QLoRA预处理
from utils.model_resolver import ensure_local_model
from utils.seeds import set_global_seed



# # --- quiet noisy warnings/logs ---
# import os, warnings
# from transformers.utils import logging as hf_logging

# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# os.environ.setdefault("PYTHONWARNINGS", "ignore")

# hf_logging.set_verbosity_error()

# # 精确屏蔽常见重复提示
# warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer than.*")
# warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
# warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")
# warnings.filterwarnings("ignore", category=UserWarning, module="trl.trainer.dpo_trainer")




def main(args):
    set_global_seed(args.seed)

    # ---- 基于项目根解析路径（不受CWD影响）----
    ROOT = Path(__file__).resolve().parents[1]
    def to_abs(p, default_rel):
        p = (ROOT / default_rel) if (p is None) else (Path(p) if os.path.isabs(p) else ROOT / p)
        return str(p)

    train_file = to_abs(args.train_file, "data_proc/sft_train.jsonl")
    val_file   = to_abs(args.val_file,   "data_proc/sft_val.jsonl")
    out_dir    = to_abs(args.out_dir,    "out/sft_qwen7b")

    # ---- 基座模型（仅本地/或断点续传）----
    base_dir = ensure_local_model(args.base_model, allow_download=not bool(args.local_only))

    # ---- 4bit量化 + QLoRA 预处理 ----
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_dir, quantization_config=bnb_cfg, device_map="auto")
    # 关键：为k-bit训练做准备（启用输入梯度、禁用cache、cast norm等）
    model = prepare_model_for_kbit_training(model)
    # 梯度检查点：新版API可传use_reentrant=False，旧版则回退
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)

    # ---- 数据集 ----
    def to_text(ex):
        # 把我们SFT格式拼成纯文本（collator会自动复制labels）
        return {"text": f"{ex['instruction']}\n{ex['output']}"}

    ds_tr = load_dataset("json", data_files=train_file, split="train").map(to_text)
    ds_va = load_dataset("json", data_files=val_file,   split="train").map(to_text)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)
    ds_tr = ds_tr.map(tok_fn, batched=True, remove_columns=ds_tr.column_names)
    ds_va = ds_va.map(tok_fn, batched=True, remove_columns=ds_va.column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    os.makedirs(out_dir, exist_ok=True)

    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,            # ★ 这行一定要有
        learning_rate=args.lr,
        evaluation_strategy=("steps" if args.eval_steps>0 else "no"),
        eval_steps=args.eval_steps,          # ★
        save_steps=args.save_steps,          # ★
        logging_steps=args.logging_steps,    # ★
        save_total_limit=2, bf16=True,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        gradient_checkpointing=True,
        report_to="none",
    )


    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va, data_collator=collator)
    trainer.train()
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print(f"[OK] SFT model saved to {out_dir}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
#     ap.add_argument("--train_file", type=str, default=None)
#     ap.add_argument("--val_file",   type=str, default=None)
#     ap.add_argument("--out_dir",    type=str, default=None)
#     ap.add_argument("--epochs",  type=float, default=2.0)
#     ap.add_argument("--lr",      type=float, default=2e-4)
#     ap.add_argument("--grad_accum", type=int, default=8)
#     ap.add_argument("--lora_r", type=int, default=16)
#     ap.add_argument("--lora_alpha", type=int, default=32)
#     ap.add_argument("--lora_dropout", type=float, default=0.05)
#     ap.add_argument("--eval_steps", type=int, default=200)
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--local_only", type=int, default=0, help="1=仅用本地模型，不联网下载")
#     ap.add_argument("--max_len", type=int, default=256)   # ★ 降到 256
#     ap.add_argument("--max_steps", type=int, default=2000) # ★ 强约束训练步数

#     main(ap.parse_args())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen_Qwen2.5-7B-Instruct")
    ap.add_argument("--train_file", type=str, default=None)
    ap.add_argument("--val_file",   type=str, default=None)
    ap.add_argument("--out_dir",    type=str, default=None)

    # 只保留一份
    ap.add_argument("--max_len",   type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=200)   # 烟囱测试用，训练期间优先生效

    ap.add_argument("--epochs",  type=float, default=1.0)   # 当 max_steps>0 时会被忽略
    ap.add_argument("--lr",      type=float, default=2e-4)
    ap.add_argument("--grad_accum",   type=int, default=8)
    ap.add_argument("--lora_r",       type=int, default=16)
    ap.add_argument("--lora_alpha",   type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # 评估/保存/日志步频（训练期间生效）
    ap.add_argument("--eval_steps",    type=int, default=50)
    ap.add_argument("--save_steps",    type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_only", type=int, default=1, help="1=仅用本地模型，不联网下载")
    main(ap.parse_args())
