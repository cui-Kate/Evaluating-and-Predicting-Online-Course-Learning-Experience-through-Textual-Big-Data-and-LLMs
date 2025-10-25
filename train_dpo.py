
# -*- coding: utf-8 -*-
"""
DPO on 8GB with QLoRA 4-bit.
Key changes for stability on Windows + TRL 0.9.6:
- Keep 'prompt/chosen/rejected' for TRL internal tokenize_row.
- Strong pre-filter: remove empty/untokenizable rows BEFORE Trainer.
- Custom data_collator to sanitize any None and pad safely (bypass TRL's collator).
"""
import argparse, os
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.model_resolver import ensure_local_model
from utils.seeds import set_global_seed

import os
os.environ["ACCELERATE_DISABLE_FLOPS_CALCULATION"] = "1"  # 仅禁用FLOPs估算，训练不受影响

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
    ROOT = Path(__file__).resolve().parents[1]

    def abs_path(p, fallback):
        if p is None:
            return str(ROOT / fallback)
        return str(Path(p) if os.path.isabs(p) else (ROOT / p))

    train_file = abs_path(args.train_file, "data_proc/dpo_train.jsonl")
    val_file   = abs_path(args.val_file,   "data_proc/dpo_val.jsonl")
    out_dir    = abs_path(args.out_dir,    "out/dpo_qwen7b")

    base_dir = ensure_local_model(args.base_model, allow_download=not bool(args.local_only))

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "right"
    tok.model_max_length = max(args.max_length, args.max_prompt_length)

    # --- 4bit + LoRA ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(base_dir, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora)

    # --- Load raw jsonl ---
    ds_tr = load_dataset("json", data_files=train_file, split="train")
    ds_va = load_dataset("json", data_files=val_file,   split="train")

    # ---- 1) 清洗：去 None/去首尾空格/三列非空 ----
    def sanitize(ex):
        for k in ("prompt","chosen","rejected"):
            v = ex.get(k, "")
            if v is None: v = ""
            ex[k] = str(v).strip()
        ex["_ok_txt"] = int(all(len(ex[k]) > 0 for k in ("prompt","chosen","rejected")))
        return ex

    ds_tr = ds_tr.map(sanitize).filter(lambda e: e["_ok_txt"] == 1).remove_columns(["_ok_txt"])
    ds_va = ds_va.map(sanitize).filter(lambda e: e["_ok_txt"] == 1).remove_columns(["_ok_txt"])

    # ---- 2) 预检查：分词后长度必须 > 0（含截断），否则剔除 ----
    max_prompt = int(args.max_prompt_length)
    max_total  = int(args.max_length)

    def tokenizable_ok(ex):
        # prompt 按 max_prompt 截断；chosen/rejected 以 max_total 为上限
        p = tok(ex["prompt"],   add_special_tokens=False, truncation=True, max_length=max_prompt)["input_ids"] or []
        c = tok(ex["chosen"],   add_special_tokens=False, truncation=True, max_length=max_total)["input_ids"]  or []
        r = tok(ex["rejected"], add_special_tokens=False, truncation=True, max_length=max_total)["input_ids"]  or []
        ok = (len(p) > 0 and len(c) > 0 and len(r) > 0
              and all(isinstance(x, int) for x in p)
              and all(isinstance(x, int) for x in c)
              and all(isinstance(x, int) for x in r))
        return {"_ok_tok": int(ok)}

    ds_tr = ds_tr.map(tokenizable_ok).filter(lambda e: e["_ok_tok"] == 1).remove_columns(["_ok_tok"])
    ds_va = ds_va.map(tokenizable_ok).filter(lambda e: e["_ok_tok"] == 1).remove_columns(["_ok_tok"])

    print("[DPO] train/eval sizes after tokenizable filter:", len(ds_tr), len(ds_va))
    print("[DPO] sample keys:", ds_tr.column_names)
    print("[DPO] sample record (truncated):",
          {k: (ds_tr[0][k][:60] + "..." if isinstance(ds_tr[0][k], str) and len(ds_tr[0][k])>60 else ds_tr[0][k])
           for k in ["prompt","chosen","rejected"]})

    # --- DPO config ---
    dpo_args = DPOConfig(
        output_dir=out_dir,
        optim="paged_adamw_8bit" ,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        eval_strategy=("steps" if args.eval_steps > 0 else "no"),
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,   # 必须 False
        report_to=[],
        beta=0.1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # --- 自定义 collator：把 TRL tokenize_row 的输出做“强健 pad”，清理 None ---
    def make_safe_collator(pad_id: int, label_pad_id: int = -100):
        def _to_int_list(x):
            if x is None: return []
            if isinstance(x, list):
                out = []
                for v in x:
                    if isinstance(v, int):
                        out.append(v)
                    elif v is None:
                        # 替换孤立 None
                        out.append(pad_id)
                    elif isinstance(v, float) and v.is_integer():
                        out.append(int(v))
                    else:
                        # 极端情况：丢弃非整型
                        continue
                return out
            # 非 list 的奇怪类型，直接丢弃
            return []

        def _collate(features):
            # 预期字段（TRL tokenize_row 通常会给出这些；缺的就按空来）
            fields = [
                "prompt_input_ids", "prompt_attention_mask",
                "chosen_input_ids", "chosen_attention_mask",
                "rejected_input_ids", "rejected_attention_mask",
                # 兼容部分版本会生成 labels；没有也不报错
                "chosen_labels", "rejected_labels",
            ]

            # 先转为纯 int list，统计每个 field 的最大长度
            clean_feats = []
            max_len = {f: 0 for f in fields}
            for ex in features:
                ex2 = {}
                for f in fields:
                    ex2[f] = _to_int_list(ex.get(f, None))
                    if len(ex2[f]) > max_len[f]:
                        max_len[f] = len(ex2[f])
                clean_feats.append(ex2)

            # 为缺失的 labels 用 input_ids 填充（DPO 不直接用它们算 loss，但有些版本会访问）
            for ex in clean_feats:
                if len(ex["chosen_labels"]) == 0:
                    ex["chosen_labels"] = ex["chosen_input_ids"][:]
                    if len(ex["chosen_labels"]) > 0:
                        # 用 -100 屏蔽 prompt 段：这里无法区分 prompt 长度，保守全部保留；对 DPO 影响极小
                        pass
                if len(ex["rejected_labels"]) == 0:
                    ex["rejected_labels"] = ex["rejected_input_ids"][:]

            # 某些 batch 内同一 field 的最大长度可能为 0（极端过滤后），给个最小值，避免 torch.stack 空张量
            for f in fields:
                if max_len[f] == 0:
                    max_len[f] = 1

            # pad 并转 tensor
            batch = {}
            for f in fields:
                seqs = []
                is_label = f.endswith("labels")
                pad_val = label_pad_id if is_label else (1 if f.endswith("attention_mask") else pad_id)
                for ex in clean_feats:
                    seq = ex[f]
                    if len(seq) < max_len[f]:
                        seq = seq + [pad_val] * (max_len[f] - len(seq))
                    elif len(seq) > max_len[f]:
                        seq = seq[:max_len[f]]
                    seqs.append(seq)
                dtype = torch.long
                batch[f] = torch.tensor(seqs, dtype=dtype)
            return batch
        return _collate

    data_collator = make_safe_collator(tok.pad_token_id, label_pad_id=-100)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        tokenizer=tok,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=data_collator,   # ★ 用我们的 collator，避开 TRL 的 None 问题
    )

    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[OK] DPO model saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_file", type=str, default=None)
    ap.add_argument("--val_file",   type=str, default=None)
    ap.add_argument("--out_dir",    type=str, default=None)

    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--lr",        type=float, default=5e-5)
    ap.add_argument("--grad_accum",type=int, default=8)

    ap.add_argument("--eval_steps",    type=int, default=0)
    ap.add_argument("--save_steps",    type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)

    ap.add_argument("--max_length",        type=int, default=256)
    ap.add_argument("--max_prompt_length", type=int, default=128)

    ap.add_argument("--lora_r",       type=int, default=16)
    ap.add_argument("--lora_alpha",   type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_only", type=int, default=1)
    main(ap.parse_args())
