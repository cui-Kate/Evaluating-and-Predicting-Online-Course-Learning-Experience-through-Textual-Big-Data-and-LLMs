# -*- coding: utf-8 -*-
"""
推断：对 Real Review.xlsx 中每条评论生成 0-10 分；输出 review_scores.csv
并按课程计算 mean 与 20%截尾均值，输出 course_scores.csv
"""
import argparse, re, json
import pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from utils.seeds import set_global_seed
from utils.robust_stats import trimmed_mean, winsorized_mean


from peft import PeftModel
from transformers import BitsAndBytesConfig







def extract_score(text: str):
    # 优先抓取JSON字段，再兜底抓纯数字
    m = re.search(r'\{[^{}]*"review_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text, flags=re.S)
    if m: return float(m.group(1))
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', text)
    return float(m.group(1)) if m else np.nan

def build_prompt(overview, review):
    return (
        "你是在线课程评价员。请基于【课程概述】与【用户评价】，"
        "输出JSON：{\"review_score\": <0-10数字，两位小数>}。\n"
        f"【课程概述】{overview}\n【用户评价】{review}\n"
        "仅输出JSON对象。"
    )

def main(args):
    set_global_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    out_dir  = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读表并合并
    rev = pd.read_excel(data_dir / "Real Review.xlsx")
    co  = pd.read_excel(data_dir / "Course Overview.xlsx")
    rev = rev.rename(columns={"用户id": "user_id", "发言文本": "text", "课程id": "course_id",
                              "情感分数": "sent_score", "情感倾向": "sent_label"})
    co  = co.rename(columns={"课程ID": "course_id", "课程概述": "overview"})
    df  = rev.merge(co, on="course_id", how="left")
    df  = df[["user_id","course_id","text","overview","sent_score","sent_label"]].copy()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    

    from utils.model_resolver import ensure_local_model

    # 1) 解析基座路径（优先用 --base_model；否则从 LoRA 目录里 adapter_config.json 里读）
    base_dir = args.base_model
    if base_dir is None:
        import json, os
        cfg_path = os.path.join(args.model_dir, "adapter_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_dir = adapter_cfg.get("base_model_name_or_path", None)
    if base_dir is None:
        raise ValueError("未能确定 base_model。请通过 --base_model 显式指定基座模型路径/名称。")

    base_dir = ensure_local_model(base_dir, allow_download=False)

    # 2) 按需用4bit加载基座
    bnb = None
    kwargs = {"device_map": "auto"}
    if int(getattr(args, "use_4bit", 1)) == 1:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"  # 你的 4060 支持 bf16；若出错可改 "float16"
        )
        kwargs["quantization_config"] = bnb

    base = AutoModelForCausalLM.from_pretrained(base_dir, **kwargs)

    # 3) 套上 LoRA 适配器
    model = PeftModel.from_pretrained(base, args.model_dir)
    model.eval()


    rows = []
    for i, r in df.iterrows():
        prompt = build_prompt(r["overview"], r["text"])
        ipt = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ipt, max_new_tokens=args.max_new_tokens, do_sample=False)
        s = tok.decode(out[0], skip_special_tokens=True)
        sc = extract_score(s)
        rows.append({
            "user_id": r["user_id"], "course_id": r["course_id"],
            "pred_score": sc, "raw_output": s
        })

    out_rev = pd.DataFrame(rows)
    out_rev.to_csv(out_dir / "review_scores.csv", index=False)

    # 课程聚合（均值 + 截尾 + 温莎化作为附录）
    agg = (out_rev.groupby("course_id")["pred_score"]
           .agg(mean_score="mean",
                trimmed_mean_20=lambda s: trimmed_mean(s, 0.1),
                winsorized_mean_20=lambda s: winsorized_mean(s, 0.1))
           .reset_index())
    agg.to_csv(out_dir / "course_scores.csv", index=False)
    print("[OK] Saved to out/review_scores.csv & out/course_scores.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="../out/sft_qwen2p5_7b")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_model", type=str, required=False, default=None, help="本地基座模型路径或HF名称")
    ap.add_argument("--use_4bit", type=int, default=1, help="推断时是否用4bit加载基座（默认1）")

    main(ap.parse_args())
