# -*- coding: utf-8 -*-
"""
读取两份xlsx，合并→清洗→划分（按课程分组）→导出SFT与DPO用jsonl（稳健版）。
1) 自动规范列名：去空格/统一小写/替换特殊符号为下划线；
2) 双语&别名对齐：user_id/text/course_id/sent_score/sent_label, overview；
3) 严格断言必备列，打印实际列名，便于快速定位问题。
"""
import os, json, argparse, re
import pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

from utils.seeds import set_global_seed

# ---------- 工具函数 ----------
_NORM_SUB = re.compile(r"[\s\u3000/\\|:;,\-\u2013\u2014]+")  # 空格/全角空格/分隔符等 → _
def norm_col(c: str) -> str:
    if not isinstance(c, str): c = str(c)
    c = c.strip()
    c = c.replace("\ufeff", "")  # 处理BOM
    c = _NORM_SUB.sub("_", c)
    c = re.sub(r"_+", "_", c)
    return c.strip("_").lower()

# 多语言/别名映射到标准列名
ALIAS2STD = {
    # user_id
    "用户id": "user_id", "用户_id": "user_id", "userid": "user_id", "user id": "user_id",
    "user_id": "user_id",
    # text
    "发言文本": "text", "评论": "text", "评价文本": "text", "review": "text", "content": "text",
    "text": "text",
    # course_id
    "课程id": "course_id", "课程_id": "course_id", "course id": "course_id",
    "课程编号": "course_id", "课程代码": "course_id", "课程": "course_id",
    "course_id": "course_id", "course": "course_id",
    # sent_score
    "情感分数": "sent_score", "sentiment_score": "sent_score", "score": "sent_score",
    "label_score": "sent_score", "sent_score": "sent_score",
    # sent_label
    "情感倾向": "sent_label", "sentiment_label": "sent_label", "label": "sent_label",
    "sent_label": "sent_label",
    # overview
    "课程概述": "overview", "course_overview": "overview", "overview": "overview",
    "course_description": "overview", "描述": "overview"
}

REQ_REV = {"user_id", "text", "course_id", "sent_score", "sent_label"}
REQ_OV  = {"course_id", "overview"}

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 规范化列名
    cols_norm = [norm_col(c) for c in df.columns]
    df = df.copy()
    df.columns = cols_norm
    # 2) 别名对齐
    new_cols = []
    for c in df.columns:
        new_cols.append(ALIAS2STD.get(c, c))
    df.columns = new_cols
    return df

def linear_map_to_0_10(s, mn=None, mx=None):
    s = pd.to_numeric(s, errors="coerce")
    mn = np.nanmin(s) if mn is None else mn
    mx = np.nanmax(s) if mx is None else mx
    den = max(1e-8, (mx - mn))
    return np.clip((s - mn) / den * 10, 0, 10)

def build_sft_jsonl(df, outp):
    with open(outp, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            instr = (
                "你是在线课程评价员。请基于【课程概述】与【用户评价】，"
                "输出一个 JSON：{\"review_score\": <0-10的数字，两位小数>}\n"
                f"【课程概述】{r['overview']}\n【用户评价】{r['text']}\n"
                "仅输出JSON对象。"
            )
            out = json.dumps({"review_score": float(r["score_0_10"])}, ensure_ascii=False)
            rec = {"instruction": instr, "input": "", "output": out,
                   "course_id": r["course_id"], "user_id": r["user_id"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def build_dpo_jsonl(df, outp, max_pairs_per_course=200):
    pairs = []
    for cid, g in df.groupby("course_id"):
        g = g.sort_values("score_0_10").reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        lo, hi = g.iloc[: n//2], g.iloc[n//2 :]
        m = min(len(lo), len(hi), max_pairs_per_course)
        for i in range(m):
            better, worse = hi.iloc[i], lo.iloc[i]
            ctx = f"【课程概述】{better['overview']}\n"
            chosen   = f"{ctx}【用户评价】{better['text']}\n仅输出JSON：{{\"review_score\": <0-10数字>}}"
            rejected = f"{ctx}【用户评价】{worse['text']}\n仅输出JSON：{{\"review_score\": <0-10数字>}}"
            pairs.append({"prompt": "你是课程评价打分助手。", "chosen": chosen, "rejected": rejected})
    with open(outp, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


