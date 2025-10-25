# -*- coding: utf-8 -*-
"""
评测：将预测分与 test_meta（或全量的 sent_score）对齐，输出MAE/RMSE/R2/Spearman/Pearson，
并给出与情感三分类的一致性（方向正确率/AUC）。
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from scipy.stats import spearmanr, pearsonr

def direction_accuracy(y_true, y_pred, thr=None):
    # thr 为空则用 y_true 的中位数作为正负阈
    if thr is None: thr = np.median(y_true)
    t = (y_true >= thr).astype(int)
    p = (y_pred >= thr).astype(int)
    return (t == p).mean()

def main(args):
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "out"

    pred = pd.read_csv(out_dir / "review_scores.csv")
    # 若存在 test_meta（prepare_dataset产生），优先使用；否则用全量 Real Review 的 sent_score
    data_proc = root / "data_proc" / "test_meta.csv"
    if data_proc.exists():
        gt = pd.read_csv(data_proc)
        gt = gt[["user_id","course_id","sent_score","sent_label"]]
    else:
        rv = pd.read_excel(root / "data" / "Real Review.xlsx")
        rv = rv.rename(columns={"用户id":"user_id","课程id":"course_id","情感分数":"sent_score","情感倾向":"sent_label"})
        gt = rv[["user_id","course_id","sent_score","sent_label"]]

    df = pred.merge(gt, on=["user_id","course_id"], how="inner").dropna(subset=["pred_score","sent_score"])
    # 将原 sent_score 线性映射到 0-10 作为金标（如另有人工金标，请替换此列）
    s = df["sent_score"].astype(float)
    y = ( (s - s.min()) / max(1e-8, (s.max()-s.min())) * 10.0 ).clip(0,10)
    yhat = df["pred_score"].astype(float).clip(0,10)

    mae = mean_absolute_error(y, yhat)
    rmse= mean_squared_error(y, yhat, squared=False)
    r2  = r2_score(y, yhat)
    sp  = spearmanr(y, yhat).correlation
    pe  = pearsonr(y, yhat).statistic

    # 与三分类方向一致性（把 正/中/负 → pos=1/neg=0，中性忽略或按>=中位数判）
    df2 = df.copy()
    df2["bin_true"] = (y >= np.median(y)).astype(int)
    df2 = df2.dropna(subset=["bin_true","pred_score"])
    dir_acc = direction_accuracy(y, yhat)
    try:
        auc = roc_auc_score(df2["bin_true"], yhat)
    except Exception:
        auc = np.nan

    res = pd.DataFrame([{
        "MAE": mae, "RMSE": rmse, "R2": r2, "Spearman": sp, "Pearson": pe,
        "DirAcc": dir_acc, "AUC_bin": auc, "N": len(df)
    }])
    res.to_csv(out_dir / "eval_metrics.csv", index=False)
    print(res.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
