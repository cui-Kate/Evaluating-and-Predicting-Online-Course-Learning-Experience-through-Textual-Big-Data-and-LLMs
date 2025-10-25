import numpy as np

def trimmed_mean(x, proportion_to_cut=0.1):
    """两端各截 proportion_to_cut，默认0.1→总20%截尾。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0: return np.nan
    x.sort()
    k = int(len(x) * proportion_to_cut)
    if len(x) - 2*k <= 0: return np.nan
    return float(x[k: len(x)-k].mean())

def winsorized_mean(x, proportion_to_cut=0.1):
    """两端各截并压回边界（温莎化），用于附录敏感性分析。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0: return np.nan
    x.sort()
    k = int(len(x) * proportion_to_cut)
    lo, hi = x[k], x[-k-1] if (len(x)-k-1) >= 0 else x[-1]
    x = np.clip(x, lo, hi)
    return float(x.mean())
