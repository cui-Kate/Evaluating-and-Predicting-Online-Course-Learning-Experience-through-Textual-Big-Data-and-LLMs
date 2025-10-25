# 统一随机种子，保证复现（含PyTorch/CUDA）
import os, random, numpy as np
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms = False  # 兼顾速度
    except Exception:
        pass
