# src/utils/model_resolver.py
# -*- coding: utf-8 -*-
import os, re
from pathlib import Path
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN", None)
# 项目根目录下的 models/
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _safe_dir(repo_id: str) -> str:
    # 把 repo_id 中的非法/分隔符替换为下划线，确保目录名稳定
    return re.sub(r"[^a-zA-Z0-9._\\-]+", "_", repo_id)

def _has_required_files(local_dir: Path) -> bool:
    if not local_dir.exists(): 
        return False
    files = {p.name for p in local_dir.iterdir() if p.is_file()}
    has_cfg = "config.json" in files
    has_tok = ("tokenizer.json" in files) or ({"vocab.json","merges.txt"} <= files)
    has_wt  = any(n.endswith(".safetensors") for n in files) or any(n.endswith(".bin") for n in files)
    return has_cfg and has_tok and has_wt

# def ensure_local_model(repo_id: str, revision: str = None, allow_download: bool = True) -> str:
#     """
#     返回本地模型目录。
#     - allow_download=False: 仅检查本地，缺失则抛错；绝不联网下载（纯离线）。
#     - allow_download=True: 先查本地，缺失则触发断点续传下载（更稳）。
#     """
#     local_dir = MODELS_DIR / _safe_dir(repo_id)
#     local_dir.mkdir(parents=True, exist_ok=True)

#     # 本地已就绪，直接返回
#     if _has_required_files(local_dir):
#         return str(local_dir)

#     # 纯离线模式：不就绪就报错
#     if not allow_download:
#         raise FileNotFoundError(f"[LocalOnly] 缺少模型文件：{local_dir}。请先手动放置基座权重。")

#     # 联网模式：更稳的断点续传设置（禁用 hf_transfer，限并发，只拉必要文件）
#     os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
#     snapshot_download(
#         repo_id=repo_id,
#         revision=revision,
#         local_dir=local_dir,
#         resume_download=True,
#         token=HF_TOKEN,
#         max_workers=4,
#         allow_patterns=[
#             "*.json","*.safetensors","*.bin","tokenizer*","vocab*","merges.txt",
#             "special_tokens_map.json","generation_config.json","*.model","*.txt"
#         ],
#     )
#     return str(local_dir)


# src/utils/model_resolver.py
from pathlib import Path
from huggingface_hub import snapshot_download
import os

def ensure_local_model(base_or_id: str, allow_download: bool = True) -> str:
    root = Path(__file__).resolve().parents[1]
    models_root = root / "models"

    p = Path(base_or_id)

    # 1) 绝对路径或现存相对路径：直接用
    if p.is_absolute():
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"[LocalOnly] 缺少模型文件：{p}")

    if (root / base_or_id).exists():
        return str(root / base_or_id)

    # 2) 否则当作 HF repo id，映射到 models/ 下
    local_dir = models_root / base_or_id.replace("/", "_")
    if local_dir.exists():
        return str(local_dir)

    if not allow_download:
        raise FileNotFoundError(f"[LocalOnly] 缺少模型文件：{local_dir}。请先手动放置基座权重。")

    # 3) 允许下载则断点续传到 local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=base_or_id, local_dir=str(local_dir), local_dir_use_symlinks=False, resume_download=True)
    return str(local_dir)
