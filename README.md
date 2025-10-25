# Evaluating-and-Predicting-Online-Course-Learning-Experience-through-Textual-Big-Data-and-LLMs
code of Evaluating and Predicting Online Course Learning Experience through Textual Big Data and LLMs


# 0) 安装依赖（PyTorch 按环境安装，这里略）
pip install "transformers==4.44.2" "peft==0.12.0" "bitsandbytes==0.43.1" \
            "trl==0.10.1" "accelerate==0.34.2" "datasets==2.20.0" \
            "pandas==2.2.2" "openpyxl==3.1.5" "scipy==1.13.1" "scikit-learn==1.5.1" "huggingface_hub==0.24.5"

# 1) 放置数据
#   project_root/data/Course Overview.xlsx
#   project_root/data/Real Review.xlsx

# 2) 生成数据集与切分（固定随机种子）
python src\prepare_dataset.py --sft_max_per_course 5 --dpo_max_pairs_per_course 4 --seed 42


# 3) 训练和评测

## 1) Qwen2.5-7B-Instruct（SFT）

Windows（PowerShell / CMD 通用）

python src\train_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --out_dir out\sft_qwen7b


python src\infer_and_aggregate.py --model_dir out\sft_qwen7b


python src\evaluate.py


mkdir out\res\sft_qwen7b


move out\review_scores.csv out\res\sft_qwen7b\


move out\course_scores.csv out\res\sft_qwen7b\


move out\eval_metrics.csv  out\res\sft_qwen7b\


## 2) Qwen2.5-7B-Instruct（DPO）
python src\train_dpo.py --base_model Qwen/Qwen2.5-7B-Instruct --out_dir out\dpo_qwen7b

python src\infer_and_aggregate.py --model_dir out\dpo_qwen7b

python src\evaluate.py


mkdir out\res\dpo_qwen7b

move out\review_scores.csv out\res\dpo_qwen7b\

move out\course_scores.csv out\res\dpo_qwen7b\

move out\eval_metrics.csv  out\res\dpo_qwen7b\




## 3) DeepSeek-LLM-7B-Chat（SFT）

python src\train_sft.py --base_model deepseek-ai/deepseek-llm-7b-chat --out_dir out\sft_deepseek7b


python src\infer_and_aggregate.py --model_dir out\sft_deepseek7b

python src\evaluate.py

mkdir out\res\sft_deepseek7b


move out\review_scores.csv out\res\sft_deepseek7b\

move out\course_scores.csv out\res\sft_deepseek7b\


move out\eval_metrics.csv  out\res\sft_deepseek7b\


## 4) DeepSeek-LLM-7B-Chat（DPO）

python src\train_dpo.py --base_model deepseek-ai/deepseek-llm-7b-chat --out_dir out\dpo_deepseek7b

python src\infer_and_aggregate.py --model_dir out\dpo_deepseek7b

python src\evaluate.py

mkdir out\res\dpo_deepseek7b

move out\review_scores.csv out\res\dpo_deepseek7b\


move out\course_scores.csv out\res\dpo_deepseek7b\


move out\eval_metrics.csv  out\res\dpo_deepseek7b\




## 5) Llama-3.1-8B-Instruct（SFT）

若显存不足可把 train_sft.py 的 --lora_r 8、--grad_accum 调大

python src\train_sft.py --base_model meta-llama/Llama-3.1-8B-Instruct --out_dir out\sft_llama8b

python src\infer_and_aggregate.py --model_dir out\sft_llama8b

python src\evaluate.py

mkdir out\res\sft_llama8b

move out\review_scores.csv out\res\sft_llama8b\

move out\course_scores.csv out\res\sft_llama8b\

move out\eval_metrics.csv  out\res\sft_llama8b\


## 6) Llama-3.1-8B-Instruct（DPO）

python src\train_dpo.py --base_model meta-llama/Llama-3.1-8B-Instruct --out_dir out\dpo_llama8b

python src\infer_and_aggregate.py --model_dir out\dpo_llama8b

python src\evaluate.py

mkdir out\res\dpo_llama8b


move out\review_scores.csv out\res\dpo_llama8b\


move out\course_scores.csv out\res\dpo_llama8b\


move out\eval_metrics.csv  out\res\dpo_llama8b\


## 7)可选：一键跑完“训练→推断→评测”的组合（示例）

Windows（PowerShell）


$combos = @(
  @{tag="sft_qwen7b";     base="Qwen/Qwen2.5-7B-Instruct";             mode="sft"},
  @{tag="dpo_qwen7b";     base="Qwen/Qwen2.5-7B-Instruct";             mode="dpo"},
  @{tag="sft_deepseek7b"; base="deepseek-ai/deepseek-llm-7b-chat";     mode="sft"},
  @{tag="dpo_deepseek7b"; base="deepseek-ai/deepseek-llm-7b-chat";     mode="dpo"},
  @{tag="sft_llama8b";    base="meta-llama/Llama-3.1-8B-Instruct";     mode="sft"},
  @{tag="dpo_llama8b";    base="meta-llama/Llama-3.1-8B-Instruct";     mode="dpo"}
)




foreach ($c in $combos) {
  if ($c.mode -eq "sft") { python src\train_sft.py --base_model $c.base --out_dir ("out\"+$c.tag) }
  else                   { python src\train_dpo.py --base_model $c.base --out_dir ("out\"+$c.tag) }
  python src\infer_and_aggregate.py --model_dir ("out\"+$c.tag)
  python src\evaluate.py
  New-Item -ItemType Directory -Force -Path ("out\res\"+$c.tag) | Out-Null
  Move-Item out\review_scores.csv ("out\res\"+$c.tag+"\")
  Move-Item out\course_scores.csv ("out\res\"+$c.tag+"\")
  Move-Item out\eval_metrics.csv  ("out\res\"+$c.tag+"\")
}
