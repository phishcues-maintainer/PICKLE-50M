#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/pickle50"
mkdir -p results/lm-eval-sable-q3 logs

worker_threads="${WORKER_THREADS:-$(nproc)}"
batch_size="${BATCH_SIZE:-64}"
export HF_HOME="$HOME/pickle50/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$worker_threads"
export MKL_NUM_THREADS="$worker_threads"
export OPENBLAS_NUM_THREADS="$worker_threads"

{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "worker_threads=$worker_threads"
  echo "batch_size=$batch_size"
  uname -a
  lscpu
  free -h
  venv/bin/python --version
} > logs/lm-eval-sable-q3.meta

venv/bin/lm_eval \
  --model hf \
  --model_args "pretrained=$HOME/pickle50/eval-sable-q3,dtype=float32" \
  --tasks piqa,hellaswag,arc_easy,arc_challenge \
  --num_fewshot 0 \
  --batch_size "$batch_size" \
  --device cpu \
  --output_path results/lm-eval-sable-q3 \
  2>&1 | tee logs/lm-eval-sable-q3.log

echo "finished_utc=$(date -u +%FT%TZ)" >> logs/lm-eval-sable-q3.meta
