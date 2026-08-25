#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/pickle50"
mkdir -p results/lm-eval-balanced logs

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
} > logs/lm-eval-balanced.meta

venv/bin/lm_eval run \
  --model hf \
  --model_args "pretrained=$HOME/pickle50/eval-balanced,trust_remote_code=True" \
  --tasks piqa hellaswag arc_easy arc_challenge \
  --num_fewshot 0 \
  --batch_size "$batch_size" \
  --device cpu \
  --output_path results/lm-eval-balanced \
  2>&1 | tee logs/lm-eval-balanced.log

echo "finished_utc=$(date -u +%FT%TZ)" >> logs/lm-eval-balanced.meta
