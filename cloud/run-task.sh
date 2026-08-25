#!/usr/bin/env bash
set -euo pipefail

task="${1:?usage: run-task.sh TASK [BATCH_SIZE]}"
batch_size="${2:-64}"
cd "$HOME/pickle50"
mkdir -p "results/$task" logs

export HF_HOME="$HOME/pickle50/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=true
worker_threads="${WORKER_THREADS:-$(nproc)}"
export OMP_NUM_THREADS="$worker_threads"
export MKL_NUM_THREADS="$worker_threads"
export OPENBLAS_NUM_THREADS="$worker_threads"

{
  echo "task=$task"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "batch_size=$batch_size"
} > "logs/$task.meta"

venv/bin/lm_eval run \
  --model hf \
  --model_args "pretrained=$HOME/pickle50/eval-balanced,trust_remote_code=True" \
  --tasks "$task" \
  --num_fewshot 0 \
  --batch_size "$batch_size" \
  --device cpu \
  --output_path "results/$task" \
  2>&1 | tee "logs/$task.log"

echo "finished_utc=$(date -u +%FT%TZ)" >> "logs/$task.meta"
touch "DONE-$task"
