#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/pickle50"
mkdir -p logs results

{
  echo "setup_started_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  lscpu
  free -h
} > logs/machine.txt

python3 -m venv venv
venv/bin/python -m pip install --upgrade pip wheel
venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0
venv/bin/pip install --no-cache-dir \
  lm_eval==0.4.12 \
  accelerate \
  transformers==4.52.3 \
  tokenmonster==1.1.12 \
  safetensors

export PYTHONPATH="$HOME/pickle50/tools"
venv/bin/python tools/verify_model.py artifacts/pickle-50m-balanced-refined.pklm \
  --json results/cloud-model-verify.json
venv/bin/python tools/export_hf.py artifacts/pickle-50m-balanced-refined.pklm \
  --out "$HOME/pickle50/eval-balanced"

{
  venv/bin/python --version
  venv/bin/python - <<'PY'
import lm_eval, torch, transformers
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"lm_eval={getattr(lm_eval, '__version__', '0.4.12')}")
PY
  echo "setup_finished_utc=$(date -u +%FT%TZ)"
} >> logs/machine.txt

touch READY
