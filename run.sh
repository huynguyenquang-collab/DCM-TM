#!/bin/bash
set -e

cd /home/huynguyenquang/DCM-TM

echo "=== Step 1: Preprocessing NIPS corpus ==="
uv run python scripts/preprocess_nips.py \
    --input data/NIPS_raw/papers.csv \
    --output-dir data/NIPS_processed

echo "=== Step 2: Training LLM-CoNTM ==="
uv run python run_llm_contm.py \
    --data-dir data/NIPS_processed \
    --output-dir outputs_llm_contm \
    --n-topics 50 \
    --epochs 80 \
    --device cuda

echo "=== Step 3: Evaluating final topics ==="
uv run python scripts/evaluation.py \
    --output-dir outputs_llm_contm \
    --data-dir data/NIPS \
    --top-n 10

echo "=== Done ==="
