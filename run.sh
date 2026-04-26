#!/bin/bash
set -e

cd /home/huynguyenquang/DCM-TM

echo "=== Step 1: Preprocessing NIPS corpus ==="
uv run python scripts/preprocess_nips.py \
    --input data/NIPS_raw/papers.csv \
    --output-dir data/NIPS

echo "=== Step 2: Running main pipeline ==="
uv run python main.py --data-dir data/NIPS --device cuda
