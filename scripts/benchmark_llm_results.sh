#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

REPO_ID="Qwen/Qwen3-8B-FP8"
SEED=0

# Run compressed (dfloat)
python benchmark_llm_results.py --repo_id $REPO_ID --seed $SEED

# Run baseline (original)
python benchmark_llm_results.py --repo_id $REPO_ID --no_compress --seed $SEED
