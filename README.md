# ECF8

[ICLR 2026] Official implementation of "To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression with Exponent Concentration"

[[OpenReview](https://openreview.net/forum?id=XI1CeufywD)] [[arXiv]](https://arxiv.org/abs/2510.02676)

## Install

Install the dependencies.

```bash
conda create -n ecf python=3.12
conda activate ecf
pip install nv
uv pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/modelscope/DiffSynth-Studio.git@084bc2f
pip install -e . && rm -rf dfloat.egg-info
```

## Reproducibility

This is an example of running Qwen/Qwen3-8B-FP8.

```bash
cd scripts
REPO_ID=Qwen/Qwen3-8B-FP8

# Step 1: Compress the model using DFloat encoding and validate CUDA decoding correctness
python compress.py --repo_id $REPO_ID --save_model --n_processes 16 --validate_cuda

# Step 2: Measure inference speed (tokens/sec, latency, GPU memory) for compressed vs original
CUDA_VISIBLE_DEVICES=0 python inference_llm.py --repo_id $REPO_ID --batch_size 16 --num_tokens 1024 --write_csv --filename inference_llm_h100.csv
CUDA_VISIBLE_DEVICES=0 python inference_llm.py --repo_id $REPO_ID --batch_size 16 --num_tokens 1024 --no_compress --write_csv --filename inference_llm_h100.csv

# Step 3: Benchmark downstream task accuracy for compressed vs original
bash benchmark_llm_results.sh
```

To run more models, please refer to the `scripts/compress.sh`, `scripts/inference_llm.sh`, and `scripts/inference_dfsyn.sh`.

## Results

| repo_id | model_type | boolq_mean | boolq_stderr |
|---------|------------|-------|--------------|
| Qwen/Qwen3-8B-FP8 | compressed | 0.82 | 0.0386 |
| Qwen/Qwen3-8B-FP8 | original | 0.82 | 0.0386 |
