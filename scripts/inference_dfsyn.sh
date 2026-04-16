export CUDA_VISIBLE_DEVICES=0

python inference_dfsyn.py --repo_id "black-forest-labs/FLUX.1-dev" --n_prompts 5 --seed 2025 --no_compress --offload
python inference_dfsyn.py --repo_id "black-forest-labs/FLUX.1-dev" --n_prompts 5 --seed 2025

REPO_ID_LIST=(
    "Qwen/Qwen-Image"
    "black-forest-labs/FLUX.1-dev"
    "Wan-AI/Wan2.1-T2V-14B"
    "Wan-AI/Wan2.2-T2V-A14B"
)

for REPO_ID in ${REPO_ID_LIST[@]}; do
    python inference_dfsyn.py --repo_id $REPO_ID --n_prompts 5 --seed 2025
    python inference_dfsyn.py --repo_id $REPO_ID --n_prompts 5 --seed 2025 --no_compress --offload
done
