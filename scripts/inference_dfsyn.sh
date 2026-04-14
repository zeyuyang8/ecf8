export CUDA_VISIBLE_DEVICES=0

REPO_ID_LIST=(
    "Qwen/Qwen-Image"
    "black-forest-labs/FLUX.1-dev"
    "Wan-AI/Wan2.1-T2V-14B"
    "Wan-AI/Wan2.2-T2V-A14B"
)

for REPO_ID in ${REPO_ID_LIST[@]}; do
    python inference_dfsyn.py --repo_id $REPO_ID --seed 2025
    python inference_dfsyn.py --repo_id $REPO_ID --seed 2025 --no_compress --offload
done
