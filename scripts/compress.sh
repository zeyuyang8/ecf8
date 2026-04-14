# LLMs
REPO_ID_LIST=(
    "Qwen/Qwen3-8B-FP8"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
)

for repo_id in ${REPO_ID_LIST[@]}; do
    python compress.py --repo_id $repo_id --save_model --n_processes 16 --validate_cuda
    echo "Done for $repo_id"
done

# DMs
REPO_ID_LIST=(
    "black-forest-labs/FLUX.1-dev"
    "Wan-AI/Wan2.1-T2V-14B"
    "Wan-AI/Wan2.2-T2V-A14B"
    "Qwen/Qwen-Image"
)

for repo_id in ${REPO_ID_LIST[@]}; do
    python compress.py --repo_id $repo_id --save_model --n_processes 16 --validate_cuda
    echo "Done for $repo_id"
done

echo "Exiting script."
exit 0

LARGE_REPO_ID_LIST=(
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    "deepseek-ai/DeepSeek-R1-0528"
)

for repo_id in ${LARGE_REPO_ID_LIST[@]}; do
    python compress.py --repo_id $repo_id --save_model --n_processes 16 --validate_cuda
    echo "Done for $repo_id"
done
