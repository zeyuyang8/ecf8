# 1xGH200 (96 GB)
export CUDA_VISIBLE_DEVICES=0

REPO_ID_LIST=(
    "Qwen/Qwen3-8B-FP8"
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
)

for REPO_ID in ${REPO_ID_LIST[@]}; do
    for BATCH_SIZE in 1 2 4 8 16 32; do
        for NUM_TOKENS in 512 1024 2048 4096; do
            python inference_llm.py --repo_id $REPO_ID --batch_size $BATCH_SIZE --num_tokens $NUM_TOKENS --write_csv --filename inference_llm_gh200.csv
            python inference_llm.py --repo_id $REPO_ID --batch_size $BATCH_SIZE --num_tokens $NUM_TOKENS --no_compress --write_csv --filename inference_llm_gh200.csv
        done
    done
done

python inference_llm.py --repo_id RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --batch_size 48 --num_tokens 1024 --write_csv --filename inference_llm_gh200.csv
python inference_llm.py --repo_id Qwen/Qwen3-8B-FP8 --batch_size 16 --num_tokens 1024 --write_csv --filename inference_llm_gh200.csv

# 8xH200 (141 GB)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

REPO_ID_LIST=(
    "deepseek-ai/DeepSeek-R1-0528"
)

for REPO_ID in ${REPO_ID_LIST[@]}; do
    for BATCH_SIZE in 32 16 8 4 2 1; do
        for NUM_TOKENS in 4096 2048 1024 512; do
            python inference_llm.py --repo_id $REPO_ID --batch_size $BATCH_SIZE --num_tokens $NUM_TOKENS --write_csv --filename inference_llm_h200.csv
            python inference_llm.py --repo_id $REPO_ID --batch_size $BATCH_SIZE --num_tokens $NUM_TOKENS --no_compress --write_csv --filename inference_llm_h200.csv
        done
    done
done

python inference_llm.py --repo_id deepseek-ai/DeepSeek-R1-0528 --batch_size 1 --num_tokens 512 --write_csv --filename inference_llm_h200.csv

# 4xH200 (141 GB)
export CUDA_VISIBLE_DEVICES=0,1,2,3

REPO_ID_LIST=(
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
)

for REPO_ID in ${REPO_ID_LIST[@]}; do
    for BATCH_SIZE in 32 16 8 4 2 1; do
        for NUM_TOKENS in 4096 2048 1024 512; do
            python inference_llm.py --repo_id $REPO_ID --batch_size $BATCH_SIZE --num_tokens $NUM_TOKENS --write_csv --filename inference_llm_h200.csv
            python inference_llm.py --repo_id $REPO_ID --batch_size $BATCH_SIZE --num_tokens $NUM_TOKENS --no_compress --write_csv --filename inference_llm_h200.csv
        done
    done
done

python inference_llm.py --repo_id Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --batch_size 64 --num_tokens 1024 --write_csv --filename inference_llm_h200.csv
