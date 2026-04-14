import os
import time
from argparse import ArgumentParser

import toml
import torch
import wandb

try:
    from dfloat.run import DFloatModel, get_dfloat_model_name_or_path
except ImportError:
    print("DFloat is not installed. Must run this script with flag `--no_compress`")
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.toml")
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-8B-FP8")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Question: What is a binary tree and its applications? Answer:",
    )
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--write_csv", action="store_true")
    parser.add_argument("--filename", type=str, default="inference_llm.csv")
    args = parser.parse_args()

    wandb.init(project="llm-inference", config=args)

    # Check for FlashAttention 2 availability
    attn_implementation = None

    # Load configuration from TOML file
    config = toml.load(args.config)

    # Get model paths from configuration
    dfloat_dir = config["download"]["dfloat_dir"]
    dfloat_repo_id, dfloat_model_id = get_dfloat_model_name_or_path(
        args.repo_id, dfloat_dir
    )

    print(f"DFloat model path: {dfloat_repo_id}")
    print(f"Local model path: {args.repo_id}")
    print(f"No compress: {args.no_compress}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, args.filename)
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        no_compress, repo_id, num_tokens, batch_size = (
            args.no_compress,
            args.repo_id,
            args.num_tokens,
            args.batch_size,
        )
        no_compress_tag = "original" if args.no_compress else "compressed"
        # Check if the row exists
        row = df[
            (df["no_compress"] == no_compress_tag)
            & (df["repo_id"] == repo_id)
            & (df["num_tokens"] == num_tokens)
            & (df["batch_size"] == batch_size)
        ]
        if len(row) > 0:
            print(f"Results for {args.repo_id} already exist in {results_file}")
            print(row)
            exit()
        else:
            print(
                f"Getting new results for {args.repo_id} with `no_compress` {no_compress_tag}, `num_tokens` {num_tokens}, `batch_size` {batch_size}"
            )

    if args.no_compress:
        model = AutoModelForCausalLM.from_pretrained(
            args.repo_id,
            device_map="auto",
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # Use low memory usage mode
        )
    else:
        if args.repo_id == "deepseek-ai/DeepSeek-R1-0528":
            trust_remote_code = False
        else:
            trust_remote_code = True
        model = DFloatModel.from_pretrained(
            dfloat_repo_id,
            dfloat_model_id,
            args.repo_id,  # This is needed for FP8 models
            device_map="auto",
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
        )

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Warm-up pass to compile kernel and avoid cold start latency
    prompt = " ".join(["a"] * 128)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    del inputs
    del outputs

    # Set random seed for deterministic sampling
    set_seed(args.seed)

    # Prepare batch of prompts
    prompts = [args.prompt] * args.batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Generate output and measure latency
    with torch.no_grad():
        start_time = time.time()
        output = model.generate(
            **inputs,
            max_new_tokens=args.num_tokens,
            do_sample=True,  # Enables sampling; set to False for greedy
        )
        torch.cuda.synchronize()
        end_time = time.time()

    # Decode generated tokens and compute throughput
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
    latency = end_time - start_time

    # GPU memory tracking
    allocated = 0
    peak_allocated = 0
    for device_id in range(torch.cuda.device_count()):
        allocated += torch.cuda.memory_allocated(device_id)
        peak_allocated += torch.cuda.max_memory_allocated(device_id)

    allocated /= 1024**2  # Convert to MB
    peak_allocated /= 1024**2  # Convert to MB

    # Print generated results and generation speed
    print("Generated Texts:")
    for i, text in enumerate(generated_texts):
        print(f"[Sample {i+1}]: {text}")
    print(f"Decoding Latency for {args.num_tokens} tokens: {latency:.4f} seconds")
    print(f"Tokens per second: {args.num_tokens * args.batch_size / latency:.2f}")
    print(f"GPU Memory Allocated: {allocated:.2f} MB")
    print(f"GPU Peak Memory Usage: {peak_allocated:.2f} MB")

    no_compress_tag = "original" if args.no_compress else "compressed"

    # Open a csv file and write the results
    if args.write_csv:
        os.makedirs("results", exist_ok=True)
        with open(f"results/{args.filename}", "a") as f:
            # Write header if file doesn't exist or is empty
            file_is_empty = (
                not os.path.exists(f"results/{args.filename}")
                or os.path.getsize(f"results/{args.filename}") == 0
            )

            if file_is_empty:
                f.write(
                    "no_compress,repo_id,num_tokens,batch_size,latency,tokens_per_second,"
                    "gpu_memory_allocated_mb,gpu_peak_memory_usage_mb\n"
                )
            f.write(
                f"{no_compress_tag},{args.repo_id},{args.num_tokens},{args.batch_size},"
                f"{latency:.4f},{args.num_tokens * args.batch_size / latency:.2f},"
                f"{allocated:.2f},{peak_allocated:.2f}\n"
            )

    wandb.log(
        {
            "latency": latency,
            "tokens_per_second": args.num_tokens * args.batch_size / latency,
            "gpu_memory_allocated_mb": allocated,
            "gpu_peak_memory_usage_mb": peak_allocated,
        }
    )
    wandb.finish()
