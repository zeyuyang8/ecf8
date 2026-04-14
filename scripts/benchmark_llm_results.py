# flake8: noqa: E402

import argparse
import os

# --- CONFIGURATION: Set Cache Directory ---
os.environ["HF_DATASETS_CACHE"] = "./data"

import csv

import toml
import torch

# --- LM Evaluation Harness Imports ---
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# --- Model Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from dfloat.run import DFloatModel, get_dfloat_model_name_or_path


def load_baseline_model(args):
    print(f"Loading Baseline Model: {args.repo_id}")
    trust_remote_code = True

    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.repo_id, trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_compressed_model(args):
    config = toml.load(args.config)
    dfloat_dir = config["download"]["dfloat_dir"]
    dfloat_repo_id, dfloat_model_id = get_dfloat_model_name_or_path(
        args.repo_id, dfloat_dir
    )

    print(f"Loading Compressed Model: {dfloat_repo_id}")
    trust_remote_code = True

    model = DFloatModel.from_pretrained(
        dfloat_repo_id,
        dfloat_model_id,
        args.repo_id,
        device="cuda",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.repo_id, trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-8B-FP8")
    parser.add_argument("--config", type=str, default="./config.toml")
    parser.add_argument(
        "--tasks", type=str, default="boolq", help="Comma separated tasks"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=str, default="32")
    parser.add_argument(
        "--limit",
        type=float,
        default=100,
        help="Limit samples per task for speed",
    )
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="Use baseline model (no compression)",
    )
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    if args.batch_size.lower() == "auto":
        batch_size_arg = "auto"
    else:
        batch_size_arg = int(args.batch_size)

    # Load model
    if args.no_compress:
        model, tokenizer = load_baseline_model(args)
        model_tag = "original"
    else:
        model, tokenizer = load_compressed_model(args)
        model_tag = "compressed"

    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size_arg,
        trust_remote_code=True,
    )

    task_list = args.tasks.split(",")
    set_seed(args.seed)

    # Evaluate and collect results
    # row: repo_id, model_type, task1_score, task2_score, ...
    row = {"repo_id": args.repo_id, "model_type": model_tag}

    for task_name in task_list:
        print(f"\n=== Running Task: {task_name} ({model_tag}) ===")

        results = simple_evaluate(
            model=lm_obj,
            tasks=[task_name],
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            numpy_random_seed=args.seed,
        )

        task_results = results["results"].get(task_name, {})
        # Extract primary accuracy and its stderr
        acc = task_results.get("acc,none", task_results.get("exact_match,none"))
        acc_stderr = task_results.get(
            "acc_stderr,none", task_results.get("exact_match_stderr,none")
        )
        row[task_name] = acc
        row[f"{task_name}_stderr"] = acc_stderr
        print(f"  {task_name} ({model_tag}): {acc} ± {acc_stderr}")

    # Write CSV (append if file exists)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "accuracy.csv")

    file_exists = os.path.exists(csv_path)
    fieldnames = ["repo_id", "model_type"]
    for t in task_list:
        fieldnames.extend([t, f"{t}_stderr"])
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nResults appended to {csv_path}")


if __name__ == "__main__":
    main()
