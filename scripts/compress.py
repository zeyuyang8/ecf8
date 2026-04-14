import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.toml")
parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-8B-FP8")
parser.add_argument("--bytes_per_thread", type=int, default=8)
parser.add_argument("--threads_per_block", type=int, default=512)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--n_processes", type=int, default=36)
parser.add_argument("--sequential", action="store_true")
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--validate_cuda", action="store_true")
parser.add_argument("--thread_pool", action="store_true")

args = parser.parse_args()

import os

DEFAULT_N_THREADS = 1
os.environ["OPENBLAS_NUM_THREADS"] = f"{DEFAULT_N_THREADS}"
os.environ["MKL_NUM_THREADS"] = f"{DEFAULT_N_THREADS}"
os.environ["OMP_NUM_THREADS"] = f"{DEFAULT_N_THREADS}"

import json
import operator
import pickle
import re

import toml
import torch
import torch.nn as nn

torch.set_num_threads(1)  # Intra-op parallelism
torch.set_num_interop_threads(1)  # Inter-op parallelism

from multiprocessing import get_context

import numpy as np
import pandas as pd

# from dahuffman import HuffmanCodec
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if args.thread_pool:
    from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor
else:
    from concurrent.futures import ProcessPoolExecutor
import time

from dfloat.const import get_fp8_pattern_dict
from dfloat.run import _decode, get_dfloat_model_name_or_path
from dfloat.utils import (
    encode_exponents_for_cuda,
    find_layers_by_data_type_and_layer_type,
    find_layers_by_selection,
    flatten_block_of_weights,
    get_exponents_and_other_4bits_fp8_e4m3,
    get_luts_from_huffman_table,
    get_nbit_codec,
    locate_block_by_regex,
)
from diffsynth.core.vram.layers import AutoWrappedLinear
from safetensors.torch import load_file, save_file
from transformers.integrations.finegrained_fp8 import FP8Linear

package_version = "0.2.0"


def validate_cuda_decoding_fp8(result, tensors, device):
    flat_weights = tensors["flat_weights"]
    luts = tensors["luts"]
    encoded = tensors["encoded"]
    packed_other_4bits = tensors["packed_other_4bits"]
    output_positions = tensors["output_positions"]
    gaps = tensors["gaps"]

    threads_per_block = result["threads_per_block"]
    bytes_per_thread = result["bytes_per_thread"]

    # Get some statistics
    n_luts = luts.shape[0]
    n_elements = packed_other_4bits.numel() * 2
    n_bytes = encoded.numel()

    # CUDA validation
    blocks_per_grid = (int(np.ceil(n_bytes / (threads_per_block * bytes_per_thread))),)
    output_positions_np = output_positions.view(torch.uint64).numpy()
    elements_per_block = output_positions_np[1:] - output_positions_np[:-1]
    max_elements_per_block = elements_per_block.max().item()
    shared_mem_size = threads_per_block * 4 + 4 + max_elements_per_block * 1

    # Move to GPU
    (
        cuda_luts,
        cuda_encoded,
        cuda_packed_other_4bits,
        cuda_output_positions,
        cuda_gaps,
    ) = list(
        map(
            lambda x: x.to(device),
            [luts, encoded, packed_other_4bits, output_positions, gaps],
        )
    )
    cuda_outputs = torch.empty(n_elements, dtype=torch.float8_e4m3fn, device=device)

    # Decoding
    _decode(
        grid=blocks_per_grid,
        block=(threads_per_block,),
        shared_mem=shared_mem_size,
        args=[
            cuda_luts.data_ptr(),
            cuda_encoded.data_ptr(),
            cuda_packed_other_4bits.data_ptr(),
            cuda_output_positions.data_ptr(),
            cuda_gaps.data_ptr(),
            cuda_outputs.data_ptr(),
            n_luts,
            n_bytes,
            n_elements,
        ],
    )

    assert (flat_weights == cuda_outputs.cpu()).all().item()
    # print('Decoding is correct')


def process_fp8_block(block_item, info_dict, print_time=False):
    block_name, layers = block_item

    save_path = info_dict["save_path"]
    model_fp8_regex = info_dict["model_fp8_regex"]
    bytes_per_thread = info_dict["bytes_per_thread"]
    threads_per_block = info_dict["threads_per_block"]

    cache_dir = os.path.join(save_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    block_save_path = os.path.join(cache_dir, block_name + ".safetensors")
    metadata_save_path = os.path.join(cache_dir, block_name + ".pkl")
    layer_names = list(layers.keys())
    layers_to_delete = []
    for idx in range(len(layer_names)):
        layer_name = layer_names[idx]
        try:
            assert layer_name.startswith(block_name + ".") or layer_name == block_name
            layers_to_delete.append(layer_name)
        except AssertionError:
            print(f"layer name: {layer_name} block name: {block_name}")
            raise Exception("Layer name mismatch")
        layer_name = layer_name[len(block_name) + 1 :]
        layer_names[idx] = layer_name

    # Find which regex matched (don't modify global variable here)
    matched_regex = None
    for regex in model_fp8_regex:
        if re.fullmatch(regex, block_name):
            matched_regex = regex
            break

    _entropy = None
    if os.path.exists(block_save_path) and os.path.exists(metadata_save_path):
        with open(metadata_save_path, "rb") as f:
            _metadata = pickle.load(f)
        if "entropy" in _metadata:
            _entropy = _metadata["entropy"]
        else:
            tensors = load_file(block_save_path)
            flat_weights = tensors["flat_weights"]
            exponents, packed_other_4bits = get_exponents_and_other_4bits_fp8_e4m3(
                flat_weights
            )
            vals, freqs = torch.unique(exponents, return_counts=True)
            probabilities = freqs / freqs.sum()
            _entropy = -torch.sum(probabilities * torch.log2(probabilities)).item()
    elif os.path.exists(block_save_path) and not os.path.exists(metadata_save_path):
        tensors = load_file(block_save_path)
        flat_weights = tensors["flat_weights"]
        exponents, packed_other_4bits = get_exponents_and_other_4bits_fp8_e4m3(
            flat_weights
        )
        vals, freqs = torch.unique(exponents, return_counts=True)
        probabilities = freqs / freqs.sum()
        _entropy = -torch.sum(probabilities * torch.log2(probabilities)).item()
    else:
        # Flatten the weights and get the exponents and other 4 bits
        t_start = time.time()
        flat_weights, split_positions = flatten_block_of_weights(
            layers, dtype=torch.float8_e4m3fn
        )
        if print_time:
            print(f"Flattening took {time.time() - t_start} seconds")

        # print(f'block name: {block_name} split positions: {split_positions}')
        t_start = time.time()
        exponents, packed_other_4bits = get_exponents_and_other_4bits_fp8_e4m3(
            flat_weights
        )
        if print_time:
            print(f"Exponent extraction took {time.time() - t_start} seconds")

        # Get frequencies of unique exponents and calculate entropy
        t_start = time.time()
        vals, freqs = torch.unique(exponents, return_counts=True)
        probabilities = freqs / freqs.sum()
        _entropy = -torch.sum(probabilities * torch.log2(probabilities)).item()
        # print(f'entropy: {_entropy}')
        if print_time:
            print(f"Entropy calculation took {time.time() - t_start} seconds")

        # Get Huffman tree
        _counter = {}
        for v, f in zip(vals.tolist(), freqs.tolist()):
            _counter[v] = f
        # print(_counter)

        # print('Getting Huffman tree')
        t_start = time.time()
        # _codec = HuffmanCodec.from_frequencies(_counter)
        codec, counter, table = get_nbit_codec(_counter, max_n_bits=16)
        # codec.print_code_table()
        if print_time:
            print(f"Huffman tree took {time.time() - t_start} seconds")

        # Get LUTs from Huffman table for decoding
        luts = get_luts_from_huffman_table(table)
        t_start = time.time()
        encoded, gaps, output_positions = encode_exponents_for_cuda(
            exponents,
            codec,
            bytes_per_thread,
            threads_per_block,
        )
        if print_time:
            print(f"Encoding took {time.time() - t_start} seconds")

        # Return everything for CUDA validation in main process
        tensors = {
            "flat_weights": flat_weights,
            "luts": luts,
            "encoded": encoded,
            "packed_other_4bits": packed_other_4bits,
            "output_positions": output_positions.view(torch.uint8),
            "gaps": gaps,
            "split_positions": split_positions,
        }
        save_file(tensors, block_save_path)
        del flat_weights, exponents, packed_other_4bits, vals, freqs, probabilities
        del luts, encoded, gaps, output_positions, split_positions, tensors

    metadata = {
        "block_name": block_name,
        "layer_names": layer_names,
        "matched_regex": matched_regex,
        "layers_to_delete": layers_to_delete,  # Return layer names to delete
        "threads_per_block": threads_per_block,
        "bytes_per_thread": bytes_per_thread,
        "block_save_path": block_save_path,
        "entropy": _entropy,
    }
    # Save metadata as pickle
    with open(metadata_save_path, "wb") as f:
        pickle.dump(metadata, f)

    import gc

    gc.collect()

    return metadata


def _worker(args):
    # args is ((block_name, layers), info_dict)
    return process_fp8_block(*args)


def process_fp8_block_metadata(block_name, info_dict):
    save_path = info_dict["save_path"]
    model_fp8_regex = info_dict["model_fp8_regex"]

    cache_dir = os.path.join(save_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    metadata_save_path = os.path.join(cache_dir, block_name + ".pkl")

    # Find which regex matched (don't modify global variable here)
    matched_regex = None
    for regex in model_fp8_regex:
        if re.fullmatch(regex, block_name):
            matched_regex = regex
            break

    with open(metadata_save_path, "rb") as f:
        metadata = pickle.load(f)
    metadata["matched_regex"] = matched_regex

    with open(metadata_save_path, "wb") as f:
        pickle.dump(metadata, f)

    return metadata


def process_all_fp8_blocks_parallel(
    blocks_dict, num_processes, reverse=False, info_dict=None
):
    if not reverse:
        block_items = list(blocks_dict.items())
    else:
        block_items = list(reversed(blocks_dict.items()))

    skip_processing = True
    for block_name, layers in block_items:
        save_path = info_dict["save_path"]
        cache_dir = os.path.join(save_path, "cache")
        block_save_path = os.path.join(cache_dir, block_name + ".safetensors")
        metadata_save_path = os.path.join(cache_dir, block_name + ".pkl")
        already_processed = os.path.exists(block_save_path) and os.path.exists(
            metadata_save_path
        )
        if already_processed:
            print(f"{block_name} already processed")
            continue
        else:
            skip_processing = False

    if skip_processing:
        results = []
        block_names = list(blocks_dict.keys())
        for block_name in tqdm(block_names, desc="Loading metadata"):
            results.append(process_fp8_block_metadata(block_name, info_dict))
        return results

    with ProcessPoolExecutor(max_workers=num_processes) as exe:
        results = list(
            tqdm(
                exe.map(process_fp8_block, block_items, [info_dict] * len(block_items)),
                total=len(block_items),
                desc="Processing FP8 blocks",
            )
        )
    return results


def process_all_fp8_blocks_parallel_with_recycle(
    blocks_dict, num_processes, reverse=False, info_dict=None
):
    if not reverse:
        block_items = list(blocks_dict.items())
    else:
        block_items = list(reversed(blocks_dict.items()))

    # Spawned workers + recycle per task to prevent SHM accumulation
    ctx = get_context("spawn")
    iterable = [((k, v), info_dict) for k, v in block_items]

    with ctx.Pool(processes=num_processes, maxtasksperchild=1) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_worker, iterable, chunksize=1),
                total=len(block_items),
                desc="Processing FP8 blocks",
            )
        )
    return results


if __name__ == "__main__":
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16

    bytes_per_thread = args.bytes_per_thread
    threads_per_block = args.threads_per_block

    # Load model from local directory
    config = toml.load(args.config)
    if args.repo_id == "black-forest-labs/FLUX.1-dev":
        from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device="cpu",
            model_configs=[
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="flux1-dev.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="text_encoder/model.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="text_encoder_2/",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="ae.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
            ],
        )
        pipe.enable_vram_management(num_persistent_param_in_dit=0)
        model = pipe
    elif args.repo_id == "Wan-AI/Wan2.1-T2V-14B":
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="diffusion_pytorch_model*.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="Wan2.1_VAE.pth",
                    offload_dtype=torch.float8_e4m3fn,
                ),
            ],
        )
        pipe.enable_vram_management(num_persistent_param_in_dit=0)
        model = pipe
    elif args.repo_id == "Wan-AI/Wan2.2-T2V-A14B":
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="Wan2.1_VAE.pth",
                    offload_dtype=torch.float8_e4m3fn,
                ),
            ],
        )
        pipe.enable_vram_management(num_persistent_param_in_dit=0)
        model = pipe
    elif args.repo_id == "Qwen/Qwen-Image":
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="text_encoder/model*.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
                ModelConfig(
                    model_id=args.repo_id,
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    offload_dtype=torch.float8_e4m3fn,
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id=args.repo_id, origin_file_pattern="tokenizer/"
            ),
        )
        pipe.enable_vram_management(num_persistent_param_in_dit=0)
        model = pipe
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.repo_id, torch_dtype=torch_dtype
        )

    dfloat_dir = config["download"]["dfloat_dir"]
    dfloat_repo_id, dfloat_model_id = get_dfloat_model_name_or_path(
        args.repo_id, dfloat_dir
    )
    save_path = dfloat_repo_id
    os.makedirs(save_path, exist_ok=True)

    # Search for FP8 blocks based on regular expression
    model_pattern_dict = get_fp8_pattern_dict(model.__class__.__name__)
    model_bf16_regex = model_pattern_dict["bf16"]
    model_bf16_regex_dict = {}
    for regex in model_bf16_regex:
        model_bf16_regex_dict[regex] = []

    model_fp8_regex = model_pattern_dict["fp8"]
    model_fp8_regex_dict = {}
    for regex in model_fp8_regex:
        model_fp8_regex_dict[regex] = None

    fp8_layers_by_block = {}

    if isinstance(model_fp8_regex, list):
        fp8_blocks = locate_block_by_regex(model, "|".join(model_fp8_regex))
        for block_name, block in fp8_blocks.items():
            temp = find_layers_by_data_type_and_layer_type(
                block,
                target_dtype=torch.float8_e4m3fn,
                layers=[torch.nn.Linear, AutoWrappedLinear, FP8Linear],
                name=block_name,
            )
            fp8_layers_by_block[block_name] = temp
    elif isinstance(model_fp8_regex, dict):
        for regex, selected_names in model_fp8_regex.items():
            fp8_blocks = locate_block_by_regex(model, regex)
            for block_name, block in fp8_blocks.items():
                temp = find_layers_by_selection(block, selected_names, name=block_name)
                # breakpoint()
                fp8_layers_by_block[block_name] = temp

    info_dict = {
        "save_path": save_path,
        "model_fp8_regex": model_fp8_regex,
        "bytes_per_thread": bytes_per_thread,
        "threads_per_block": threads_per_block,
    }

    # Process blocks in parallel (CPU only)
    if args.sequential:
        fp8_results = []
        for block_name, layers in fp8_layers_by_block.items():
            print(f"Processing {block_name}")
            print("-" * 100)
            fp8_results.append(
                process_fp8_block(
                    (block_name, layers),
                    info_dict,
                )
            )
    else:
        fp8_results = process_all_fp8_blocks_parallel(
            fp8_layers_by_block,
            num_processes=args.n_processes,
            reverse=args.reverse,
            info_dict=info_dict,
        )

    # Validate CUDA decoding and update model in main process
    # Create a CSV file if it doesn't exist
    entropy_file = os.path.join(save_path, "entropy.csv")
    if not os.path.exists(entropy_file):
        with open(entropy_file, "w") as f:
            f.write("block_name,entropy\n")

    for result in tqdm(fp8_results, desc="Validating CUDA decoding"):
        block_name = result["block_name"]
        getter = operator.attrgetter(block_name)
        sub_module = getter(model)

        entropy = result["entropy"]
        # Write a line to a CSV file
        with open(entropy_file, "a") as f:
            f.write(f"{block_name},{entropy}\n")

        # CUDA validation
        block_save_path = result["block_save_path"]
        tensors = load_file(block_save_path)
        if args.validate_cuda:
            validate_cuda_decoding_fp8(result, tensors, device)

        # Delete layers from the actual model in main process
        for layer_name in result["layers_to_delete"]:
            if layer_name == block_name:
                assert len(result["layers_to_delete"]) == 1
                if isinstance(sub_module, nn.Linear):
                    delattr(sub_module, "weight")
                elif isinstance(sub_module, nn.Embedding):
                    delattr(sub_module, "weight")
                else:
                    raise Exception("Unsupported block type")
            else:
                parts = layer_name.split(".")
                parent_name = ".".join(parts[:-1])
                attr_name = parts[-1]

                parent_getter = operator.attrgetter(parent_name)
                parent_module = parent_getter(model)
                delattr(parent_module, attr_name)

        # Register buffers
        sub_module.register_buffer("luts", tensors["luts"])
        sub_module.register_buffer("encoded", tensors["encoded"])
        sub_module.register_buffer("packed_other_4bits", tensors["packed_other_4bits"])
        sub_module.register_buffer("output_positions", tensors["output_positions"])
        sub_module.register_buffer("gaps", tensors["gaps"])
        sub_module.register_buffer("split_positions", tensors["split_positions"])

        if args.repo_id in [
            "black-forest-labs/FLUX.1-dev",
            "Wan-AI/Wan2.1-T2V-14B",
            "Wan-AI/Wan2.2-T2V-A14B",
            "Qwen/Qwen-Image",
        ]:
            state_dict = sub_module.state_dict()
            state_dict = {
                f"{block_name}.{key}": value for key, value in state_dict.items()
            }
            save_file(state_dict, os.path.join(save_path, block_name + ".safetensors"))

        # Update regex dict
        layer_names = result["layer_names"]
        matched_regex = result["matched_regex"]

        if matched_regex:
            if model_fp8_regex_dict[matched_regex] is None:
                model_fp8_regex_dict[matched_regex] = layer_names
            else:
                try:
                    assert model_fp8_regex_dict[matched_regex] == layer_names
                except AssertionError:
                    print(
                        f"matched regex: {matched_regex} expected: {model_fp8_regex_dict[matched_regex]} actual: {layer_names}"
                    )
                    raise Exception("Regex mismatch")

    # Drop duplicate in entropy.csv
    df = pd.read_csv(entropy_file)
    df = df.drop_duplicates(subset=["block_name"])
    df = df.sort_values(by="block_name")
    df.to_csv(entropy_file, index=False)

    regex_json = {
        "fp8": model_fp8_regex_dict,
        "bf16": model_bf16_regex_dict,
    }
    print(regex_json)

    if args.save_model:
        if args.repo_id in [
            "black-forest-labs/FLUX.1-dev",
            "Wan-AI/Wan2.1-T2V-14B",
            "Wan-AI/Wan2.2-T2V-A14B",
            "Qwen/Qwen-Image",
        ]:
            dfloat_config = {
                "version": package_version,
                "threads_per_block": threads_per_block,
                "bytes_per_thread": bytes_per_thread,
            }
            with open(os.path.join(save_path, "dfloat_config.json"), "w") as f:
                json.dump(dfloat_config, f, indent=4)
        else:
            model.config.dfloat_config = {
                "version": package_version,
                "threads_per_block": threads_per_block,
                "bytes_per_thread": bytes_per_thread,
            }
            model.save_pretrained(save_path)

            tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
            tokenizer.save_pretrained(save_path)

        with open(os.path.join(save_path, "pattern_dict.json"), "w") as f:
            json.dump(regex_json, f, indent=4)
